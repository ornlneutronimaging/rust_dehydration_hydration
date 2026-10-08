//! Running the correction: hand the image stack to the `mbirtorch.hsnt`
//! command line (see [`crate::hsnt_cli`]) and turn what it writes back into
//! a stack of frames.
//!
//! [`run_correction`] is the synchronous, GUI-free core (also used by the
//! headless `--run` mode); [`start_correction`] wraps it on a background
//! thread streaming progress and log lines to the egui app over a channel.

pub use crate::hsnt_cli::HsntParams as CorrectionParams;
use crate::hsnt_cli::{run_denoise, Artifact, HsntReport, ScratchDir};
use crate::loader::ImageStack;
use anyhow::{Context, Result};
use ndarray::Array2;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

pub struct CorrectionOutput {
    /// Corrected frames, same order as the input stack (spatially binned by
    /// `bin` when this is a preview run).
    pub frames: Vec<Array2<f32>>,
    /// Per-pixel mean over the corrected frames (the notebook's "integrated
    /// corrected image").
    pub integrated_mean: Array2<f32>,
    /// The raw frames the correction actually ran on, kept only when they
    /// differ from the input stack (bin > 1) so the comparison views and
    /// profiles have a like-for-like reference.
    pub binned_raw: Option<Vec<Array2<f32>>>,
    /// Spatial binning factor (1 = full resolution, >1 = preview).
    pub bin: usize,
    pub elapsed_seconds: f64,
    /// What the CLI reported about the fit.
    pub report: HsntReport,
    /// The CLI's by-products (report, dehydrated file, plots, log), for the
    /// export folder.
    pub artifacts: Vec<Artifact>,
    /// Every line the CLI printed.
    pub log: Vec<String>,
    /// The mask the solver saw (binned for previews), when one was set.
    pub mask: Option<Array2<bool>>,
    /// Pixels handed to the solver.
    pub n_pixels: usize,
    /// The component maps as images (0 outside the mask).
    pub maps: Vec<Array2<f32>>,
}

impl CorrectionOutput {
    pub fn is_preview(&self) -> bool {
        self.bin > 1
    }
}

/// Mean-bin every frame over `bin`×`bin` blocks (edges that do not fill a
/// block are dropped). `bin = 1` returns a plain copy.
pub fn bin_frames(frames: &[Array2<f32>], bin: usize) -> Vec<Array2<f32>> {
    if bin <= 1 {
        return frames.to_vec();
    }
    frames
        .iter()
        .map(|f| {
            let (h, w) = (f.nrows() / bin, f.ncols() / bin);
            Array2::from_shape_fn((h, w), |(y, x)| {
                let mut acc = 0.0f32;
                for dy in 0..bin {
                    for dx in 0..bin {
                        acc += f[(y * bin + dy, x * bin + dx)];
                    }
                }
                acc / (bin * bin) as f32
            })
        })
        .collect()
}

/// Synchronous correction of a whole stack, or of the pixels `mask` selects
/// (the stack's size; true = keep — the other pixels are not sent to the
/// solver and keep their raw values). `bin > 1` runs a spatially binned
/// preview. Progress arrives as `(stage, fraction)`, the CLI's output line
/// by line in `log`; setting `cancel` kills the Python process.
pub fn run_correction(
    stack: &ImageStack,
    params: CorrectionParams,
    bin: usize,
    mask: Option<&Array2<bool>>,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
    log: &mut dyn FnMut(&str),
) -> Result<CorrectionOutput> {
    let started = std::time::Instant::now();

    progress("Preparing data", 0.0);
    let bin = bin.max(1);
    if stack.frames.is_empty() {
        anyhow::bail!("empty stack");
    }
    if let Some(m) = mask
        && m.dim() != (stack.height, stack.width)
    {
        anyhow::bail!(
            "the mask is {}×{} px, the stack {}×{}",
            m.ncols(),
            m.nrows(),
            stack.width,
            stack.height
        );
    }
    let binned_raw = (bin > 1).then(|| bin_frames(&stack.frames, bin));
    let work_frames: &[Array2<f32>] = binned_raw.as_deref().unwrap_or(&stack.frames);
    let (h, w) = work_frames[0].dim();
    let work_mask = mask.map(|m| crate::mask::bin_mask(m, bin));

    let outcome = run_denoise(work_frames, work_mask.as_ref(), &params, cancel, progress, log)?;

    progress("Assembling corrected stack", 0.99);
    let integrated_mean = integrated_mean(&outcome.frames, h, w);
    let mut artifacts = outcome.artifacts;
    artifacts.extend(image_artifacts(&outcome.maps, work_mask.as_ref(), stack.orientation, log));
    progress("Assembling corrected stack", 1.0);

    Ok(CorrectionOutput {
        frames: outcome.frames,
        integrated_mean,
        binned_raw,
        bin,
        elapsed_seconds: started.elapsed().as_secs_f64(),
        report: outcome.report,
        artifacts,
        log: outcome.log,
        mask: work_mask,
        n_pixels: outcome.n_pixels,
        maps: outcome.maps,
    })
}

/// The component maps (`hsnt_map_<i>.tif`, float32) and the mask
/// (`mask.tif`, 8-bit, 255 = selected) as TIFF files in the on-disk
/// orientation of the input, encoded through a scratch folder.
fn image_artifacts(
    maps: &[Array2<f32>],
    mask: Option<&Array2<bool>>,
    orientation: crate::loader::Orientation,
    log: &mut dyn FnMut(&str),
) -> Vec<Artifact> {
    let mut out = Vec::new();
    let encode = || -> Result<Vec<Artifact>> {
        let scratch = ScratchDir::create()?;
        let mut files = Vec::new();
        for (i, map) in maps.iter().enumerate() {
            let name = format!("hsnt_map_{}.tif", i + 1);
            let path = scratch.path().join(&name);
            crate::export::write_f32_tiff(&path, map, orientation)?;
            files.push(Artifact { name, bytes: std::fs::read(&path).context("read back the map")? });
        }
        if let Some(m) = mask {
            let path = scratch.path().join("mask.tif");
            crate::mask::save_file(&path, m, orientation)?;
            files.push(Artifact { name: "mask.tif".to_owned(), bytes: std::fs::read(&path).context("read back the mask")? });
        }
        Ok(files)
    };
    match encode() {
        Ok(files) => out.extend(files),
        Err(e) => log(&format!("[dehydration_hydration] cannot encode the map / mask images: {e:#}")),
    }
    out
}

pub enum CorrectionMsg {
    Progress { stage: String, fraction: f32 },
    /// One line of the CLI's output.
    Log(String),
    Done(Result<CorrectionOutput, String>),
}

/// Spawn the correction thread for the GUI. Poll the returned receiver each
/// frame; store `cancel` and set it to stop the run.
pub fn start_correction(
    stack: Arc<ImageStack>,
    params: CorrectionParams,
    bin: usize,
    mask: Option<Arc<Array2<bool>>>,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
) -> Receiver<CorrectionMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let tx_progress = tx.clone();
        let ctx_progress = ctx.clone();
        let mut progress = |stage: &str, fraction: f32| {
            let _ = tx_progress.send(CorrectionMsg::Progress {
                stage: stage.to_owned(),
                fraction,
            });
            ctx_progress.request_repaint();
        };
        let tx_log = tx.clone();
        let ctx_log = ctx.clone();
        let mut log = |line: &str| {
            let _ = tx_log.send(CorrectionMsg::Log(line.to_owned()));
            ctx_log.request_repaint();
        };
        let result = run_correction(&stack, params, bin, mask.as_deref(), &cancel, &mut progress, &mut log);
        let _ = tx.send(CorrectionMsg::Done(result.map_err(|e| format!("{e:#}"))));
        ctx.request_repaint();
    });
    rx
}

fn integrated_mean(frames: &[Array2<f32>], h: usize, w: usize) -> Array2<f32> {
    let mut acc = Array2::<f32>::zeros((h, w));
    for f in frames {
        acc += f;
    }
    if !frames.is_empty() {
        acc /= frames.len() as f32;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_averages_blocks_and_drops_ragged_edges() {
        let f = Array2::from_shape_fn((5, 6), |(y, x)| (y * 6 + x) as f32);
        let binned = bin_frames(&[f], 2);
        assert_eq!(binned[0].dim(), (2, 3)); // 5→2 rows (last dropped), 6→3 cols
        // Block (0,0): mean of values at (0,0),(0,1),(1,0),(1,1) = (0+1+6+7)/4
        assert_eq!(binned[0][(0, 0)], 3.5);
    }

    #[test]
    fn integrated_mean_averages_frames() {
        let a = Array2::from_elem((2, 2), 1.0f32);
        let b = Array2::from_elem((2, 2), 3.0f32);
        assert_eq!(integrated_mean(&[a, b], 2, 2), Array2::from_elem((2, 2), 2.0f32));
    }
}
