//! Running the correction: build the (points × bands) hyperspectral matrix
//! from the image stack, run [`crate::hsnt::hyper_denoise`], and reshape the
//! result back into a stack of frames.
//!
//! [`run_correction`] is the synchronous, GUI-free core (also used by the
//! headless `--run` mode); [`start_correction`] wraps it on a background
//! thread streaming progress to the egui app over a channel.

use crate::hsnt::{hyper_denoise, DatasetType, HsntParams};
use crate::loader::ImageStack;
use crate::nmf::BetaLoss;
use anyhow::Result;
use ndarray::Array2;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::Receiver;
use std::sync::Arc;

/// The user's "number of materials" is multiplied by this before it is
/// handed to the correction, so the factorization keeps a comfortable
/// number of degrees of freedom. Surfaced in the GUI, the CLI help and the
/// provenance file — `num_materials` everywhere else stays the user's value.
pub const MATERIALS_FACTOR: usize = 4;

/// The user-facing parameters — the ones the notebook exposes. The rest of
/// [`HsntParams`] keeps the notebook's defaults.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CorrectionParams {
    pub dataset_type: DatasetType,
    pub num_materials: usize,
    /// Multiplier on the material count (after [`MATERIALS_FACTOR`]) giving
    /// the NMF subspace dimension: `safety_factor × MATERIALS_FACTOR × N`,
    /// capped at the number of images.
    pub safety_factor: f64,
    pub beta_loss: BetaLoss,
    pub max_iter: usize,
}

impl Default for CorrectionParams {
    fn default() -> Self {
        Self {
            dataset_type: DatasetType::Attenuation,
            num_materials: 2,
            safety_factor: 16.0,
            beta_loss: BetaLoss::Frobenius,
            max_iter: 300,
        }
    }
}

impl CorrectionParams {
    pub fn to_hsnt(self) -> HsntParams {
        HsntParams {
            dataset_type: self.dataset_type,
            num_materials: self.num_materials * MATERIALS_FACTOR,
            safety_factor: self.safety_factor,
            beta_loss: self.beta_loss,
            max_iter: self.max_iter,
            ..HsntParams::default()
        }
    }
}

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
    pub subspace_dimension: usize,
    pub elapsed_seconds: f64,
    /// Pixels that went into the NMF (all of them without a mask); at the
    /// working resolution.
    pub pixels_corrected: usize,
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

/// Synchronous correction of a whole stack, or of the pixels `mask`
/// selects (the others keep their input values). `bin > 1` runs a spatially
/// binned preview. Progress arrives as `(stage, fraction)`; setting `cancel`
/// aborts at the solver's next iteration.
pub fn run_correction(
    stack: &ImageStack,
    params: CorrectionParams,
    bin: usize,
    mask: Option<&Array2<bool>>,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<CorrectionOutput> {
    let started = std::time::Instant::now();

    progress("Preparing data", 0.0);
    let bin = bin.max(1);
    let (work_frames, binned_raw) = if bin > 1 {
        let binned = bin_frames(&stack.frames, bin);
        (binned.clone(), Some(binned))
    } else {
        (stack.frames.clone(), None)
    };
    let (h, w) = match work_frames.first() {
        Some(f) => (f.nrows(), f.ncols()),
        None => anyhow::bail!("empty stack"),
    };
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
    let work_mask = mask.map(|m| crate::mask::bin_mask(m, bin));
    let selected: Option<Vec<usize>> = work_mask
        .as_ref()
        .map(|m| m.iter().enumerate().filter(|(_, keep)| **keep).map(|(i, _)| i).collect());
    if selected.as_ref().is_some_and(|s| s.is_empty()) {
        anyhow::bail!("the mask selects no pixel");
    }
    let x = match &selected {
        Some(rows) => selected_rows_to_matrix(&work_frames, rows),
        None => frames_to_matrix(&work_frames, h, w),
    };
    let pixels_corrected = x.nrows();
    progress("Preparing data", 1.0);
    let denoised = hyper_denoise(x, &params.to_hsnt(), cancel, progress)?;
    progress("Assembling corrected stack", 0.0);
    let frames = match &selected {
        // Only the selected pixels change; the others keep their input values.
        Some(rows) => {
            let mut frames = work_frames.clone();
            for (i, frame) in frames.iter_mut().enumerate() {
                let col = denoised.column(i);
                let flat = frame.as_slice_mut().expect("standard layout");
                for (k, &p) in rows.iter().enumerate() {
                    flat[p] = col[k];
                }
            }
            frames
        }
        None => matrix_to_frames(&denoised, h, w),
    };
    let integrated_mean = integrated_mean(&frames, h, w);
    progress("Assembling corrected stack", 1.0);
    Ok(CorrectionOutput {
        frames,
        integrated_mean,
        binned_raw,
        bin,
        subspace_dimension: params.to_hsnt().subspace_dimension(),
        elapsed_seconds: started.elapsed().as_secs_f64(),
        pixels_corrected,
    })
}

pub enum CorrectionMsg {
    Progress { stage: String, fraction: f32 },
    Done(Result<CorrectionOutput, String>),
}

/// Spawn the correction thread for the GUI. Poll the returned receiver each
/// frame; store `cancel` and set it to stop the solver.
pub fn start_correction(
    stack: Arc<ImageStack>,
    params: CorrectionParams,
    bin: usize,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
) -> Receiver<CorrectionMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut progress = |stage: &str, fraction: f32| {
            let _ = tx.send(CorrectionMsg::Progress {
                stage: stage.to_owned(),
                fraction,
            });
            ctx.request_repaint();
        };
        let result = run_correction(&stack, params, bin, None, &cancel, &mut progress);
        let _ = tx.send(CorrectionMsg::Done(result.map_err(|e| format!("{e:#}"))));
        ctx.request_repaint();
    });
    rx
}

/// (points × bands) matrix: row = pixel (row-major over the frame), column =
/// image index. The image index is the spectral axis — the same layout as
/// the notebook's `swapaxes(raw, 0, 2)` + reshape, up to a pixel ordering
/// that the round-trip undoes.
pub fn frames_to_matrix(frames: &[Array2<f32>], h: usize, w: usize) -> Array2<f64> {
    let n = frames.len();
    let mut x = Array2::<f64>::zeros((h * w, n));
    for (i, frame) in frames.iter().enumerate() {
        let mut col = x.column_mut(i);
        for (dst, &v) in col.iter_mut().zip(frame.iter()) {
            *dst = f64::from(v);
        }
    }
    x
}

/// The (selected points × bands) matrix of the pixels at the row-major
/// indices `rows` (what the mask keeps), in that order.
fn selected_rows_to_matrix(frames: &[Array2<f32>], rows: &[usize]) -> Array2<f64> {
    let n = frames.len();
    let mut x = Array2::<f64>::zeros((rows.len(), n));
    for (i, frame) in frames.iter().enumerate() {
        let flat = frame.as_slice().expect("standard layout");
        let mut col = x.column_mut(i);
        for (dst, &p) in col.iter_mut().zip(rows) {
            *dst = f64::from(flat[p]);
        }
    }
    x
}

fn matrix_to_frames(x: &Array2<f32>, h: usize, w: usize) -> Vec<Array2<f32>> {
    let n = x.ncols();
    (0..n)
        .map(|i| {
            let col = x.column(i);
            Array2::from_shape_fn((h, w), |(y, xx)| col[y * w + xx])
        })
        .collect()
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
    use std::path::PathBuf;

    fn make_stack(frames: Vec<Array2<f32>>) -> ImageStack {
        let (h, w) = (frames[0].nrows(), frames[0].ncols());
        let sources = frames.iter().map(|_| PathBuf::from("x.tif")).collect();
        ImageStack {
            frames,
            width: w,
            height: h,
            sources,
            detector: Default::default(),
            orientation: crate::loader::Orientation::Transpose,
            nonfinite_fixed: 0,
        }
    }

    #[test]
    fn matrix_roundtrip_preserves_pixels() {
        let f0 = Array2::from_shape_fn((3, 4), |(y, x)| (y * 4 + x) as f32);
        let f1 = f0.mapv(|v| v * 10.0);
        let stack = make_stack(vec![f0.clone(), f1.clone()]);
        let x = frames_to_matrix(&stack.frames, 3, 4);
        assert_eq!(x.dim(), (12, 2));
        assert_eq!(x[[5, 0]], 5.0);
        assert_eq!(x[[5, 1]], 50.0);
        let back = matrix_to_frames(&x.mapv(|v| v as f32), 3, 4);
        assert_eq!(back[0], f0);
        assert_eq!(back[1], f1);
    }

    #[test]
    fn binning_averages_blocks_and_drops_ragged_edges() {
        let f = Array2::from_shape_fn((5, 6), |(y, x)| (y * 6 + x) as f32);
        let binned = bin_frames(&[f], 2);
        assert_eq!(binned[0].dim(), (2, 3)); // 5→2 rows (last dropped), 6→3 cols
        // Block (0,0): mean of values at (0,0),(0,1),(1,0),(1,1) = (0+1+6+7)/4
        assert_eq!(binned[0][(0, 0)], 3.5);
    }

    #[test]
    fn preview_run_returns_binned_frames_and_reference() {
        let frames: Vec<Array2<f32>> = (0..6)
            .map(|i| {
                Array2::from_shape_fn((8, 8), |(y, x)| {
                    1.0 + (i as f32) * 0.1 + ((y + x) as f32) * 0.01
                })
            })
            .collect();
        let stack = make_stack(frames);
        let cancel = AtomicBool::new(false);
        let out = run_correction(
            &stack,
            CorrectionParams {
                max_iter: 60,
                ..Default::default()
            },
            2,
            None,
            &cancel,
            &mut |_, _| {},
        )
        .unwrap();
        assert!(out.is_preview());
        assert_eq!(out.pixels_corrected, 16);
        assert_eq!(out.frames[0].dim(), (4, 4));
        assert_eq!(out.binned_raw.as_ref().unwrap()[0].dim(), (4, 4));
        assert_eq!(out.frames.len(), 6);
    }

    #[test]
    fn masked_run_corrects_only_the_selected_pixels() {
        let frames: Vec<Array2<f32>> = (0..6)
            .map(|i| {
                Array2::from_shape_fn((8, 8), |(y, x)| {
                    // Left half: a smooth sample; right half: zero fill.
                    if x < 4 { 1.0 + (i as f32) * 0.1 + ((y + x) as f32) * 0.01 } else { 0.0 }
                })
            })
            .collect();
        let stack = make_stack(frames.clone());
        let mask = Array2::from_shape_fn((8, 8), |(_, x)| x < 4);
        let cancel = AtomicBool::new(false);
        let params = CorrectionParams { max_iter: 60, ..Default::default() };
        let out = run_correction(&stack, params, 1, Some(&mask), &cancel, &mut |_, _| {}).unwrap();
        assert_eq!(out.pixels_corrected, 32);
        assert_eq!(out.frames.len(), 6);
        for (i, f) in out.frames.iter().enumerate() {
            for y in 0..8 {
                for x in 4..8 {
                    assert_eq!(f[(y, x)], frames[i][(y, x)], "outside the mask keeps its value");
                }
            }
        }
        // A wrong-sized mask is refused.
        let bad = Array2::from_elem((4, 4), true);
        assert!(run_correction(&stack, params, 1, Some(&bad), &cancel, &mut |_, _| {}).is_err());
        // An empty mask too.
        let none = Array2::from_elem((8, 8), false);
        assert!(run_correction(&stack, params, 1, Some(&none), &cancel, &mut |_, _| {}).is_err());
    }
}
