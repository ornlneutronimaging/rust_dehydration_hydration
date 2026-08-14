//! Running the correction on a background thread: builds the (points ×
//! bands) hyperspectral matrix from the image stack, runs
//! [`crate::hsnt::hyper_denoise`], and reshapes the result back into a stack
//! of frames. Progress and the final result stream to the UI over a channel.

use crate::hsnt::{hyper_denoise, DatasetType, HsntParams};
use crate::loader::ImageStack;
use crate::nmf::BetaLoss;
use anyhow::Result;
use ndarray::Array2;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

/// The user-facing parameters — the ones the notebook exposes. The rest of
/// [`HsntParams`] keeps the notebook's defaults.
#[derive(Clone, Copy, PartialEq)]
pub struct CorrectionParams {
    pub dataset_type: DatasetType,
    pub num_materials: usize,
    pub beta_loss: BetaLoss,
    pub max_iter: usize,
}

impl Default for CorrectionParams {
    fn default() -> Self {
        Self {
            dataset_type: DatasetType::Attenuation,
            num_materials: 2,
            beta_loss: BetaLoss::Frobenius,
            max_iter: 300,
        }
    }
}

impl CorrectionParams {
    pub fn to_hsnt(self) -> HsntParams {
        HsntParams {
            dataset_type: self.dataset_type,
            num_materials: self.num_materials,
            beta_loss: self.beta_loss,
            max_iter: self.max_iter,
            ..HsntParams::default()
        }
    }
}

pub struct CorrectionOutput {
    /// Corrected frames, same order/shape as the input stack.
    pub frames: Vec<Array2<f32>>,
    /// Per-pixel mean over the corrected frames (the notebook's "integrated
    /// corrected image").
    pub integrated_mean: Array2<f32>,
    pub subspace_dimension: usize,
    pub elapsed_seconds: f64,
}

pub enum CorrectionMsg {
    Progress { stage: String, fraction: f32 },
    Done(Result<CorrectionOutput, String>),
}

/// Spawn the correction thread. Poll the returned receiver each UI frame;
/// store `cancel` and set it to stop the solver at its next iteration.
pub fn start_correction(
    stack: Arc<ImageStack>,
    params: CorrectionParams,
    cancel: Arc<AtomicBool>,
    ctx: egui::Context,
) -> Receiver<CorrectionMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let started = std::time::Instant::now();
        let result = run(&stack, params, &cancel, &tx, &ctx);
        let _ = tx.send(CorrectionMsg::Done(result.map_err(|e| format!("{e:#}")).map(
            |(frames, integrated_mean)| CorrectionOutput {
                frames,
                integrated_mean,
                subspace_dimension: params.to_hsnt().subspace_dimension(),
                elapsed_seconds: started.elapsed().as_secs_f64(),
            },
        )));
        ctx.request_repaint();
    });
    rx
}

type FramesAndMean = (Vec<Array2<f32>>, Array2<f32>);

fn run(
    stack: &ImageStack,
    params: CorrectionParams,
    cancel: &AtomicBool,
    tx: &Sender<CorrectionMsg>,
    ctx: &egui::Context,
) -> Result<FramesAndMean> {
    let report = |stage: &str, fraction: f32| {
        let _ = tx.send(CorrectionMsg::Progress {
            stage: stage.to_owned(),
            fraction,
        });
        ctx.request_repaint();
    };

    report("Preparing data", 0.0);
    let x = stack_to_matrix(stack);
    report("Preparing data", 1.0);

    let mut progress = |stage: &str, fraction: f32| report(stage, fraction);
    let denoised = hyper_denoise(x, &params.to_hsnt(), cancel, &mut progress)?;

    report("Assembling corrected stack", 0.0);
    let out = matrix_to_frames(&denoised, stack.height, stack.width);
    let mean = integrated_mean(&out, stack.height, stack.width);
    report("Assembling corrected stack", 1.0);
    Ok((out, mean))
}

/// (points × bands) matrix: row = pixel (row-major over the frame), column =
/// image index. The image index is the spectral axis — the same layout as
/// the notebook's `swapaxes(raw, 0, 2)` + reshape, up to a pixel ordering
/// that the round-trip undoes.
pub fn stack_to_matrix(stack: &ImageStack) -> Array2<f64> {
    let (h, w, n) = (stack.height, stack.width, stack.frames.len());
    let mut x = Array2::<f64>::zeros((h * w, n));
    for (i, frame) in stack.frames.iter().enumerate() {
        let mut col = x.column_mut(i);
        for (dst, &v) in col.iter_mut().zip(frame.iter()) {
            *dst = f64::from(v);
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
            transposed_on_load: true,
        }
    }

    #[test]
    fn matrix_roundtrip_preserves_pixels() {
        let f0 = Array2::from_shape_fn((3, 4), |(y, x)| (y * 4 + x) as f32);
        let f1 = f0.mapv(|v| v * 10.0);
        let stack = make_stack(vec![f0.clone(), f1.clone()]);
        let x = stack_to_matrix(&stack);
        assert_eq!(x.dim(), (12, 2));
        assert_eq!(x[[5, 0]], 5.0);
        assert_eq!(x[[5, 1]], 50.0);
        let back = matrix_to_frames(&x.mapv(|v| v as f32), 3, 4);
        assert_eq!(back[0], f0);
        assert_eq!(back[1], f1);
    }
}
