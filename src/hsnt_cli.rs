//! The correction engine of the beta: Harel Dor's `mbirtorch.hsnt` package
//! (branch `hsnt` of harel55/mbirtorch, squashed into cabouman/mbirtorch's
//! `prerelease` on 2026-10-06), driven through its command line
//!
//! ```text
//! python -m mbirtorch.hsnt denoise <input.h5> -o <dir> [options]
//! ```
//!
//! in the pixi environment of the `mbirtorch_hsnt` checkout next to this
//! repository. The algorithm is no longer the least-squares NMF of the
//! production tool but the maximum-likelihood factorization X = W·H of the
//! attenuation under the Poisson likelihood of the counts (the "NNAL" fit),
//! with the number of components estimated by likelihood-ratio tests when
//! it is not given. See `docs/source/usr_hsnt.rst` in the checkout.
//!
//! Hand-off: the stack loaded (and oriented) by this program is written to
//! a scratch HDF5 file in the hsnt layout (`data` of shape (rows, cols,
//! bins), float32, spectral axis last), the CLI writes `<stem>_denoised.h5`
//! (`data` of shape (1, rows, cols, bins)) plus the dehydrated file, a JSON
//! report and two PNG plots; the denoised data is read back into frames and
//! the small by-products are kept in memory for the export folder. The
//! scratch folder is removed afterwards.

use anyhow::{bail, Context, Result};
use ndarray::{s, Array2, Array3, Ix4};
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Duration;

/// Interpreter of the `cuda` pixi environment of the hsnt checkout
/// (`pixi install -e cuda` in [`MBIRTORCH_CHECKOUT`]): torch with CUDA 13,
/// mbirtorch (editable, so the checked-out branch is what runs), h5py,
/// matplotlib. Overridden by [`PYTHON_ENV_VAR`].
pub const DEFAULT_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/mbirtorch_hsnt/.pixi/envs/cuda/bin/python";
/// The mbirtorch clone the beta runs: branch `hsnt` tracking
/// `harel` = https://github.com/harel55/mbirtorch (origin = cabouman).
pub const MBIRTORCH_CHECKOUT: &str = "/SNS/VENUS/shared/software/git/mbirtorch_hsnt";
pub const MBIRTORCH_BRANCH: &str = "hsnt (harel55/mbirtorch)";
/// Environment variable naming another Python interpreter with mbirtorch.
pub const PYTHON_ENV_VAR: &str = "DEHY_HSNT_PYTHON";
/// Environment variable naming the folder the exchange files are written
/// to (default: the system temporary folder, i.e. `$TMPDIR` or `/tmp`).
pub const SCRATCH_ENV_VAR: &str = "DEHY_HSNT_SCRATCH";

/// The interpreter that runs the hsnt command line.
pub fn python() -> PathBuf {
    std::env::var_os(PYTHON_ENV_VAR)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_PYTHON))
}

/// Folder the per-run exchange folders are created in.
pub fn scratch_root() -> PathBuf {
    if let Some(dir) = std::env::var_os(SCRATCH_ENV_VAR).filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    let user = std::env::var("USER").unwrap_or_else(|_| "user".to_owned());
    std::env::temp_dir().join(format!("dehydration_hydration_{user}"))
}

/// Short commit hash + branch of the mbirtorch checkout, read from its
/// `.git` folder (no git subprocess), for the About dialog and the
/// provenance file. `None` when the checkout is not there.
pub fn mbirtorch_commit() -> Option<String> {
    mbirtorch_commit_in(Path::new(MBIRTORCH_CHECKOUT))
}

fn mbirtorch_commit_in(checkout: &Path) -> Option<String> {
    let git = checkout.join(".git");
    let head = std::fs::read_to_string(git.join("HEAD")).ok()?;
    let head = head.trim();
    let (hash, branch) = if let Some(reference) = head.strip_prefix("ref: ") {
        let branch = reference.rsplit('/').next().unwrap_or(reference).to_owned();
        let hash = std::fs::read_to_string(git.join(reference))
            .ok()
            .map(|s| s.trim().to_owned())
            .or_else(|| {
                // Packed refs: "<hash> <ref>" lines.
                let packed = std::fs::read_to_string(git.join("packed-refs")).ok()?;
                packed
                    .lines()
                    .find(|l| l.ends_with(reference))
                    .and_then(|l| l.split_whitespace().next())
                    .map(str::to_owned)
            })?;
        (hash, branch)
    } else {
        (head.to_owned(), "detached".to_owned())
    };
    let short: String = hash.chars().take(7).collect();
    Some(format!("{short} ({branch})"))
}

// ----- parameters ------------------------------------------------------------

/// What the loaded values are (`--input-type`). `Auto` lets the CLI infer
/// it: non-negative values up to 1.05 are transmissions, negative values
/// attenuations; non-negative values above 1.05 cannot be told apart and
/// the run fails with a message asking for an explicit type.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum InputType {
    Auto,
    Transmission,
    Attenuation,
}

impl InputType {
    pub const ALL: [InputType; 3] = [InputType::Transmission, InputType::Attenuation, InputType::Auto];

    pub fn label(self) -> &'static str {
        match self {
            InputType::Auto => "auto",
            InputType::Transmission => "transmission",
            InputType::Attenuation => "attenuation",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(InputType::Auto),
            "transmission" => Some(InputType::Transmission),
            "attenuation" => Some(InputType::Attenuation),
            _ => None,
        }
    }
}

/// Number of components (`--rank`): estimated by likelihood-ratio tests, or
/// given. About the number of distinct materials in the field of view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rank {
    Auto,
    Fixed(usize),
}

impl Rank {
    pub fn label(self) -> String {
        match self {
            Rank::Auto => "auto".to_owned(),
            Rank::Fixed(n) => n.to_string(),
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.eq_ignore_ascii_case("auto") {
            return Some(Rank::Auto);
        }
        s.parse::<usize>().ok().filter(|n| *n >= 1).map(Rank::Fixed)
    }
}

/// How the component spectra are estimated (`--spectra`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Spectra {
    /// The spectra that best fit the measured counts (default).
    Mle,
    /// Re-estimate without the low-dose bias of the non-negative fit;
    /// worth it with many pixels.
    Unconstrained,
    /// Decide which components each pixel contains, then refit; zeroes the
    /// background of the maps. Needs the dose.
    Support,
}

impl Spectra {
    pub const ALL: [Spectra; 3] = [Spectra::Mle, Spectra::Unconstrained, Spectra::Support];

    pub fn label(self) -> &'static str {
        match self {
            Spectra::Mle => "mle",
            Spectra::Unconstrained => "unconstrained",
            Spectra::Support => "support",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mle" => Some(Spectra::Mle),
            "unconstrained" => Some(Spectra::Unconstrained),
            "support" => Some(Spectra::Support),
            _ => None,
        }
    }
}

/// Compute device (`--device`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Device {
    /// CUDA when available, else the CPU.
    Auto,
    Cpu,
    Cuda(u8),
}

impl Device {
    pub fn label(self) -> String {
        match self {
            Device::Auto => "auto".to_owned(),
            Device::Cpu => "cpu".to_owned(),
            Device::Cuda(n) => format!("cuda:{n}"),
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "auto" => Some(Device::Auto),
            "cpu" => Some(Device::Cpu),
            "cuda" => Some(Device::Cuda(0)),
            _ => s
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<u8>().ok())
                .map(Device::Cuda),
        }
    }
}

/// Full solve on the device or streamed by chunks of pixels (`--mode`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SolveMode {
    Auto,
    Full,
    Stream,
}

impl SolveMode {
    pub const ALL: [SolveMode; 3] = [SolveMode::Auto, SolveMode::Full, SolveMode::Stream];

    pub fn label(self) -> &'static str {
        match self {
            SolveMode::Auto => "auto",
            SolveMode::Full => "full",
            SolveMode::Stream => "stream",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(SolveMode::Auto),
            "full" => Some(SolveMode::Full),
            "stream" => Some(SolveMode::Stream),
            _ => None,
        }
    }
}

/// `torch.compile` of the solver kernels (`--compile`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Compile {
    Auto,
    On,
    Off,
}

impl Compile {
    pub const ALL: [Compile; 3] = [Compile::Auto, Compile::On, Compile::Off];

    pub fn label(self) -> &'static str {
        match self {
            Compile::Auto => "auto",
            Compile::On => "on",
            Compile::Off => "off",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Compile::Auto),
            "on" => Some(Compile::On),
            "off" => Some(Compile::Off),
            _ => None,
        }
    }
}

/// The user-facing parameters of a run — the options of
/// `mbirtorch-hsnt denoise` this program exposes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct HsntParams {
    pub input_type: InputType,
    pub rank: Rank,
    /// Largest number of components the automatic estimate considers.
    pub max_rank: usize,
    pub spectra: Spectra,
    /// Open-beam counts per pixel and bin, when known: gives the fit a
    /// chi-square against the Poisson noise, and is required by
    /// [`Spectra::Support`].
    pub dose: Option<f64>,
    pub device: Device,
    pub mode: SolveMode,
    /// Solver step cap of a full solve.
    pub max_steps: usize,
    /// Relative loss change per step below which the solve stops.
    pub rel_tol: f64,
    /// Stream mode: polish passes over the data.
    pub max_passes: usize,
    pub compile: Compile,
}

impl Default for HsntParams {
    fn default() -> Self {
        Self {
            // The stacks this tool loads are normalized (autoreduce /
            // NeuNorm) images: transmission ratios.
            input_type: InputType::Transmission,
            rank: Rank::Auto,
            max_rank: 6,
            spectra: Spectra::Mle,
            dose: None,
            device: Device::Auto,
            mode: SolveMode::Auto,
            max_steps: 1000,
            rel_tol: 1e-8,
            max_passes: 5,
            compile: Compile::Auto,
        }
    }
}

impl HsntParams {
    /// Checks the CLI would reject anyway, reported before anything is
    /// written.
    pub fn validate(&self) -> Result<()> {
        if self.spectra == Spectra::Support && self.dose.is_none() {
            bail!("the 'support' spectra estimator needs the dose (open-beam counts per pixel and bin)");
        }
        if let Some(d) = self.dose
            && !(d.is_finite() && d > 0.0)
        {
            bail!("the dose must be a positive number (got {d})");
        }
        if self.max_rank < 1 {
            bail!("max rank must be at least 1");
        }
        if let Rank::Fixed(n) = self.rank
            && n < 1
        {
            bail!("the rank must be at least 1");
        }
        if self.max_steps < 1 {
            bail!("max steps must be at least 1");
        }
        if !(self.rel_tol.is_finite() && self.rel_tol >= 0.0) {
            bail!("rel tol must be a non-negative number");
        }
        Ok(())
    }

    /// The options after `denoise <input> -o <dir>`.
    pub fn cli_args(&self) -> Vec<String> {
        let mut args = vec![
            "--input-type".to_owned(),
            self.input_type.label().to_owned(),
            "--rank".to_owned(),
            self.rank.label(),
            "--max-rank".to_owned(),
            self.max_rank.to_string(),
            "--spectra".to_owned(),
            self.spectra.label().to_owned(),
            "--device".to_owned(),
            self.device.label(),
            "--mode".to_owned(),
            self.mode.label().to_owned(),
            "--max-steps".to_owned(),
            self.max_steps.to_string(),
            "--rel-tol".to_owned(),
            format!("{:e}", self.rel_tol),
            "--max-passes".to_owned(),
            self.max_passes.to_string(),
            "--compile".to_owned(),
            self.compile.label().to_owned(),
        ];
        if let Some(d) = self.dose {
            args.push("--dose".to_owned());
            args.push(format!("{d}"));
        }
        args.push("-v".to_owned());
        args
    }

    /// One-line human summary (status bar, logs).
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{} data, rank {}, spectra {}, device {}, {} solve",
            self.input_type.label(),
            self.rank.label(),
            self.spectra.label(),
            self.device.label(),
            self.mode.label()
        );
        if let Some(d) = self.dose {
            s.push_str(&format!(", dose {d}"));
        }
        s
    }
}

// ----- results ---------------------------------------------------------------

/// A small by-product of the run kept in memory (the dehydrated file, the
/// JSON report, the PNG plots, the log), written into the export folder.
#[derive(Clone, Debug)]
pub struct Artifact {
    pub name: String,
    pub bytes: Vec<u8>,
}

/// What the run's JSON report and final line say about the fit.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct HsntReport {
    pub rank: Option<u64>,
    /// How the rank was obtained ("rank 2 estimated by likelihood-ratio
    /// tests (…)", "rank 3 given").
    pub rank_note: String,
    /// "full" or "stream".
    pub mode: String,
    pub spectra: String,
    pub input_type: String,
    pub dose: Option<f64>,
    pub steps: Option<u64>,
    pub solve_seconds: Option<f64>,
    pub loss: Option<f64>,
    pub reduced_chi2: Option<f64>,
    pub relative_residual: Option<f64>,
    pub memory_plan: String,
    pub gpu_peak_gib: Option<f64>,
    /// The data checks, as (level, message).
    pub checks: Vec<(String, String)>,
    /// WARNING lines of the run's log.
    pub warnings: Vec<String>,
    /// The CLI's final "done: …" line.
    pub summary_line: String,
}

impl HsntReport {
    /// Parse `<stem>_report.json`. Missing fields stay at their defaults.
    pub fn parse(json: &str) -> Result<Self> {
        let doc: serde_json::Value = serde_json::from_str(json).context("parse the hsnt report")?;
        let result = doc.get("result").cloned().unwrap_or(serde_json::Value::Null);
        let fit = result.get("fit").cloned().unwrap_or(serde_json::Value::Null);
        let text = |v: &serde_json::Value, key: &str| {
            v.get(key).and_then(|x| x.as_str()).unwrap_or_default().to_owned()
        };
        let mut checks = Vec::new();
        if let Some(list) = doc.get("checks").and_then(|c| c.as_array()) {
            for c in list {
                checks.push((text(c, "level"), text(c, "message")));
            }
        }
        Ok(Self {
            rank: result.get("rank").and_then(|v| v.as_u64()),
            rank_note: text(&result, "rank_note"),
            mode: text(&result, "mode"),
            spectra: text(&result, "spectra"),
            input_type: text(&doc, "input_type"),
            dose: doc.get("dose").and_then(|v| v.as_f64()),
            steps: result.get("steps").and_then(|v| v.as_u64()),
            solve_seconds: result.get("solve_seconds").and_then(|v| v.as_f64()),
            loss: result.get("loss_final").and_then(|v| v.as_f64()),
            reduced_chi2: fit.get("reduced_chi2").and_then(|v| v.as_f64()),
            relative_residual: fit.get("relative_residual").and_then(|v| v.as_f64()),
            memory_plan: text(&result, "memory_plan"),
            gpu_peak_gib: result.get("gpu_peak_gib").and_then(|v| v.as_f64()),
            checks,
            warnings: Vec::new(),
            summary_line: String::new(),
        })
    }

    /// The report as JSON (for the provenance file).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "rank": self.rank,
            "rank_note": self.rank_note,
            "mode": self.mode,
            "spectra": self.spectra,
            "input_type": self.input_type,
            "dose": self.dose,
            "steps": self.steps,
            "solve_seconds": self.solve_seconds,
            "loss_final": self.loss,
            "reduced_chi2": self.reduced_chi2,
            "relative_residual": self.relative_residual,
            "memory_plan": self.memory_plan,
            "gpu_peak_gib": self.gpu_peak_gib,
            "checks": self.checks.iter().map(|(l, m)| serde_json::json!({"level": l, "message": m})).collect::<Vec<_>>(),
            "warnings": self.warnings,
            "summary": self.summary_line,
        })
    }

    /// Verdict on the reduced chi-square, as the CLI words it.
    pub fn chi2_verdict(&self) -> Option<&'static str> {
        let chi2 = self.reduced_chi2?;
        Some(if (0.8..=1.3).contains(&chi2) {
            "at the Poisson noise level"
        } else if chi2 > 1.3 {
            "above the noise level: the rank may be too small, the model misspecified or the dose overestimated"
        } else {
            "below the noise level: the dose is underestimated, or the noise model overstates the variance"
        })
    }
}

/// Everything a run returns.
pub struct RunOutcome {
    /// Denoised frames, same order and size as the input frames. With a
    /// mask, the pixels outside it keep their input values.
    pub frames: Vec<Array2<f32>>,
    pub report: HsntReport,
    pub artifacts: Vec<Artifact>,
    /// Every line the CLI printed (stderr log + stdout), in order.
    pub log: Vec<String>,
    /// The component maps as images (0 outside the mask), one per component.
    pub maps: Vec<Array2<f32>>,
    /// Pixels handed to the solver.
    pub n_pixels: usize,
}

// ----- exchange files --------------------------------------------------------

/// Rows per block when the stack is streamed to / from the exchange file:
/// about 64 MiB of float32 per block, rounded to a multiple of `align` rows
/// (the chunk height of the data set being read, so no chunk is visited
/// twice; HDF5 re-reads a partially written chunk, which made row blocks
/// smaller than the chunk height crawl).
fn rows_per_block(w: usize, k: usize, align: usize) -> usize {
    let align = align.max(1);
    let rows = ((64usize << 20) / (w.max(1) * k.max(1) * 4)).clamp(1, 256);
    (rows / align).max(1) * align
}

/// Write `frames` (each (h, w)) as the hsnt layout: `data` of shape (h, w,
/// K) float32, chunked along rows and bins the way the CLI reads them (bins
/// k0:k1 over all pixels). Progress is the fraction of rows written.
pub fn write_input_h5(
    path: &Path,
    frames: &[Array2<f32>],
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    let k = frames.len();
    let Some(first) = frames.first() else {
        bail!("empty stack");
    };
    let (h, w) = first.dim();
    let file = hdf5::File::create(path).with_context(|| format!("create {}", path.display()))?;
    // One row block = one row of chunks, so every chunk is written once,
    // whole; 16 bins per chunk, since the CLI reads bins k0:k1 over all
    // pixels.
    let rb = rows_per_block(w, k, 1).min(h);
    let ds = file
        .new_dataset::<f32>()
        .chunk((rb, w, k.min(16)))
        .shape((h, w, k))
        .create("data")
        .context("create the data set")?;
    let mut r0 = 0;
    while r0 < h {
        let r1 = (r0 + rb).min(h);
        let mut block = Array3::<f32>::zeros((r1 - r0, w, k));
        for (i, frame) in frames.iter().enumerate() {
            let src = frame.slice(s![r0..r1, ..]);
            let mut dst = block.slice_mut(s![.., .., i]);
            dst.assign(&src);
        }
        ds.write_slice(&block, s![r0..r1, .., ..])
            .with_context(|| format!("write rows {r0}..{r1}"))?;
        progress(r1 as f32 / h as f32);
        r0 = r1;
    }
    Ok(())
}

/// Read `<stem>_denoised.h5` (`data` of shape (1, h, w, K) or (h, w, K))
/// back into K frames of (h, w). Progress is the fraction of rows read.
pub fn read_denoised_h5(path: &Path, progress: &mut dyn FnMut(f32)) -> Result<Vec<Array2<f32>>> {
    let file = hdf5::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let ds = file.dataset("data").context("no 'data' set in the denoised file")?;
    let shape = ds.shape();
    let (views, h, w, k) = match shape.as_slice() {
        [v, h, w, k] => (*v, *h, *w, *k),
        [h, w, k] => (1, *h, *w, *k),
        other => bail!("unexpected denoised data shape {other:?}"),
    };
    if views != 1 {
        bail!("the denoised file holds {views} views; this program writes one");
    }
    let mut frames: Vec<Array2<f32>> = (0..k).map(|_| Array2::<f32>::zeros((h, w))).collect();
    // Read whole rows of chunks (the CLI writes chunks of 64 rows).
    let chunk_rows = ds
        .chunk()
        .and_then(|c| c.get(c.len() - 3).copied())
        .unwrap_or(1)
        .clamp(1, h.max(1));
    let rb = rows_per_block(w, k, chunk_rows);
    let mut r0 = 0;
    while r0 < h {
        let r1 = (r0 + rb).min(h);
        let block = if shape.len() == 4 {
            ds.read_slice::<f32, _, Ix4>(s![0..1, r0..r1, .., ..])
                .with_context(|| format!("read rows {r0}..{r1}"))?
        } else {
            ds.read_slice::<f32, _, ndarray::Ix3>(s![r0..r1, .., ..])
                .with_context(|| format!("read rows {r0}..{r1}"))?
                .insert_axis(ndarray::Axis(0))
        };
        for (i, frame) in frames.iter_mut().enumerate() {
            let src = block.slice(s![0, .., .., i]);
            frame.slice_mut(s![r0..r1, ..]).assign(&src);
        }
        progress(r1 as f32 / h as f32);
        r0 = r1;
    }
    Ok(frames)
}

/// Pixels per block when a masked stack is streamed as a (pixels × bins)
/// table: about 64 MiB of float32 per block.
fn pixels_per_block(k: usize) -> usize {
    ((64usize << 20) / (k.max(1) * 4)).clamp(1, 1 << 16)
}

/// Write only the pixels at `positions` (row, col) as the hsnt layout's
/// 2-D form: `data` of shape (pixels, bins), one row per selected pixel in
/// the order of `positions`. The CLI treats a 2-D data set as (pixels, bins)
/// with no image axes (so its rank estimate cannot pool neighbours).
pub fn write_input_h5_pixels(
    path: &Path,
    frames: &[Array2<f32>],
    positions: &[(usize, usize)],
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    let k = frames.len();
    let n = positions.len();
    if k == 0 || n == 0 {
        bail!("nothing to write: {k} image(s), {n} selected pixel(s)");
    }
    let file = hdf5::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let pb = pixels_per_block(k).min(n);
    let ds = file
        .new_dataset::<f32>()
        .chunk((pb, k.min(16)))
        .shape((n, k))
        .create("data")
        .context("create the data set")?;
    let mut p0 = 0;
    while p0 < n {
        let p1 = (p0 + pb).min(n);
        let mut block = Array2::<f32>::zeros((p1 - p0, k));
        for (i, &(y, x)) in positions[p0..p1].iter().enumerate() {
            for (j, frame) in frames.iter().enumerate() {
                block[(i, j)] = frame[(y, x)];
            }
        }
        ds.write_slice(&block, s![p0..p1, ..])
            .with_context(|| format!("write pixels {p0}..{p1}"))?;
        progress(p1 as f32 / n as f32);
        p0 = p1;
    }
    Ok(())
}

/// Read a denoised file written for [`write_input_h5_pixels`] (`data` of
/// shape (1, pixels, 1, bins) or (pixels, bins)) into `frames` at
/// `positions`; the other pixels of `frames` are left as they are.
pub fn read_denoised_pixels_into(
    path: &Path,
    positions: &[(usize, usize)],
    frames: &mut [Array2<f32>],
    progress: &mut dyn FnMut(f32),
) -> Result<()> {
    let file = hdf5::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let ds = file.dataset("data").context("no 'data' set in the denoised file")?;
    let shape = ds.shape();
    let (n, k, four_d) = match shape.as_slice() {
        [1, n, 1, k] => (*n, *k, true),
        [n, k] => (*n, *k, false),
        other => bail!("unexpected denoised data shape {other:?} for a masked run"),
    };
    if n != positions.len() || k != frames.len() {
        bail!(
            "the denoised file holds {n} pixels × {k} bins, the mask selected {} pixels of {} images",
            positions.len(),
            frames.len()
        );
    }
    let pb = pixels_per_block(k).min(n);
    let mut p0 = 0;
    while p0 < n {
        let p1 = (p0 + pb).min(n);
        let block: Array2<f32> = if four_d {
            ds.read_slice::<f32, _, Ix4>(s![0..1, p0..p1, 0..1, ..])
                .with_context(|| format!("read pixels {p0}..{p1}"))?
                .into_shape_with_order((p1 - p0, k))
                .context("reshape the denoised block")?
        } else {
            ds.read_slice::<f32, _, ndarray::Ix2>(s![p0..p1, ..])
                .with_context(|| format!("read pixels {p0}..{p1}"))?
        };
        for (i, &(y, x)) in positions[p0..p1].iter().enumerate() {
            for (j, frame) in frames.iter_mut().enumerate() {
                frame[(y, x)] = block[(i, j)];
            }
        }
        progress(p1 as f32 / n as f32);
        p0 = p1;
    }
    Ok(())
}

/// The component maps of a dehydrated file (`subspace_data`, shape
/// (…, rank)), one flat vector per component over the leading axes in
/// row-major order: the frame's pixels for a full run, the selected pixels
/// (in `positions` order) for a masked one.
pub fn read_dehydrated_maps(path: &Path) -> Result<Vec<Vec<f32>>> {
    let file = hdf5::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let ds = file
        .dataset("subspace_data")
        .context("no 'subspace_data' in the dehydrated file")?;
    let shape = ds.shape();
    let Some(&rank) = shape.last() else {
        bail!("subspace_data is a scalar");
    };
    let n: usize = shape[..shape.len() - 1].iter().product();
    let flat = ds.read_raw::<f32>().context("read subspace_data")?;
    if flat.len() != n * rank {
        bail!("subspace_data holds {} values, expected {}×{rank}", flat.len(), n);
    }
    Ok((0..rank)
        .map(|r| (0..n).map(|p| flat[p * rank + r]).collect())
        .collect())
}

/// A scratch folder under [`scratch_root`], removed when dropped.
pub struct ScratchDir {
    dir: PathBuf,
}

impl ScratchDir {
    pub fn path(&self) -> &Path {
        &self.dir
    }

    pub fn create() -> Result<Self> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let root = scratch_root();
        std::fs::create_dir_all(&root)
            .with_context(|| format!("create the scratch folder {}", root.display()))?;
        let dir = root.join(format!(
            "hsnt_{}_{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("create the exchange folder {}", dir.display()))?;
        Ok(Self { dir })
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

// ----- running the CLI -------------------------------------------------------

/// Stem of the exchange files (`input.h5` → `input_denoised.h5`, …).
const STEM: &str = "input";

/// The by-products copied from the exchange folder into memory, as
/// (CLI name, name in the export folder).
const ARTIFACTS: [(&str, &str); 4] = [
    ("input_report.json", "hsnt_report.json"),
    ("input_dehydrated.h5", "hsnt_dehydrated.h5"),
    ("input_spectra.png", "hsnt_spectra.png"),
    ("input_maps.png", "hsnt_maps.png"),
];

/// Where a CLI log line puts the run, as (stage text, overall fraction).
/// The CLI's stderr lines look like `16:46:00 INFO    factorization: …`;
/// its stdout carries the rank search and streamed passes.
pub fn stage_of_line(line: &str, rank_auto: bool) -> Option<(String, f32)> {
    let msg = strip_log_prefix(line);
    let m = msg.trim_start();
    if m.starts_with("HDF5 ") || m.starts_with("input type:") {
        Some(("mbirtorch: loading the data…".to_owned(), 0.22))
    } else if m.starts_with("loaded in") || m.starts_with("check:") {
        let next = if rank_auto {
            "mbirtorch: data checked — estimating the rank (likelihood-ratio tests)…"
        } else {
            "mbirtorch: data checked — planning the solve…"
        };
        Some((next.to_owned(), 0.3))
    } else if m.starts_with("rank search") {
        Some(("mbirtorch: estimating the rank (likelihood-ratio tests)…".to_owned(), 0.35))
    } else if m.starts_with("rank ") && m.contains("estimated") {
        Some((format!("mbirtorch: {}", m.split("; pass --rank").next().unwrap_or(m)), 0.45))
    } else if let Some(rest) = m.strip_prefix("plan: ") {
        let mode = rest.split(" solve").next().unwrap_or("").trim();
        Some((format!("mbirtorch: solving ({mode} solve)…"), 0.5))
    } else if m.starts_with("pass ") {
        Some((format!("mbirtorch: streamed {}", m.split(':').next().unwrap_or(m)), 0.6))
    } else if m.starts_with("factorization:") || m.starts_with("maps for the given basis") {
        Some(("mbirtorch: factorization done — writing the dehydrated file…".to_owned(), 0.7))
    } else if m.starts_with("wrote ") && m.contains("_dehydrated.h5") {
        Some(("mbirtorch: fit diagnostics…".to_owned(), 0.74))
    } else if m.starts_with("wrote ") && m.contains("_denoised.h5") {
        Some(("mbirtorch: writing the report and the plots…".to_owned(), 0.84))
    } else if m.starts_with("wrote ") && m.contains(".png") {
        Some(("mbirtorch: done — reading the corrected stack…".to_owned(), 0.88))
    } else {
        None
    }
}

/// `16:46:00 INFO    message` → `message`; other lines unchanged.
pub fn strip_log_prefix(line: &str) -> &str {
    let b = line.as_bytes();
    let is_time = b.len() > 9
        && b[2] == b':'
        && b[5] == b':'
        && b[8] == b' '
        && b[..8].iter().all(|c| c.is_ascii_digit() || *c == b':');
    if !is_time {
        return line;
    }
    let rest = &line[9..];
    // Level word (INFO, WARNING, ERROR, DEBUG) padded to 7 characters.
    match rest.split_once(' ') {
        Some((level, msg)) if level.chars().all(|c| c.is_ascii_uppercase()) => msg.trim_start(),
        _ => rest,
    }
}

/// Level of a CLI stderr line ("INFO", "WARNING", …), when it has one.
pub fn log_level(line: &str) -> Option<&str> {
    let b = line.as_bytes();
    if b.len() > 9 && b[8] == b' ' && b[..8].iter().all(|c| c.is_ascii_digit() || *c == b':') {
        line[9..].split_whitespace().next().filter(|w| w.chars().all(|c| c.is_ascii_uppercase()))
    } else {
        None
    }
}

enum Line {
    Out(String),
    Err(String),
    Closed,
}

/// What to show for a failed run: the CLI's own one-line error (`error: …`,
/// its last line on stderr) or the exception line of a traceback; failing
/// both, the last few meaningful lines.
pub fn failure_summary(lines: &[String]) -> String {
    let meaningful = |l: &&String| {
        let t = l.trim();
        !t.is_empty() && !t.chars().all(|c| c == '^' || c == '~')
    };
    if let Some(err) = lines.iter().rev().find(|l| l.starts_with("error:")) {
        return err.clone();
    }
    if let Some(exc) = lines.iter().rev().filter(meaningful).find(|l| {
        !l.starts_with(char::is_whitespace)
            && l.split_once(':')
                .is_some_and(|(name, _)| name.ends_with("Error") || name.ends_with("Exception"))
    }) {
        return exc.clone();
    }
    let tail: Vec<&str> = lines.iter().rev().filter(meaningful).take(3).map(String::as_str).collect();
    tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
}

/// One run of the command line. `Ok` even when the CLI fails (`failure`
/// holds its message); `Err` only when it cannot be launched or was
/// cancelled.
struct CliRun {
    lines: Vec<String>,
    warnings: Vec<String>,
    summary_line: String,
    failure: Option<String>,
    cmdline: String,
}

fn run_cli(
    python: &Path,
    input: &Path,
    out_dir: &Path,
    params: &HsntParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
    log: &mut dyn FnMut(&str),
) -> Result<CliRun> {
    let mut cmd = std::process::Command::new(python);
    cmd.arg("-m")
        .arg("mbirtorch.hsnt")
        .arg("denoise")
        .arg(input)
        .arg("-o")
        .arg(out_dir)
        .args(params.cli_args())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        // Unbuffered prints, so the rank search / streamed passes arrive live.
        .env("PYTHONUNBUFFERED", "1");
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // Die with this process (a killed GUI must not leave a solve holding
        // the GPU's memory).
        // SAFETY: prctl is async-signal-safe; nothing else runs in the child
        // before exec.
        unsafe {
            cmd.pre_exec(|| {
                libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                Ok(())
            });
        }
    }
    let cmdline = format!(
        "{} -m mbirtorch.hsnt denoise {} -o {} {}",
        python.display(),
        input.display(),
        out_dir.display(),
        params.cli_args().join(" ")
    );
    log(&format!("[dehydration_hydration] $ {cmdline}"));
    progress("mbirtorch: starting Python…", 0.16);
    let mut child = cmd
        .spawn()
        .with_context(|| format!("launch {}", python.display()))?;

    let (tx, rx) = mpsc::channel::<Line>();
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let tx_out = tx.clone();
    let out_reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = tx_out.send(Line::Out(line));
        }
        let _ = tx_out.send(Line::Closed);
    });
    let err_reader = std::thread::spawn(move || {
        for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
            let _ = tx.send(Line::Err(line));
        }
        let _ = tx.send(Line::Closed);
    });

    let mut lines: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();
    let mut summary_line = String::new();
    let mut closed = 0;
    let mut killed = false;
    let rank_auto = params.rank == Rank::Auto;
    loop {
        if !killed && cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            killed = true;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(Line::Closed) => {
                closed += 1;
                if closed == 2 {
                    break;
                }
            }
            Ok(Line::Out(line)) | Ok(Line::Err(line)) => {
                if let Some((stage, frac)) = stage_of_line(&line, rank_auto) {
                    progress(&stage, frac);
                }
                if log_level(&line) == Some("WARNING") {
                    warnings.push(strip_log_prefix(&line).to_owned());
                }
                if line.starts_with("done:") {
                    summary_line = line.clone();
                }
                log(&line);
                lines.push(line);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    let status = child.wait().context("wait for the mbirtorch process")?;
    let _ = out_reader.join();
    let _ = err_reader.join();
    if killed || cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    let failure = (!status.success()).then(|| format!("({status}) {}", failure_summary(&lines)));
    Ok(CliRun {
        lines,
        warnings,
        summary_line,
        failure,
        cmdline,
    })
}

/// Run `mbirtorch-hsnt denoise` on `frames`, or on the pixels `mask`
/// selects (true = keep; the others are not sent and keep their input
/// values in the result). Progress is `(stage, fraction)`; every CLI line
/// goes to `log` as it arrives. Setting `cancel` kills the Python process.
pub fn run_denoise(
    frames: &[Array2<f32>],
    mask: Option<&Array2<bool>>,
    params: &HsntParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
    log: &mut dyn FnMut(&str),
) -> Result<RunOutcome> {
    params.validate()?;
    if frames.is_empty() {
        bail!("empty stack");
    }
    let positions: Option<Vec<(usize, usize)>> = match mask {
        Some(m) => {
            if m.dim() != frames[0].dim() {
                bail!("the mask is {:?}, the images {:?}", m.dim(), frames[0].dim());
            }
            let pos = crate::mask::positions(m);
            if pos.is_empty() {
                bail!("the mask selects no pixel");
            }
            Some(pos)
        }
        None => None,
    };
    let python = python();
    if !python.is_file() {
        bail!(
            "the mbirtorch Python interpreter {} does not exist — run `pixi install -e cuda` in {} \
             or point {PYTHON_ENV_VAR} at another interpreter with mbirtorch",
            python.display(),
            MBIRTORCH_CHECKOUT
        );
    }
    let exchange = ScratchDir::create()?;
    let input = exchange.dir.join(format!("{STEM}.h5"));
    let out_dir = exchange.dir.join("out");

    progress("Writing the exchange file", 0.0);
    let (h, w, k) = (frames[0].nrows(), frames[0].ncols(), frames.len());
    let n_pixels = positions.as_ref().map_or(h * w, Vec::len);
    match &positions {
        Some(pos) => {
            log(&format!(
                "[dehydration_hydration] writing {} ({} of {} px selected by the mask × {k} images, {:.2} GiB, as a pixels × bins table)",
                input.display(),
                pos.len(),
                h * w,
                (pos.len() * k * 4) as f64 / f64::from(1u32 << 30)
            ));
            write_input_h5_pixels(&input, frames, pos, &mut |f| progress("Writing the exchange file", f * 0.15))?;
        }
        None => {
            log(&format!(
                "[dehydration_hydration] writing {} ({h}×{w} px × {k} images, {:.2} GiB)",
                input.display(),
                (h * w * k * 4) as f64 / f64::from(1u32 << 30)
            ));
            write_input_h5(&input, frames, &mut |f| progress("Writing the exchange file", f * 0.15))?;
        }
    }
    if cancel.load(Ordering::Relaxed) {
        bail!("cancelled");
    }

    let mut used = *params;
    let mut run = run_cli(&python, &input, &out_dir, &used, cancel, progress, log)?;
    if let Some(failure) = &run.failure
        && params.mode == SolveMode::Auto
        && failure.to_ascii_lowercase().contains("out of memory")
    {
        // The CLI's memory plan can be wrong when the GPU is shared; the
        // streamed solve needs a fraction of the memory.
        let note = "the device ran out of memory in the full solve — retrying in stream mode";
        log(&format!("[dehydration_hydration] {note}"));
        progress("mbirtorch: out of memory — retrying in stream mode…", 0.3);
        let _ = std::fs::remove_dir_all(&out_dir);
        used.mode = SolveMode::Stream;
        let mut retry = run_cli(&python, &input, &out_dir, &used, cancel, progress, log)?;
        let mut lines = run.lines;
        lines.push(format!("[dehydration_hydration] {note}"));
        lines.extend(retry.lines);
        retry.lines = lines;
        retry.warnings.insert(0, note.to_owned());
        run = retry;
    }
    let CliRun {
        lines,
        warnings,
        summary_line,
        failure,
        cmdline,
    } = run;
    if let Some(failure) = failure {
        bail!("mbirtorch-hsnt failed: {failure}");
    }

    progress("Reading the corrected stack", 0.9);
    let denoised = out_dir.join(format!("{STEM}_denoised.h5"));
    let frames_out = match &positions {
        Some(pos) => {
            let mut out = frames.to_vec();
            read_denoised_pixels_into(&denoised, pos, &mut out, &mut |f| {
                progress("Reading the corrected stack", 0.9 + 0.1 * f)
            })?;
            out
        }
        None => {
            let out = read_denoised_h5(&denoised, &mut |f| {
                progress("Reading the corrected stack", 0.9 + 0.1 * f)
            })?;
            if out.len() != k || out[0].dim() != (h, w) {
                bail!(
                    "the denoised stack has {} images of {:?}, the input {k} of {:?}",
                    out.len(),
                    out[0].dim(),
                    (h, w)
                );
            }
            out
        }
    };
    // The component maps, scattered back into images.
    let maps = match read_dehydrated_maps(&out_dir.join(format!("{STEM}_dehydrated.h5"))) {
        Ok(flat) => flat
            .into_iter()
            .filter(|m| m.len() == n_pixels)
            .map(|m| {
                let mut img = Array2::<f32>::zeros((h, w));
                match &positions {
                    Some(pos) => {
                        for (&(y, x), v) in pos.iter().zip(m) {
                            img[(y, x)] = v;
                        }
                    }
                    None => {
                        for (dst, v) in img.iter_mut().zip(m) {
                            *dst = v;
                        }
                    }
                }
                img
            })
            .collect(),
        Err(e) => {
            log(&format!("[dehydration_hydration] cannot read the component maps: {e:#}"));
            Vec::new()
        }
    };

    let mut artifacts = Vec::new();
    let mut report = HsntReport::default();
    for (cli_name, export_name) in ARTIFACTS {
        let path = out_dir.join(cli_name);
        if let Ok(bytes) = std::fs::read(&path) {
            if cli_name.ends_with("_report.json")
                && let Ok(text) = std::str::from_utf8(&bytes)
            {
                match HsntReport::parse(text) {
                    Ok(r) => report = r,
                    Err(e) => log(&format!("[dehydration_hydration] cannot parse the report: {e:#}")),
                }
            }
            artifacts.push(Artifact {
                name: export_name.to_owned(),
                bytes,
            });
        }
    }
    report.warnings = warnings;
    report.summary_line = summary_line;
    artifacts.push(Artifact {
        name: "hsnt_log.txt".to_owned(),
        bytes: format!("$ {cmdline}\n{}\n", lines.join("\n")).into_bytes(),
    });
    drop(exchange);
    Ok(RunOutcome {
        frames: frames_out,
        report,
        artifacts,
        log: lines,
        maps,
        n_pixels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dh_hsnt_{}_{name}", std::process::id()))
    }

    #[test]
    fn cli_args_cover_every_option() {
        let p = HsntParams {
            rank: Rank::Fixed(3),
            dose: Some(42.5),
            device: Device::Cuda(1),
            ..HsntParams::default()
        };
        let args = p.cli_args();
        let s = args.join(" ");
        assert!(s.contains("--input-type transmission"), "{s}");
        assert!(s.contains("--rank 3"), "{s}");
        assert!(s.contains("--spectra mle"), "{s}");
        assert!(s.contains("--device cuda:1"), "{s}");
        assert!(s.contains("--dose 42.5"), "{s}");
        assert!(s.contains("--rel-tol 1e-8"), "{s}");
        assert!(s.ends_with("-v"), "{s}");
        assert!(HsntParams::default().cli_args().join(" ").contains("--rank auto"));
    }

    #[test]
    fn parsers_round_trip_labels() {
        for t in InputType::ALL {
            assert_eq!(InputType::parse(t.label()), Some(t));
        }
        for s in Spectra::ALL {
            assert_eq!(Spectra::parse(s.label()), Some(s));
        }
        for m in SolveMode::ALL {
            assert_eq!(SolveMode::parse(m.label()), Some(m));
        }
        for c in Compile::ALL {
            assert_eq!(Compile::parse(c.label()), Some(c));
        }
        assert_eq!(Rank::parse("auto"), Some(Rank::Auto));
        assert_eq!(Rank::parse("4"), Some(Rank::Fixed(4)));
        assert_eq!(Rank::parse("0"), None);
        assert_eq!(Device::parse("cuda"), Some(Device::Cuda(0)));
        assert_eq!(Device::parse("cuda:3"), Some(Device::Cuda(3)));
        assert_eq!(Device::parse("CPU"), Some(Device::Cpu));
        assert_eq!(Device::parse("tpu"), None);
    }

    #[test]
    fn support_needs_dose() {
        let p = HsntParams {
            spectra: Spectra::Support,
            ..HsntParams::default()
        };
        assert!(p.validate().is_err());
        assert!(HsntParams { dose: Some(10.0), ..p }.validate().is_ok());
    }

    #[test]
    fn exchange_round_trip() {
        let frames: Vec<Array2<f32>> = (0..5)
            .map(|k| Array2::from_shape_fn((7, 4), |(y, x)| (k * 100 + y * 10 + x) as f32))
            .collect();
        let dir = tmp("xchg");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("input.h5");
        let mut ticks = Vec::new();
        write_input_h5(&path, &frames, &mut |f| ticks.push(f)).unwrap();
        assert_eq!(ticks.last().copied(), Some(1.0));
        // The file the CLI would write back has a leading views axis; the
        // reader accepts both layouts.
        let back = read_denoised_h5(&path, &mut |_| {}).unwrap();
        assert_eq!(back.len(), 5);
        assert_eq!(back[3], frames[3]);
        let file = hdf5::File::open(&path).unwrap();
        let ds = file.dataset("data").unwrap();
        assert_eq!(ds.shape(), vec![7, 4, 5]);
        assert_eq!(ds.chunk(), Some(vec![7, 4, 5]));
        assert_eq!(rows_per_block(512, 2786, 64), 64);
        assert_eq!(rows_per_block(512, 2786, 1), 11);
        assert_eq!(rows_per_block(512, 80, 64), 256);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn masked_exchange_round_trip() {
        let frames: Vec<Array2<f32>> = (0..3)
            .map(|k| Array2::from_shape_fn((4, 5), |(y, x)| (k * 100 + y * 10 + x) as f32))
            .collect();
        let mask = Array2::from_shape_fn((4, 5), |(y, x)| (y + x) % 3 == 0);
        let pos = crate::mask::positions(&mask);
        let dir = tmp("xchg_masked");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("input.h5");
        write_input_h5_pixels(&path, &frames, &pos, &mut |_| {}).unwrap();
        let file = hdf5::File::open(&path).unwrap();
        assert_eq!(file.dataset("data").unwrap().shape(), vec![pos.len(), 3]);
        drop(file);
        // Read back into a zeroed stack: only the selected pixels are filled.
        let mut out: Vec<Array2<f32>> = (0..3).map(|_| Array2::zeros((4, 5))).collect();
        read_denoised_pixels_into(&path, &pos, &mut out, &mut |_| {}).unwrap();
        for ((y, x), &m) in mask.indexed_iter() {
            assert_eq!(out[2][(y, x)], if m { frames[2][(y, x)] } else { 0.0 });
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn log_lines_map_to_stages() {
        assert_eq!(
            strip_log_prefix("16:46:00 INFO    factorization: full, 77 steps in 0.4 s"),
            "factorization: full, 77 steps in 0.4 s"
        );
        assert_eq!(log_level("16:46:00 WARNING reduced chi-square"), Some("WARNING"));
        assert_eq!(log_level("done: rank 2"), None);
        let (stage, f) = stage_of_line("16:46:00 INFO    plan: full solve. NVIDIA A100", true).unwrap();
        assert_eq!(stage, "mbirtorch: solving (full solve)…");
        assert_eq!(f, 0.5);
        let (stage, _) =
            stage_of_line("16:46:00 INFO    rank 2 estimated by likelihood-ratio tests (a; b); pass --rank", true)
                .unwrap();
        assert_eq!(stage, "mbirtorch: rank 2 estimated by likelihood-ratio tests (a; b)");
        assert!(stage_of_line("  pass 3: full-data loss 1e5", false).unwrap().0.contains("pass 3"));
        assert!(stage_of_line("done: rank 2 (estimated)", false).is_none());
    }

    #[test]
    fn failure_summary_prefers_the_error_line() {
        let lines: Vec<String> = ["Traceback (most recent call last):", "  File \"x.py\", line 1", "    foo()", "    ^^^^^",
            "ValueError: the data hold no counts"].iter().map(|s| s.to_string()).collect();
        assert_eq!(failure_summary(&lines), "ValueError: the data hold no counts");
        let lines: Vec<String> = ["16:00:00 INFO    loaded", "error: --rank 9 exceeds min(pixels, bins) = 5"]
            .iter().map(|s| s.to_string()).collect();
        assert!(failure_summary(&lines).starts_with("error: --rank 9"));
        let lines: Vec<String> = ["a", "", "b"].iter().map(|s| s.to_string()).collect();
        assert_eq!(failure_summary(&lines), "a | b");
    }

    #[test]
    fn report_parses_the_cli_json() {
        let json = r#"{"input":"x","input_type":"transmission","dose":50.0,
            "checks":[{"level":"ok","message":"7,680 pixels"}],
            "result":{"mode":"full","memory_plan":"A100","spectra":"mle","steps":77,
                      "solve_seconds":0.37,"loss_final":46716.9,"gpu_peak_gib":0.14,"rank":2,
                      "rank_note":"rank 2 estimated","fit":{"relative_residual":0.2049,"reduced_chi2":2.106}}}"#;
        let r = HsntReport::parse(json).unwrap();
        assert_eq!(r.rank, Some(2));
        assert_eq!(r.steps, Some(77));
        assert_eq!(r.reduced_chi2, Some(2.106));
        assert_eq!(r.checks.len(), 1);
        assert_eq!(r.mode, "full");
        assert!(r.chi2_verdict().unwrap().starts_with("above"));
        assert_eq!(r.to_json()["rank"], 2);
    }

    #[test]
    fn commit_is_read_from_a_git_folder() {
        let dir = tmp("gitdir");
        std::fs::create_dir_all(dir.join(".git/refs/heads")).unwrap();
        std::fs::write(dir.join(".git/HEAD"), "ref: refs/heads/hsnt\n").unwrap();
        std::fs::write(dir.join(".git/refs/heads/hsnt"), "edb0bcba9988\n").unwrap();
        assert_eq!(mbirtorch_commit_in(&dir).as_deref(), Some("edb0bcb (hsnt)"));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Real run through the Python environment (needs the hsnt checkout and
    /// a GPU or a patient CPU): `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn real_denoise_run() {
        let k = 120;
        let frames: Vec<Array2<f32>> = (0..k)
            .map(|i| {
                Array2::from_shape_fn((32, 24), |(y, x)| {
                    let a = if (y as i32 - 16).pow(2) + (x as i32 - 12).pow(2) < 64 { 0.6 } else { 0.0 };
                    let a = a * if i > 60 { 1.5 } else { 1.0 };
                    (-a as f32).exp() * (1.0 + 0.02 * ((i * 7 + y * 3 + x) % 5) as f32 - 0.04)
                })
            })
            .collect();
        let params = HsntParams {
            rank: Rank::Fixed(1),
            ..HsntParams::default()
        };
        let cancel = AtomicBool::new(false);
        let mut stages = Vec::new();
        let out = run_denoise(&frames, None, &params, &cancel, &mut |s, f| stages.push((s.to_owned(), f)), &mut |_| {})
            .unwrap();
        assert_eq!(out.frames.len(), k);
        assert_eq!(out.report.rank, Some(1));
        assert_eq!(out.maps.len(), 1);
        assert_eq!(out.maps[0].dim(), (32, 24));
        // Masked: only the disc is solved, the rest keeps its input values.
        let mask = Array2::from_shape_fn((32, 24), |(y, x)| (y as i32 - 16).pow(2) + (x as i32 - 12).pow(2) < 100);
        let masked = run_denoise(&frames, Some(&mask), &params, &cancel, &mut |_, _| {}, &mut |_| {}).unwrap();
        assert_eq!(masked.n_pixels, crate::mask::count(&mask));
        assert_eq!(masked.frames[5][(0, 0)], frames[5][(0, 0)]);
        assert_ne!(masked.frames[5][(16, 12)], frames[5][(16, 12)]);
        assert_eq!(masked.maps[0][(0, 0)], 0.0);
        assert!(out.artifacts.iter().any(|a| a.name == "hsnt_report.json"));
        assert!(stages.iter().any(|(s, _)| s.contains("solving")), "{stages:?}");
    }
}
