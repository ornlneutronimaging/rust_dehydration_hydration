//! Dehydration/Hydration Correction — beta on mbirtorch's `hsnt` package:
//! load a stack of TIFF images, denoise it with the maximum-likelihood
//! dehydrate/rehydrate factorization of `mbirtorch.hsnt` (Harel Dor's `hsnt`
//! branch, run as a Python subprocess in its own pixi environment), inspect
//! the result, and export the corrected stack. `--run` does the same without
//! a GUI, for scripting and pipelines. The data can be given as
//! files/folders or located from its run number (`--run-number`, NeXus
//! lookup as in rust_tiff_viewer).

use dehydration_hydration::app::DehydrationApp;
use dehydration_hydration::correction::{run_correction, CorrectionParams};
use dehydration_hydration::export::{export_corrected, Provenance};
use dehydration_hydration::hsnt_cli::{self, Compile, Device, InputType, Rank, SolveMode, Spectra};
use dehydration_hydration::export::MaskInfo;
use dehydration_hydration::loader;
use dehydration_hydration::mask::{MaskRect, MaskSpec, RectMode};
use dehydration_hydration::run_lookup;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

const USAGE: &str = "\
dehydration_hydration (beta, mbirtorch hsnt) — dehydration/hydration denoising of a TIFF stack

USAGE:
  dehydration_hydration [OPTIONS] [INPUT ...]

ARGS:
  INPUT   TIFF file(s) or a folder of TIFF images (subfolders are searched
          when the folder itself has none). When omitted, the data can be
          opened from within the application (also from a run number, see
          --run-number).

OPTIONS:
      --ipts <N>           Pre-select the experiment (N or IPTS-N): the toolbar
                           IPTS drop-down, which makes the Open Folder / Open
                           Files dialogs start in /SNS/VENUS/IPTS-N/shared
                           (without it they start in /SNS/VENUS). A run number
                           lookup selects the run's IPTS by itself
  -r, --run-number <N>     Locate the data from its run number instead of
                           INPUT: finds /SNS/VENUS/IPTS-*/nexus/VENUS_<N>.nxs.h5
                           and reads the detector, the image folder and the
                           detector offset from it. In the window a Timepix
                           run asks for raw or autoreduce data (the raw
                           .fits frames cannot be loaded here, so autoreduce
                           is the usual choice); headless mode loads the
                           autoreduce TIFFs of a Timepix run and the raw
                           image(s) of any other run
  --run                    Headless mode: run the correction and export the
                           corrected stack without opening a window
                           (requires INPUT or --run-number, and --output)
  -o, --output <DIR>       Folder receiving the corrected subfolder
                           '<input>_dehydration_hydration_corrected'

  Correction (the options of `mbirtorch-hsnt denoise`):
  --input-type <TYPE>      What the values are: transmission (default — the
                           normalized stacks this tool loads), attenuation
                           (= −log(transmission)), or auto (inferred from the
                           values; fails when they cannot be told apart).
                           --dataset-type is accepted as a synonym
  --rank <N|auto>          Number of components, about the number of distinct
                           materials (default auto: estimated from the data by
                           likelihood-ratio tests). --materials N is a synonym
  --max-rank <N>           Largest rank the estimate considers (default 6)
  --spectra <HOW>          mle (default) | unconstrained | support — how the
                           component spectra are estimated; support needs
                           --dose
  --dose <D>               Open-beam counts per pixel and bin, when known:
                           gives the fit a chi-square against the Poisson
                           noise (default: unknown)
  --device <DEV>           auto (default: CUDA when available) | cpu | cuda |
                           cuda:N
  --mode <MODE>            auto (default) | full | stream — whole solve on the
                           device or streamed by chunks of pixels
  --max-steps <N>          Solver step cap of a full solve (default 1000;
                           --max-iter is a synonym)
  --rel-tol <F>            Relative loss change per step that stops the solve
                           (default 1e-8)
  --max-passes <N>         Stream mode: polish passes over the data (default 5)
  --compile <auto|on|off>  torch.compile of the solver kernels (default auto)
  --beta-loss, --safety-factor
                           Options of the production (NMF) tool: accepted and
                           ignored, with a warning, so existing scripts run

  Mask (only the selected pixels are sent to mbirtorch; the others keep
  their raw values in the corrected stack; all pixels when no mask option):
  --mask <FILE>            Mask image (TIFF or .npy, nonzero = selected), read
                           with the stack's detector orientation
  --mask-include <x0,y0,x1,y1>
                           Keep only the pixels inside this rectangle
                           (half-open pixel bounds; repeatable: inside any)
  --mask-exclude <x0,y0,x1,y1>
                           Drop the pixels inside this rectangle (repeatable)
  --mask-range <LO:HI>     Keep the pixels whose integrated (summed) value
                           lies in [LO, HI]

  --bin <B>                Spatial binning factor (default 1 = full
                           resolution; >1 exports a binned preview)
  -t, --offset <MICROSEC>  Detector offset: constant added to the TOF values
                           of the spectra file (µs, default 0); shifts the
                           TOF and wavelength axes of the profile plots
  --detector <NAME>        Force the detector the stack is loaded as, which
                           decides its orientation: timepix (frames
                           transposed), ccd (flipped vertically and horizontally), qhy (rotated 90°
                           counterclockwise) or as-is. By default it is
                           recognized from the folder layout (images/tpx1,
                           images/ikonxl, …); the toolbar has a combobox.
                           Exported images are always written back in the
                           on-disk orientation of the input
  -h, --help               Show this help

ENVIRONMENT:
  DEHY_HSNT_PYTHON         Python interpreter with mbirtorch (default: the cuda
                           pixi environment of git/mbirtorch_hsnt)
  DEHY_HSNT_SCRATCH        Folder for the exchange HDF5 files (default: the
                           system temporary folder)

The correction is `mbirtorch.hsnt denoise`: the maximum-likelihood (Poisson)
factorization X = W·H of the attenuation (dehydration), multiplied back into
denoised data (rehydration), after M. S. N. Chowdhury et al., \"Fast
Hyperspectral Neutron Tomography\", IEEE Trans. Comput. Imaging 11, 663-677
(2025), as re-implemented by Harel Dor (mbirtorch branch hsnt).
";

struct Cli {
    inputs: Vec<PathBuf>,
    run: bool,
    output: Option<PathBuf>,
    params: CorrectionParams,
    bin: usize,
    offset_us: f64,
    detector: Option<loader::Detector>,
    /// `--run-number`: locate the data from the run's NeXus file.
    run_number: Option<u32>,
    /// `--ipts`: pre-select the experiment the open dialogs start in.
    ipts: Option<u32>,
    /// `--mask*`: the pixel mask definition (file loaded after the stack).
    mask_file: Option<PathBuf>,
    mask_rects: Vec<MaskRect>,
    mask_range: Option<(f32, f32)>,
}

fn parse_args() -> Result<Cli, String> {
    let mut cli = Cli {
        inputs: Vec::new(),
        run: false,
        output: None,
        params: CorrectionParams::default(),
        bin: 1,
        offset_us: 0.0,
        detector: None,
        run_number: None,
        ipts: None,
        mask_file: None,
        mask_rects: Vec::new(),
        mask_range: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut value = |name: &str| {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--run" => cli.run = true,
            "-r" | "--run-number" | "--run_number" => {
                let v = value("--run-number")?;
                cli.run_number = match v.trim().parse::<u32>() {
                    Ok(n) if n > 0 => Some(n),
                    _ => return Err(format!("invalid --run-number '{v}': expected a positive integer")),
                };
            }
            "--ipts" => {
                let v = value("--ipts")?;
                let digits = v.trim().to_ascii_uppercase();
                let digits = digits.strip_prefix("IPTS-").unwrap_or(&digits);
                cli.ipts = match digits.parse::<u32>() {
                    Ok(n) if n > 0 => Some(n),
                    _ => return Err(format!("invalid --ipts '{v}': expected a number or IPTS-<number>")),
                };
            }
            "-o" | "--output" => cli.output = Some(PathBuf::from(value("--output")?)),
            "--input-type" | "--input_type" | "--dataset-type" | "--dataset_type" => {
                let v = value("--input-type")?;
                cli.params.input_type = InputType::parse(&v)
                    .ok_or_else(|| format!("unknown input type '{v}': expected transmission, attenuation or auto"))?;
            }
            "--rank" | "--materials" => {
                let v = value("--rank")?;
                cli.params.rank = Rank::parse(&v)
                    .ok_or_else(|| format!("invalid --rank '{v}': expected a positive integer or auto"))?;
            }
            "--max-rank" | "--max_rank" => {
                cli.params.max_rank = value("--max-rank")?
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or_else(|| "--max-rank must be a positive integer".to_owned())?;
            }
            "--spectra" => {
                let v = value("--spectra")?;
                cli.params.spectra = Spectra::parse(&v)
                    .ok_or_else(|| format!("unknown --spectra '{v}': expected mle, unconstrained or support"))?;
            }
            "--dose" => {
                let v = value("--dose")?;
                let d: f64 = v.parse().map_err(|e| format!("invalid --dose '{v}': {e}"))?;
                if !(d.is_finite() && d > 0.0) {
                    return Err(format!("--dose must be a positive number (got {v})"));
                }
                cli.params.dose = Some(d);
            }
            "--device" => {
                let v = value("--device")?;
                cli.params.device = Device::parse(&v)
                    .ok_or_else(|| format!("invalid --device '{v}': expected auto, cpu, cuda or cuda:N"))?;
            }
            "--mode" => {
                let v = value("--mode")?;
                cli.params.mode = SolveMode::parse(&v)
                    .ok_or_else(|| format!("invalid --mode '{v}': expected auto, full or stream"))?;
            }
            "--max-steps" | "--max_steps" | "--max-iter" | "--max_iter" => {
                cli.params.max_steps = value("--max-steps")?
                    .parse::<usize>()
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or_else(|| "--max-steps must be a positive integer".to_owned())?;
            }
            "--rel-tol" | "--rel_tol" => {
                let v = value("--rel-tol")?;
                let t: f64 = v.parse().map_err(|e| format!("invalid --rel-tol '{v}': {e}"))?;
                if !(t.is_finite() && t >= 0.0) {
                    return Err(format!("--rel-tol must be a non-negative number (got {v})"));
                }
                cli.params.rel_tol = t;
            }
            "--max-passes" | "--max_passes" => {
                cli.params.max_passes = value("--max-passes")?
                    .parse::<usize>()
                    .map_err(|_| "--max-passes must be a non-negative integer".to_owned())?;
            }
            "--compile" => {
                let v = value("--compile")?;
                cli.params.compile = Compile::parse(&v)
                    .ok_or_else(|| format!("invalid --compile '{v}': expected auto, on or off"))?;
            }
            "--mask" => cli.mask_file = Some(PathBuf::from(value("--mask")?)),
            "--mask-include" | "--mask_include" | "--mask-exclude" | "--mask_exclude" => {
                let v = value(&a)?;
                let mode = if a.contains("include") { RectMode::Include } else { RectMode::Exclude };
                cli.mask_rects.push(MaskRect::parse(&v, mode).ok_or_else(|| {
                    format!("invalid {a} '{v}': expected x0,y0,x1,y1 with x1 > x0 and y1 > y0")
                })?);
            }
            "--mask-range" | "--mask_range" => {
                let v = value("--mask-range")?;
                let (lo, hi) = v
                    .split_once(':')
                    .and_then(|(a, b)| Some((a.trim().parse::<f32>().ok()?, b.trim().parse::<f32>().ok()?)))
                    .filter(|(lo, hi)| lo.is_finite() && hi.is_finite() && lo <= hi)
                    .ok_or_else(|| format!("invalid --mask-range '{v}': expected LO:HI with LO <= HI"))?;
                cli.mask_range = Some((lo, hi));
            }
            "--beta-loss" | "--beta_loss" | "--safety-factor" | "--safety_factor" => {
                let v = value(&a)?;
                eprintln!("Warning: {a} {v} is an option of the production NMF tool; the beta ignores it.");
            }
            "--bin" => {
                cli.bin = value("--bin")?
                    .parse()
                    .map_err(|_| "--bin must be a positive integer".to_owned())?;
                if cli.bin == 0 {
                    return Err("--bin must be at least 1".to_owned());
                }
            }
            "-t" | "--offset" => {
                let v = value("--offset")?;
                cli.offset_us = v
                    .parse()
                    .map_err(|e| format!("invalid --offset '{v}': {e}"))?;
                if !cli.offset_us.is_finite() {
                    return Err(format!("--offset must be finite (got {})", cli.offset_us));
                }
            }
            "--detector" => {
                let v = value("--detector")?;
                cli.detector = Some(loader::Detector::parse(&v).ok_or_else(|| {
                    format!("invalid --detector '{v}': expected timepix, ccd, qhy or as-is")
                })?);
            }
            s if s.starts_with('-') => return Err(format!("unknown option: {s}")),
            _ => cli.inputs.push(PathBuf::from(a)),
        }
    }
    cli.params.validate().map_err(|e| e.to_string())?;
    if cli.run {
        if cli.inputs.is_empty() && cli.run_number.is_none() {
            return Err("--run requires an INPUT folder or files, or --run-number".to_owned());
        }
        if cli.output.is_none() {
            return Err("--run requires --output <DIR>".to_owned());
        }
    }
    Ok(cli)
}

/// Expand folders to the image files they contain, so errors (missing
/// folder, no images) surface on stderr early.
fn expand_inputs(inputs: &[PathBuf]) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match loader::list_supported_in_dir(input) {
                Ok(found) => files.extend(found),
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    std::process::exit(1);
                }
            }
        } else {
            files.push(input.clone());
        }
    }
    files
}

/// Headless: load, correct, export, print the export folder on stdout.
/// The files come from INPUT, plus those of the run given with
/// `--run-number` (autoreduce TIFFs of a Timepix run, raw images otherwise).
fn run_headless(cli: &Cli, mut files: Vec<PathBuf>) -> anyhow::Result<()> {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut daslog: Option<String> = None;
    if let Some(run) = cli.run_number {
        eprintln!("Locating run {run}…");
        let info = run_lookup::resolve(run)?;
        let (run_files, which) = info.default_files()?;
        eprintln!(
            "Run {run}: {}, {} — {which} data, {} file(s) in {}",
            info.ipts,
            info.detector_text(),
            run_files.len(),
            run_files
                .first()
                .and_then(|p| p.parent())
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        );
        daslog = info.detector.clone();
        files.extend(run_files);
    }

    eprintln!("Loading {} file(s)…", files.len());
    let last_decile = AtomicUsize::new(0);
    let detector = loader::detect_detector(&files, daslog.as_deref(), cli.detector);
    let stack = loader::load_paths_with_progress(&files, detector, |done, total| {
        let decile = done * 10 / total.max(1);
        if decile > last_decile.swap(decile, Ordering::Relaxed) {
            eprintln!("  loaded {done}/{total}");
        }
    })?;
    eprintln!(
        "Loaded {} image(s), {}×{} px ({} NaN/Inf pixel(s) zeroed); detector {}: {}.",
        stack.n_frames(),
        stack.width,
        stack.height,
        stack.nonfinite_fixed,
        stack.detector.summary(),
        stack.orientation
    );

    // The mask, from the command-line definition and the integrated image.
    let mut spec = MaskSpec {
        rects: cli.mask_rects.clone(),
        range: cli.mask_range,
        file: None,
    };
    if let Some(path) = &cli.mask_file {
        let pixels = dehydration_hydration::mask::load_file(path, stack.detector)?;
        spec.file = Some(dehydration_hydration::mask::MaskFile {
            path: path.clone(),
            pixels: std::sync::Arc::new(pixels),
        });
    }
    let mask = if spec.is_empty() {
        None
    } else {
        let mut integrated = ndarray::Array2::<f32>::zeros((stack.height, stack.width));
        for f in &stack.frames {
            integrated += f;
        }
        let m = spec.build(&integrated)?.expect("non-empty spec");
        let n = dehydration_hydration::mask::count(&m);
        eprintln!(
            "Mask: {} — {n} of {} pixel(s) selected ({:.1}%).",
            spec.describe(),
            m.len(),
            100.0 * n as f64 / m.len().max(1) as f64
        );
        Some(m)
    };

    let cancel = AtomicBool::new(false);
    let mut last_stage = String::new();
    let mut progress = |stage: &str, fraction: f32| {
        if stage != last_stage {
            eprintln!("[{:>3.0}%] {stage}", fraction * 100.0);
            last_stage = stage.to_owned();
        }
    };
    let mut log = |line: &str| eprintln!("    {line}");
    eprintln!(
        "mbirtorch hsnt ({}): {}",
        hsnt_cli::mbirtorch_commit().unwrap_or_else(|| "commit unknown".to_owned()),
        cli.params.summary()
    );
    let out = run_correction(&stack, cli.params, cli.bin, mask.as_ref(), &cancel, &mut progress, &mut log)?;
    eprintln!(
        "Correction done in {:.1} s — {}",
        out.elapsed_seconds,
        if out.report.summary_line.is_empty() {
            "no summary from mbirtorch".to_owned()
        } else {
            out.report.summary_line.clone()
        }
    );
    for w in &out.report.warnings {
        eprintln!("  warning: {w}");
    }

    let input_dir = files
        .first()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_default();
    let input_dir_name = input_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "images".to_owned());
    let (h, w) = out.integrated_mean.dim();
    let provenance = Provenance {
        input_folder: input_dir,
        num_images: out.frames.len(),
        image_width: w,
        image_height: h,
        params: cli.params,
        bin: out.bin,
        elapsed_seconds: out.elapsed_seconds,
        report: out.report.clone(),
        mbirtorch_commit: hsnt_cli::mbirtorch_commit(),
        mask: out.mask.as_ref().map(|m| MaskInfo {
            selected: out.n_pixels,
            total: m.len(),
            description: spec.describe(),
        }),
    };
    let folder = export_corrected(
        cli.output.as_deref().expect("checked in parse_args"),
        &input_dir_name,
        &out.frames,
        &stack.sources,
        stack.orientation,
        &provenance,
        &out.artifacts,
        &mut |_, _| {},
    )?;
    eprintln!("Exported {} corrected image(s).", out.frames.len());
    println!("{}", folder.display());
    Ok(())
}

fn main() -> eframe::Result<()> {
    // The classic GTK file chooser (with a typeable path and a starting
    // folder), never the XDG portal dialog, which ignores set_directory.
    if std::env::var_os("GTK_USE_PORTAL").is_none() {
        // SAFETY: called before any other thread exists (start of main).
        unsafe { std::env::set_var("GTK_USE_PORTAL", "0") };
    }
    let cli = match parse_args() {
        Ok(cli) => cli,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let files = expand_inputs(&cli.inputs);
    let offset_us = cli.offset_us;
    let detector = cli.detector;
    let run_number = cli.run_number;
    let ipts = cli.ipts;
    let params = cli.params;

    if cli.run {
        if let Err(e) = run_headless(&cli, files) {
            eprintln!("Error: {e:#}");
            std::process::exit(1);
        }
        return Ok(());
    }

    const TITLE: &str = "VENUS Dehydration / Hydration Correction (beta — mbirtorch hsnt)";
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title(TITLE),
        ..Default::default()
    };

    eframe::run_native(
        TITLE,
        native_options,
        Box::new(move |cc| {
            // Saved light/dark preference, shared by all the VENUS rust
            // tools (dark when none is saved); the toolbar has a toggle.
            cc.egui_ctx.set_theme(dehydration_hydration::theme::load());
            dehydration_hydration::theme::install_fonts(&cc.egui_ctx);
            cc.egui_ctx
                .set_zoom_factor(dehydration_hydration::zoom::load());
            let mut app = DehydrationApp::new();
            app.set_detector_offset(offset_us);
            app.set_detector_override(detector);
            app.set_ipts(ipts);
            app.set_params(params);
            if !files.is_empty() {
                app.start_load(files, &cc.egui_ctx);
            }
            if let Some(run) = run_number {
                app.set_startup_run(run);
            }
            Ok(Box::new(app))
        }),
    )
}
