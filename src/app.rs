//! The egui/eframe application: load a TIFF stack, tune the correction
//! parameters, run the mbirtorch hsnt dehydration/hydration denoising on a
//! background thread (a Python subprocess, see [`crate::hsnt_cli`]), compare
//! corrected vs raw images, inspect region profiles, and export the
//! corrected stack — the same workflow as the dehydration_hydration
//! notebook, in one native window.

use crate::colormap::Colormap;
use crate::correction::{start_correction, CorrectionMsg, CorrectionParams};
use crate::export::{start_export, ExportMsg, MaskInfo, Provenance};
use crate::hsnt_cli::{self, Artifact, Compile, Device, HsntReport, InputType, Rank, SolveMode, Spectra};
use crate::loader::{self, Detector, ImageStack, Selection};
use crate::mask::{self, MaskFile, MaskRect, MaskSpec, RectMode};
use crate::run_lookup::{self, RunInfo};
use crate::spectra;

use egui::{Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions};
use egui_plot::{Corner, CoordinatesFormatter, Legend, MarkerShape, Plot, PlotPoints, Points};
use ndarray::{s, Array2};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, TryRecvError};
use std::sync::Arc;

/// Width of the colorbar column: gradient strip + ticks + value labels.
const COLORBAR_WIDTH: f32 = 78.0;

/// Spatial binning factor of the fast preview run.
const PREVIEW_BIN: usize = 2;

/// Colors of the profile plot series (fixed, so the legend — sorted by name
/// — always matches the markers): uncorrected data in orange, corrected in
/// blue; the single-pixel spectra in lighter tints of the same two.
const UNCORRECTED_COLOR: Color32 = Color32::from_rgb(255, 140, 0);
const CORRECTED_COLOR: Color32 = Color32::from_rgb(60, 140, 255);
const PIXEL_UNCORRECTED_COLOR: Color32 = Color32::from_rgb(255, 200, 120);
const PIXEL_CORRECTED_COLOR: Color32 = Color32::from_rgb(150, 200, 255);

/// Outline colors of the mask rectangles on the integrated image.
const MASK_INCLUDE_COLOR: Color32 = Color32::from_rgb(80, 220, 80);
const MASK_EXCLUDE_COLOR: Color32 = Color32::from_rgb(255, 70, 70);

/// ORNL Neutron Imaging team logo (same asset as the other rust
/// applications) and the MBIRTorch logo (the Purdue library that runs the
/// correction), embedded in the binary and shown at the bottom-left of the
/// window. The MBIRTorch logo has a light- and a dark-background variant;
/// the one matching the active theme is displayed. The light one is the
/// official `docs/source/_static/logo.png` of the mbirtorch repository; the
/// dark one is derived from it by inverting its achromatic (black "MBIR")
/// pixels to white, the red "Torch" being kept.
const IMAGING_LOGO_BYTES: &[u8] = include_bytes!("../logos/ImagingLogo.png");
const MBIRTORCH_LOGO_LIGHT_BYTES: &[u8] = include_bytes!("../logos/mbirtorch_logo.png");
const MBIRTORCH_LOGO_DARK_BYTES: &[u8] = include_bytes!("../logos/mbirtorch_logo_dark_background.png");
const LOGO_HEIGHT: f32 = 44.0;

fn load_logo(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(name, color_image, TextureOptions::LINEAR))
}

/// Tooltip of each spectra estimator (the CLI's wording, shortened).
fn spectra_help(s: Spectra) -> &'static str {
    match s {
        Spectra::Mle => "mle: the spectra that best fit the measured counts (default)",
        Spectra::Unconstrained => {
            "unconstrained: removes a bias the best fit has at low dose; worth it with many \
             pixels (on a test phantom 7-12 dB at a million pixels, up to 2 dB at 65,000)"
        }
        Spectra::Support => {
            "support: works out which components each pixel contains, which removes the same \
             bias and zeroes the background of the maps; needs the dose"
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Raw,
    Result,
    Profiles,
}

/// Messages sent from the background loading thread to the UI.
enum LoadMsg {
    Progress { done: usize, total: usize },
    /// The detector offset of the run the file names belong to (sent before
    /// `Done`): the NeXus file found and its offset, or why the lookup
    /// failed (unmounted /SNS/VENUS, no NeXus file for that run…).
    RunOffset { run: u32, result: Result<(PathBuf, Option<f64>), String> },
    Done(anyhow::Result<ImageStack>),
}

struct LoadJob {
    rx: Receiver<LoadMsg>,
    done: usize,
    total: usize,
    /// The `RunOffset` message, kept until the stack is in.
    run_offset: Option<(u32, Result<(PathBuf, Option<f64>), String>)>,
}

/// The "Run number…" dialog: locate a run's images from its run number
/// (NeXus lookup, see [`run_lookup`]) — same dialog as rust_tiff_viewer.
struct RunLookup {
    open: bool,
    /// Text typed in the run number field.
    input: String,
    /// Focus the field on the next frame (dialog just opened).
    focus: bool,
    state: RunState,
}

impl RunLookup {
    fn new() -> Self {
        Self {
            open: false,
            input: String::new(),
            focus: false,
            state: RunState::Idle,
        }
    }
}

enum RunState {
    Idle,
    /// Lookup running on a background thread (IPTS scan + NeXus read over
    /// the network share).
    Busy {
        run: u32,
        rx: Receiver<anyhow::Result<RunInfo>>,
    },
    /// Timepix run located: waiting for the raw / autoreduce choice.
    Found(RunInfo),
    Failed { run: u32, error: String },
}

/// Which data of a located Timepix run to load.
#[derive(Clone, Copy, PartialEq)]
enum RunChoice {
    Raw,
    Autoreduce,
}

struct CorrJob {
    rx: Receiver<CorrectionMsg>,
    stage: String,
    fraction: f32,
    cancel: Arc<AtomicBool>,
}

struct ExportJob {
    rx: Receiver<ExportMsg>,
    done: usize,
    total: usize,
}

/// A finished correction, kept alongside the parameters that produced it.
struct ResultState {
    frames: Arc<Vec<Array2<f32>>>,
    integrated_mean: Array2<f32>,
    /// The (binned) raw frames the run compared against — present only for
    /// preview runs (bin > 1), where the input stack is not like-for-like.
    binned_raw: Option<Vec<Array2<f32>>>,
    /// Spatial binning factor (1 = full resolution, >1 = preview).
    bin: usize,
    elapsed_seconds: f64,
    params: CorrectionParams,
    /// What mbirtorch reported about the fit.
    report: HsntReport,
    /// The run's by-products (report, dehydrated file, plots, log), written
    /// into the export folder.
    artifacts: Vec<Artifact>,
    /// The mask the solver saw (binned for previews), when one was set.
    mask: Option<Arc<Array2<bool>>>,
    /// Pixels handed to the solver.
    n_pixels: usize,
    /// How the mask was defined, for the provenance file.
    mask_description: String,
}

impl ResultState {
    fn is_preview(&self) -> bool {
        self.bin > 1
    }

    /// Corrected-frame dimensions (h, w) — the stack's when bin = 1.
    fn dims(&self) -> (usize, usize) {
        self.integrated_mean.dim()
    }

    /// The raw frame comparable to corrected frame `i` (binned for previews).
    fn raw_frame<'a>(&'a self, stack: &'a ImageStack, i: usize) -> Option<&'a Array2<f32>> {
        match &self.binned_raw {
            Some(binned) => binned.get(i),
            None => stack.frames.get(i),
        }
    }
}

/// What the right pane of the "Corrected vs raw" view shows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RightPane {
    Raw,
    /// corrected − raw, on a symmetric color range around 0.
    Difference,
}

/// X-axis of the profile plots.
#[derive(Clone, Copy, PartialEq, Eq)]
enum XAxis {
    Index,
    TofUs,
    LambdaAngstrom,
}

impl XAxis {
    fn label(self) -> &'static str {
        match self {
            XAxis::Index => "Image index",
            XAxis::TofUs => "TOF (µs)",
            XAxis::LambdaAngstrom => "Wavelength (Å)",
        }
    }
}

/// Pixel region for the profile plots (half-open: `left..right`, `top..bottom`).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Region {
    left: usize,
    right: usize,
    top: usize,
    bottom: usize,
}

impl Region {
    fn clamped(mut self, w: usize, h: usize) -> Self {
        self.left = self.left.min(w.saturating_sub(1));
        self.right = self.right.clamp(self.left + 1, w);
        self.top = self.top.min(h.saturating_sub(1));
        self.bottom = self.bottom.clamp(self.top + 1, h);
        self
    }

    fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.left as f32 && x <= self.right as f32 && y >= self.top as f32 && y <= self.bottom as f32
    }

    /// The 8 resize handles as `((hx, vy), (image x, image y))`:
    /// hx/vy ∈ {-1, 0, 1} select the edge(s) the handle drags.
    fn handles(&self) -> [((i8, i8), (f32, f32)); 8] {
        let (x0, y0) = (self.left as f32, self.top as f32);
        let (x1, y1) = (self.right as f32, self.bottom as f32);
        let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        [
            ((-1, -1), (x0, y0)),
            ((0, -1), (mx, y0)),
            ((1, -1), (x1, y0)),
            ((-1, 0), (x0, my)),
            ((1, 0), (x1, my)),
            ((-1, 1), (x0, y1)),
            ((0, 1), (mx, y1)),
            ((1, 1), (x1, y1)),
        ]
    }
}

/// A drag in progress on the profile region. `rect` is the working rectangle
/// in image coordinates (`[x0, y0, x1, y1]`, unordered while a resize crosses
/// itself); it is committed to the integer [`Region`] on every drag frame.
struct RegionDrag {
    mode: DragMode,
    rect: [f32; 4],
}

enum DragMode {
    /// Drawing a new region; `rect[0..2]` holds the anchor corner.
    Draw,
    /// Moving the whole region; `last` is the previous pointer position.
    Move { last: (f32, f32) },
    /// Dragging the handle that controls these edges.
    Resize { hx: i8, vy: i8 },
}

fn cursor_for_handle(hx: i8, vy: i8) -> egui::CursorIcon {
    match (hx, vy) {
        (0, _) => egui::CursorIcon::ResizeVertical,
        (_, 0) => egui::CursorIcon::ResizeHorizontal,
        _ if hx == vy => egui::CursorIcon::ResizeNwSe,
        _ => egui::CursorIcon::ResizeNeSw,
    }
}

pub struct DehydrationApp {
    stack: Option<Arc<ImageStack>>,
    loading: Option<LoadJob>,
    /// Folder the images came from: names the export folder and seeds the
    /// export dialog location.
    input_dir: Option<PathBuf>,
    /// Files of the current stack (as given to [`Self::start_load`]), so a
    /// detector change can reload them.
    input_files: Vec<PathBuf>,
    /// `BL10:Exp:Det` value of the run the current stack was located from
    /// (run number lookup), feeding the automatic detector guess; `None`
    /// when the files were picked by hand.
    input_daslog: Option<String>,
    /// "🔍 Run number…" dialog (see [`run_lookup`]).
    run_lookup: RunLookup,
    /// Run number given with `--run-number`, looked up on startup.
    pending_run: Option<u32>,
    /// Detector chosen in the toolbar combobox / `--detector`, deciding how
    /// TIFF frames are oriented on load; `None` = guess from the folder
    /// layout (`images/tpx1`, `images/ikonxl`, …).
    detector_override: Option<Detector>,
    /// Recently loaded dataset folders, most recent first (persisted).
    recent: Vec<PathBuf>,
    /// IPTS chosen in the toolbar (`--ipts`, or the run of a run-number
    /// lookup): the open dialogs then start in that experiment's `shared`
    /// folder instead of the VENUS root.
    selected_ipts: Option<u32>,
    /// `IPTS-*` folders found under the VENUS root, newest first (`None`
    /// until the background scan is done).
    ipts_list: Option<Vec<u32>>,
    ipts_scan: Option<Receiver<Vec<u32>>>,
    ipts_filter: String,

    view: View,
    frame_idx: usize,
    colormap: Colormap,
    /// Track the displayed image's full range automatically until the user
    /// touches the contrast values; "Auto" re-engages it.
    contrast_auto: bool,
    vmin: f32,
    vmax: f32,
    data_min: f32,
    data_max: f32,

    /// Per-pixel sum over the raw frames (the notebook's integrated image).
    integrated_raw: Option<Array2<f32>>,
    result: Option<ResultState>,

    params: CorrectionParams,
    corr_job: Option<CorrJob>,
    export_job: Option<ExportJob>,
    last_export: Option<PathBuf>,
    /// Every line the mbirtorch command line printed during the last (or
    /// running) correction, shown by the "📜 Log" window.
    run_log: Vec<String>,
    show_log: bool,
    /// The "📈 Diagnostics" window: the spectra and maps plots of the result.
    show_diagnostics: bool,
    /// (spectra, maps) plot textures of the current result, loaded when the
    /// diagnostics window first opens.
    diag_tex: Option<[Option<TextureHandle>; 2]>,
    /// Fixed rank last typed, kept while "auto" is selected.
    fixed_rank: usize,
    /// Dose last typed, kept while the dose is unknown.
    dose_value: f64,
    /// Commit of the mbirtorch checkout, read once at startup.
    mbirtorch_commit: Option<String>,

    /// Pixel mask definition (what the Mask section edits).
    mask_spec: MaskSpec,
    /// The mask built from `mask_spec` and the integrated image; `None` =
    /// all pixels (or a build error, kept in `mask_error`).
    mask: Option<Arc<Array2<bool>>>,
    mask_dirty: bool,
    mask_error: Option<String>,
    /// Tint the excluded pixels on the image panes.
    show_mask: bool,
    /// Drag on the integrated image (raw view) adds a mask rectangle.
    mask_draw: bool,
    mask_rect_mode: RectMode,
    /// Rectangle being dragged, image coordinates `[x0, y0, x1, y1]`.
    mask_drag: Option<[f32; 4]>,
    /// Range values being edited (kept while the range is disabled).
    mask_range_values: (f32, f32),
    /// Value range of the integrated image (bounds of the range controls).
    integrated_range: (f32, f32),

    /// Right pane of the "Corrected vs raw" view.
    right_pane: RightPane,

    tex_left: Option<TextureHandle>,
    tex_right: Option<TextureHandle>,
    cbar_tex: Option<TextureHandle>,
    tex_dirty: bool,
    /// The images currently on screen (owned copies, parallel to the
    /// textures), for the cursor value read-out.
    pane_cache: (Option<Array2<f32>>, Option<Array2<f32>>),

    region: Option<Region>,
    /// Live drag on the profile region: drawing a new one, moving it, or
    /// resizing it by one of its handles.
    region_drag: Option<RegionDrag>,
    /// (uncorrected, corrected) mean intensity per frame over the region.
    profiles: Option<(Vec<f64>, Vec<f64>)>,
    /// (uncorrected, corrected) intensity per frame of the marked pixel.
    pixel_profiles: Option<(Vec<f64>, Vec<f64>)>,
    /// Pixel marked by clicking the profiles image (result coordinates).
    pixel_marker: Option<(usize, usize)>,
    profiles_dirty: bool,
    /// Plot the profiles on a log10 y-axis (non-positive values are hidden).
    log_y: bool,
    /// TOF axis (µs, one value per image) from the folder's *_Spectra.txt.
    spectra_tof_us: Option<Vec<f64>>,
    x_axis: XAxis,
    /// Source–detector distance for the wavelength conversion (m).
    distance_m: f64,
    /// Set when the offset was given on the command line (`--offset`): a
    /// stack loaded by hand then keeps it instead of the run's NeXus value.
    offset_pinned: bool,
    /// The run whose NeXus offset is in effect, to skip looking it up again
    /// when the same files are reloaded (e.g. detector change).
    offset_run: Option<u32>,
    /// Detector offset: constant added to the spectra file's TOF values, in µs
    /// (also shifts the wavelength axis).
    offset_us: f64,

    scale: f32,
    fit_requested: bool,
    /// Scroll offset to apply to the image scroll area on the next frame,
    /// set by a Ctrl+wheel zoom so the pixel under the cursor stays put.
    viewer_scroll: Option<egui::Vec2>,
    cursor: Option<(usize, usize, f32)>,
    status: String,
    /// The "ℹ mbirtorch" About dialog (algorithm provenance and versions).
    show_about: bool,
    /// (imaging, mbirtorch-light-bg, mbirtorch-dark-bg) logo textures, loaded on
    /// the first frame.
    logo_tex: Option<[Option<TextureHandle>; 3]>,
}

impl Default for DehydrationApp {
    fn default() -> Self {
        Self::new()
    }
}

impl DehydrationApp {
    pub fn new() -> Self {
        Self {
            stack: None,
            loading: None,
            input_dir: None,
            input_files: Vec::new(),
            input_daslog: None,
            run_lookup: RunLookup::new(),
            pending_run: None,
            detector_override: None,
            recent: crate::recent::load(),
            selected_ipts: None,
            ipts_list: None,
            ipts_scan: None,
            ipts_filter: String::new(),
            view: View::Raw,
            frame_idx: 0,
            colormap: Colormap::Viridis,
            contrast_auto: true,
            vmin: 0.0,
            vmax: 1.0,
            data_min: 0.0,
            data_max: 1.0,
            integrated_raw: None,
            result: None,
            params: CorrectionParams::default(),
            corr_job: None,
            export_job: None,
            last_export: None,
            run_log: Vec::new(),
            show_log: false,
            show_diagnostics: false,
            diag_tex: None,
            fixed_rank: 2,
            dose_value: 100.0,
            mbirtorch_commit: hsnt_cli::mbirtorch_commit(),
            mask_spec: MaskSpec::default(),
            mask: None,
            mask_dirty: false,
            mask_error: None,
            show_mask: true,
            mask_draw: false,
            mask_rect_mode: RectMode::Include,
            mask_drag: None,
            mask_range_values: (0.0, 1.0),
            integrated_range: (0.0, 1.0),
            right_pane: RightPane::Raw,
            tex_left: None,
            tex_right: None,
            cbar_tex: None,
            tex_dirty: false,
            pane_cache: (None, None),
            region: None,
            region_drag: None,
            profiles: None,
            pixel_profiles: None,
            pixel_marker: None,
            profiles_dirty: false,
            log_y: false,
            spectra_tof_us: None,
            x_axis: XAxis::Index,
            distance_m: spectra::DEFAULT_DISTANCE_M,
            offset_pinned: false,
            offset_run: None,
            offset_us: 0.0,
            scale: 1.0,
            fit_requested: false,
            viewer_scroll: None,
            cursor: None,
            status: "Open a folder of TIFF images to begin.".to_owned(),
            show_about: false,
            logo_tex: None,
        }
    }

    /// Set the detector offset: a constant added to the spectra file's TOF
    /// values, in µs (the `-t/--offset` command-line option).
    pub fn set_detector_offset(&mut self, offset_us: f64) {
        self.offset_us = offset_us;
        self.offset_pinned = true;
    }

    /// Correction parameters given on the command line.
    pub fn set_params(&mut self, params: CorrectionParams) {
        self.params = params;
        if let Rank::Fixed(n) = params.rank {
            self.fixed_rank = n;
        }
        if let Some(d) = params.dose {
            self.dose_value = d;
        }
    }

    /// The two logos, side by side. The MBIRTORCH variant matching the active
    /// theme is shown: the official transparent PNG (dark text) on light,
    /// the white-on-black variant on dark.
    fn logos_row(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let [imaging, mbirtorch_light, mbirtorch_dark] = self.logo_tex.get_or_insert_with(|| {
            [
                load_logo(&ctx, "imaging_logo", IMAGING_LOGO_BYTES),
                load_logo(&ctx, "mbirtorch_logo_light", MBIRTORCH_LOGO_LIGHT_BYTES),
                load_logo(&ctx, "mbirtorch_logo_dark", MBIRTORCH_LOGO_DARK_BYTES),
            ]
        });
        let mbirtorch = match ctx.theme() {
            egui::Theme::Dark => mbirtorch_dark,
            egui::Theme::Light => mbirtorch_light,
        };
        ui.horizontal(|ui| {
            if let Some(tex) = imaging {
                ui.add(egui::Image::from_texture(&*tex).max_height(LOGO_HEIGHT))
                    .on_hover_text("Neutron Imaging — Oak Ridge National Laboratory");
            }
            if let Some(tex) = mbirtorch {
                ui.add(egui::Image::from_texture(&*tex).max_height(LOGO_HEIGHT))
                    .on_hover_text("MBIRTorch — Purdue University (hsnt package by Harel Dor)");
            }
        });
    }

    // ----- about dialog ------------------------------------------------------

    /// Modal showing what algorithm this tool runs and which mbirtorch version
    /// the implementation is a port of.
    fn about_modal(&mut self, ctx: &egui::Context) {
        if !self.show_about {
            return;
        }
        let modal = egui::Modal::new(egui::Id::new("about_modal")).show(ctx, |ui| {
            ui.set_max_width(520.0);
            ui.heading("Dehydration / Hydration Correction — beta (mbirtorch hsnt)");
            ui.label(format!("Application version {}", env!("CARGO_PKG_VERSION")));
            ui.separator();
            ui.add_space(4.0);
            ui.label(
                "This beta runs the new dehydration/hydration of mbirtorch's hsnt package \
                 (Harel Dor): the maximum-likelihood factorization X = W·H of the \
                 attenuation under the Poisson statistics of the counts, with the number \
                 of components estimated by likelihood-ratio tests when it is not given. \
                 The production tool runs a native port of the earlier least-squares NMF.",
            );
            ui.add_space(6.0);
            ui.label(
                "The correction is `python -m mbirtorch.hsnt denoise` on the loaded stack, \
                 in the pixi environment of the checkout below; the stack travels through \
                 a scratch HDF5 file.",
            );
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "mbirtorch checkout: {}\nbranch {}, commit {}",
                    hsnt_cli::MBIRTORCH_CHECKOUT,
                    hsnt_cli::MBIRTORCH_BRANCH,
                    self.mbirtorch_commit.as_deref().unwrap_or("not found")
                ))
                .strong(),
            );
            ui.label(
                egui::RichText::new(format!("Python: {}", hsnt_cli::python().display())).small(),
            );
            ui.add_space(6.0);
            ui.label("Algorithm reference:");
            ui.label(
                egui::RichText::new(
                    "M. S. N. Chowdhury, D. Yang, S. Tang, S. V. Venkatakrishnan, \
                     H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman, \"Fast Hyperspectral \
                     Neutron Tomography\", IEEE Transactions on Computational Imaging, \
                     vol. 11, pp. 663–677, 2025. doi:10.1109/TCI.2025.3567854",
                )
                .small(),
            );
            ui.hyperlink_to(
                "mbirtorch hsnt documentation",
                "https://mbirtorch.readthedocs.io/en/latest/usr_hsnt.html",
            );
            ui.add_space(10.0);
            ui.vertical_centered(|ui| {
                if ui.button("  Close  ").clicked() {
                    self.show_about = false;
                }
            });
        });
        if modal.should_close() {
            self.show_about = false;
        }
    }

    // ----- mbirtorch log and diagnostics windows -------------------------------

    /// Everything the mbirtorch command line printed (stderr log + stdout).
    fn log_window(&mut self, ctx: &egui::Context) {
        if !self.show_log {
            return;
        }
        let running = self.corr_job.is_some();
        let mut text = self.run_log.join("\n");
        let n_lines = self.run_log.len();
        let warnings: Vec<String> =
            self.result.as_ref().map(|r| r.report.warnings.clone()).unwrap_or_default();
        let mut open = true;
        egui::Window::new("📜 mbirtorch hsnt log")
            .open(&mut open)
            .default_size([780.0, 420.0])
            .resizable(true)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(format!(
                        "{n_lines} line(s){}",
                        if running { " — running" } else { "" }
                    ));
                    if ui.button("Copy").clicked() {
                        ui.ctx().copy_text(text.clone());
                    }
                });
                if !warnings.is_empty() && !running {
                    ui.separator();
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 160, 30),
                        format!("⚠ {} warning(s) in the last run:", warnings.len()),
                    );
                    for w in &warnings {
                        ui.label(format!("  • {w}"));
                    }
                    ui.separator();
                }
                egui::ScrollArea::vertical()
                    .stick_to_bottom(running)
                    .show(ui, |ui| {
                        ui.add(
                            egui::TextEdit::multiline(&mut text)
                                .code_editor()
                                .desired_width(f32::INFINITY),
                        );
                    });
            });
        if !open {
            self.show_log = false;
        }
    }

    /// The spectra and maps PNGs mbirtorch plotted for the current result,
    /// with its data checks and memory plan.
    fn diagnostics_window(&mut self, ctx: &egui::Context) {
        if !self.show_diagnostics {
            return;
        }
        let Some(result) = &self.result else {
            self.show_diagnostics = false;
            return;
        };
        if self.diag_tex.is_none() {
            let find = |name: &str| {
                result
                    .artifacts
                    .iter()
                    .find(|a| a.name == name)
                    .and_then(|a| load_logo(ctx, &format!("diag_{name}"), &a.bytes))
            };
            let textures = [find("hsnt_spectra.png"), find("hsnt_maps.png")];
            self.diag_tex = Some(textures);
        }
        let Some(result) = &self.result else { return };
        let textures = self.diag_tex.clone().unwrap_or([None, None]);
        let report = result.report.clone();
        let mut open = true;
        egui::Window::new("📈 mbirtorch hsnt diagnostics")
            .open(&mut open)
            .default_size([900.0, 700.0])
            .resizable(true)
            .show(ctx, |ui| {
                if !report.summary_line.is_empty() {
                    ui.label(egui::RichText::new(&report.summary_line).small());
                }
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let avail = ui.available_width();
                    for (title, tex) in [
                        ("Component spectra", &textures[0]),
                        ("Component maps", &textures[1]),
                    ] {
                        ui.heading(title);
                        match tex {
                            Some(tex) => {
                                ui.add(
                                    egui::Image::from_texture(tex)
                                        .max_width(avail)
                                        .max_height(640.0),
                                );
                            }
                            None => {
                                ui.weak("not available");
                            }
                        }
                        ui.add_space(8.0);
                    }
                    if !report.checks.is_empty() {
                        ui.heading("Data checks");
                        for (level, msg) in &report.checks {
                            let text = format!("[{level}] {msg}");
                            match level.as_str() {
                                "ok" => {
                                    ui.label(text);
                                }
                                "warn" => {
                                    ui.colored_label(ui.visuals().warn_fg_color, text);
                                }
                                _ => {
                                    ui.colored_label(ui.visuals().error_fg_color, text);
                                }
                            }
                        }
                    }
                    if !report.memory_plan.is_empty() {
                        ui.heading("Memory plan");
                        ui.label(&report.memory_plan);
                    }
                });
            });
        if !open {
            self.show_diagnostics = false;
        }
    }

    // ----- mask ----------------------------------------------------------------

    /// Rebuild the pixel mask from its definition and the integrated image
    /// when either changed.
    fn rebuild_mask(&mut self) {
        if !self.mask_dirty {
            return;
        }
        self.mask_dirty = false;
        let Some(integrated) = &self.integrated_raw else {
            self.mask = None;
            self.mask_error = None;
            return;
        };
        match self.mask_spec.build(integrated) {
            Ok(m) => {
                self.mask = m.map(Arc::new);
                self.mask_error = None;
            }
            Err(e) => {
                self.mask = None;
                self.mask_error = Some(format!("{e:#}"));
            }
        }
        self.tex_dirty = true;
    }

    /// The mask matching the images on screen: the stack's in the raw view,
    /// the one the run saw (binned for previews) in the result views.
    fn display_mask(&self) -> Option<Arc<Array2<bool>>> {
        match self.view {
            View::Raw => self.mask.clone(),
            View::Result | View::Profiles => self.result.as_ref().and_then(|r| r.mask.clone()),
        }
    }

    fn load_mask_dialog(&mut self, stack: &ImageStack) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Load a mask image (nonzero = selected pixel)")
            .add_filter("Mask image", &["tif", "tiff", "npy"]);
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_file() else { return };
        match mask::load_file(&path, stack.detector) {
            Ok(pixels) => {
                let n = mask::count(&pixels);
                self.mask_spec.file = Some(MaskFile {
                    path: path.clone(),
                    pixels: Arc::new(pixels),
                });
                self.mask_dirty = true;
                self.status = format!("Mask loaded from {} ({n} selected pixel(s))", path.display());
            }
            Err(e) => self.status = format!("Loading the mask failed: {e:#}"),
        }
    }

    fn save_mask_dialog(&mut self, stack: &ImageStack) {
        let Some(m) = self.mask.clone() else { return };
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save the mask as an 8-bit TIFF (255 = selected pixel)")
            .add_filter("TIFF", &["tif", "tiff"])
            .set_file_name("dehydration_hydration_mask.tif");
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else { return };
        match mask::save_file(&path, &m, stack.orientation) {
            Ok(()) => self.status = format!("Mask saved to {}", path.display()),
            Err(e) => self.status = format!("Saving the mask failed: {e:#}"),
        }
    }

    /// The "Mask" section of the left panel.
    fn mask_section(&mut self, ui: &mut egui::Ui, stack: &ImageStack) {
        let total = stack.height * stack.width;
        egui::CollapsingHeader::new(egui::RichText::new("Mask").heading())
            .default_open(true)
            .show(ui, |ui| {
                ui.label(
                    egui::RichText::new(
                        "Only the selected pixels are sent to mbirtorch; the others keep \
                         their raw values in the corrected stack. Leave everything off to \
                         use all pixels.",
                    )
                    .small()
                    .weak(),
                );
                let running = self.corr_job.is_some();
                ui.add_enabled_ui(!running, |ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .toggle_value(&mut self.mask_draw, "✏ Draw rectangles")
                            .on_hover_text(
                                "Drag on the integrated image (right pane of the Raw data \
                                 view) to add a rectangle",
                            )
                            .changed()
                            && self.mask_draw
                        {
                            self.view = View::Raw;
                            self.tex_dirty = true;
                        }
                        egui::ComboBox::from_id_salt("mask_rect_mode")
                            .selected_text(self.mask_rect_mode.label())
                            .show_ui(ui, |ui| {
                                for m in [RectMode::Include, RectMode::Exclude] {
                                    ui.selectable_value(&mut self.mask_rect_mode, m, m.label());
                                }
                            })
                            .response
                            .on_hover_text(
                                "include: keep only the pixels inside (any of) the include \
                                 rectangles; exclude: drop the pixels inside",
                            );
                    });
                    let mut remove: Option<usize> = None;
                    for (i, r) in self.mask_spec.rects.iter().enumerate() {
                        ui.horizontal(|ui| {
                            let color = match r.mode {
                                RectMode::Include => MASK_INCLUDE_COLOR,
                                RectMode::Exclude => MASK_EXCLUDE_COLOR,
                            };
                            ui.colored_label(color, "■");
                            ui.label(egui::RichText::new(r.label()).small());
                            if ui.small_button("✖").on_hover_text("Remove this rectangle").clicked() {
                                remove = Some(i);
                            }
                        });
                    }
                    if let Some(i) = remove {
                        self.mask_spec.rects.remove(i);
                        self.mask_dirty = true;
                    }

                    let mut has_range = self.mask_spec.range.is_some();
                    let (lo_all, hi_all) = self.integrated_range;
                    ui.horizontal(|ui| {
                        if ui
                            .checkbox(&mut has_range, "Integrated value in")
                            .on_hover_text(
                                "Keep the pixels whose summed intensity over the stack (the \
                                 integrated image) lies in this range — e.g. to leave out the \
                                 open beam around the sample",
                            )
                            .changed()
                        {
                            self.mask_dirty = true;
                        }
                        let speed = ((hi_all - lo_all) / 500.0).max(1e-6);
                        let (mut lo, mut hi) = self.mask_range_values;
                        let r1 = ui.add_enabled(
                            has_range,
                            egui::DragValue::new(&mut lo).speed(speed).range(lo_all..=hi_all),
                        );
                        ui.label("..");
                        let r2 = ui.add_enabled(
                            has_range,
                            egui::DragValue::new(&mut hi).speed(speed).range(lo_all..=hi_all),
                        );
                        if r1.changed() || r2.changed() {
                            if lo > hi {
                                std::mem::swap(&mut lo, &mut hi);
                            }
                            self.mask_range_values = (lo, hi);
                            self.mask_dirty = true;
                        }
                    });
                    self.mask_spec.range = has_range.then_some(self.mask_range_values);

                    ui.horizontal(|ui| {
                        if ui
                            .button("📂 Load mask…")
                            .on_hover_text(
                                "A mask image (TIFF or .npy, nonzero = selected), read with the \
                                 stack's detector orientation — e.g. one saved here or made with \
                                 the Hyperspectral Masker",
                            )
                            .clicked()
                        {
                            self.load_mask_dialog(stack);
                        }
                        if ui
                            .add_enabled(self.mask.is_some(), egui::Button::new("💾 Save mask…"))
                            .on_hover_text("Write the current mask as an 8-bit TIFF (255 = selected)")
                            .clicked()
                        {
                            self.save_mask_dialog(stack);
                        }
                        if ui
                            .add_enabled(!self.mask_spec.is_empty(), egui::Button::new("Clear"))
                            .on_hover_text("Remove every rectangle, the range and the file: all pixels")
                            .clicked()
                        {
                            self.mask_spec = MaskSpec::default();
                            self.mask_dirty = true;
                        }
                    });
                    if let Some(path) = self.mask_spec.file.as_ref().map(|f| f.path.clone()) {
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(format!("file: {}", path.display()))
                                    .small()
                                    .weak(),
                            )
                            .on_hover_text(path.display().to_string());
                            if ui.small_button("✖").on_hover_text("Drop the mask file").clicked() {
                                self.mask_spec.file = None;
                                self.mask_dirty = true;
                            }
                        });
                    }
                });
                if ui
                    .checkbox(&mut self.show_mask, "Show mask (tint the excluded pixels)")
                    .changed()
                {
                    self.tex_dirty = true;
                }
                if let Some(e) = &self.mask_error {
                    ui.colored_label(ui.visuals().error_fg_color, format!("Mask error: {e}"));
                } else {
                    match &self.mask {
                        Some(m) => {
                            let n = mask::count(m);
                            ui.label(format!(
                                "{n} of {total} pixels selected ({:.1}%)",
                                100.0 * n as f64 / total.max(1) as f64
                            ));
                        }
                        None => {
                            ui.weak(format!("No mask: all {total} pixels"));
                        }
                    }
                }
            });
    }

    // ----- loading -----------------------------------------------------------

    /// Force the detector (hence the orientation) the stack is loaded with,
    /// `None` to go back to the automatic guess (`--detector`).
    pub fn set_detector_override(&mut self, detector: Option<Detector>) {
        self.detector_override = detector;
    }

    /// Locate `run` and load its data when the window opens
    /// (`--run-number`).
    pub fn set_startup_run(&mut self, run: u32) {
        self.pending_run = Some(run);
    }

    /// Detector selection for `paths`: the run's DASlog when the files were
    /// located from a run number, else the folder-layout guess — both
    /// overridden by the toolbar choice.
    fn detector_for(&self, paths: &[PathBuf], daslog: Option<&str>) -> Selection {
        loader::detect_detector(paths, daslog, self.detector_override)
    }

    /// Toolbar combobox choosing the detector (hence the orientation on
    /// load): "auto" follows the folder layout, the other entries force one.
    /// Changing it reloads the current stack.
    fn detector_combo(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.label("Detector:").on_hover_text(
            "How the TIFF frames are oriented on load: Timepix → transposed, CCD → flipped \
             vertically, QHY → rotated 90° counterclockwise. 'auto' recognizes the detector from \
             the folder layout (images/tpx1, images/ikonxl, …).",
        );
        let auto_text = match self.stack.as_ref() {
            Some(s) if s.detector.is_auto() => format!("auto: {}", s.detector.summary()),
            _ => "auto".to_owned(),
        };
        let current = match self.detector_override {
            None => auto_text.clone(),
            Some(d) => d.label().to_owned(),
        };
        let mut changed = false;
        egui::ComboBox::from_id_salt("detector")
            .selected_text(current)
            .show_ui(ui, |ui| {
                if ui
                    .selectable_label(self.detector_override.is_none(), auto_text)
                    .on_hover_text("Guess the detector from the folder layout")
                    .clicked()
                    && self.detector_override.is_some()
                {
                    self.detector_override = None;
                    changed = true;
                }
                for d in Detector::ALL {
                    if ui
                        .selectable_label(self.detector_override == Some(d), d.label())
                        .on_hover_text(d.description())
                        .clicked()
                        && self.detector_override != Some(d)
                    {
                        self.detector_override = Some(d);
                        changed = true;
                    }
                }
            });
        if let Some(s) = self.stack.as_ref() {
            ui.label(egui::RichText::new(s.orientation.label()).weak())
                .on_hover_text(s.detector.detector().description());
        }
        if changed && !self.input_files.is_empty() && self.loading.is_none() {
            let files = self.input_files.clone();
            let daslog = self.input_daslog.clone();
            self.start_load_from(files, daslog, ctx);
        }
    }

    /// Load `paths` (picked by hand: the detector is guessed from the folder
    /// layout).
    pub fn start_load(&mut self, paths: Vec<PathBuf>, ctx: &egui::Context) {
        self.start_load_from(paths, None, ctx);
    }

    /// Load `paths`, with the `BL10:Exp:Det` DASlog value of their run when
    /// they were located from a run number. Paths picked by hand whose
    /// names carry a run number (`…_Run_<run>_…`) get the detector offset
    /// of that run from its NeXus file, unless `--offset` pinned one.
    /// Returns whether the load started (`false` while a correction or an
    /// export is running).
    fn start_load_from(&mut self, paths: Vec<PathBuf>, daslog: Option<String>, ctx: &egui::Context) -> bool {
        if paths.is_empty() {
            return false;
        }
        if self.corr_job.is_some() || self.export_job.is_some() {
            self.status = "Wait for the running job to finish (or cancel it) first.".to_owned();
            return false;
        }
        let total = paths.len();
        let detector = self.detector_for(&paths, daslog.as_deref());
        // A run located from its number already applied its offset.
        let lookup_run = if daslog.is_none() && !self.offset_pinned {
            paths
                .first()
                .and_then(|p| run_lookup::run_number_of(p))
                .filter(|&run| self.offset_run != Some(run))
        } else {
            None
        };
        self.input_files = paths.clone();
        self.input_daslog = daslog;
        let (tx, rx) = std::sync::mpsc::channel();
        // Loading is parallel, so the progress callback fires from worker
        // threads; the channel sender goes behind a mutex (Sender is !Sync).
        let progress_tx = std::sync::Mutex::new(tx.clone());
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let result = loader::load_paths_with_progress(&paths, detector, |done, total| {
                if let Ok(sender) = progress_tx.lock() {
                    let _ = sender.send(LoadMsg::Progress { done, total });
                }
                ctx.request_repaint();
            });
            if let Some(run) = lookup_run {
                let result = run_lookup::run_offset(run).map_err(|e| format!("{e:#}"));
                let _ = tx.send(LoadMsg::RunOffset { run, result });
            }
            let _ = tx.send(LoadMsg::Done(result));
            ctx.request_repaint();
        });
        self.loading = Some(LoadJob { rx, done: 0, total, run_offset: None });
        self.status = format!("Loading {total} file(s)…");
        true
    }

    fn poll_load(&mut self) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                    }
                    LoadMsg::RunOffset { run, result } => job.run_offset = Some((run, result)),
                    LoadMsg::Done(res) => result = Some(res),
                }
            }
        }
        if let Some(res) = result {
            let run_offset = self.loading.take().and_then(|job| job.run_offset);
            match res {
                Ok(stack) => {
                    self.apply_stack(stack);
                    if let Some((run, result)) = run_offset {
                        self.apply_run_offset(run, result);
                    }
                }
                Err(e) => self.status = format!("Load failed: {e:#}"),
            }
        }
    }

    /// Apply the detector offset looked up from the run number in the file
    /// names, and say so in the status line (after the load summary).
    fn apply_run_offset(&mut self, run: u32, result: Result<(PathBuf, Option<f64>), String>) {
        let note = match result {
            Ok((nexus, Some(offset))) => {
                self.offset_us = offset;
                self.offset_run = Some(run);
                let file = nexus
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                format!("Run {run}: detector offset {offset:.3} µs from {file}.")
            }
            Ok((nexus, None)) => format!(
                "Run {run}: no {} log in {} — detector offset kept at {:.3} µs.",
                run_lookup::OFFSET_LOG,
                nexus.display(),
                self.offset_us
            ),
            Err(e) => format!("Run {run}: detector offset not looked up ({e}).",),
        };
        self.status = format!("{} {note}", self.status.trim_end());
    }

    fn apply_stack(&mut self, stack: ImageStack) {
        let n = stack.n_frames();
        let (w, h) = (stack.width, stack.height);
        self.input_dir = stack
            .sources
            .first()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf());
        if let Some(dir) = &self.input_dir {
            self.recent = crate::recent::add(dir);
        }

        let mut acc = Array2::<f32>::zeros((h, w));
        for f in &stack.frames {
            acc += f;
        }
        self.integrated_range = finite_range(&acc);
        if self.mask_spec.range.is_none() {
            self.mask_range_values = self.integrated_range;
        }
        self.integrated_raw = Some(acc);
        self.mask_dirty = true;
        self.mask_drag = None;

        // TOF axis from the folder's *_Spectra.txt, when it matches the
        // stack (one TOF value per image).
        self.spectra_tof_us = self
            .input_dir
            .as_ref()
            .and_then(|dir| spectra::find_spectra_file(dir))
            .and_then(|path| spectra::load_tof_us(&path).ok())
            .filter(|tof| tof.len() == n);
        if self.spectra_tof_us.is_none() {
            self.x_axis = XAxis::Index;
        }

        self.stack = Some(Arc::new(stack));
        self.result = None;
        self.profiles = None;
        self.pixel_profiles = None;
        self.pixel_marker = None;
        self.region = None;
        self.last_export = None;
        self.view = View::Raw;
        self.frame_idx = 0;
        self.contrast_auto = true;
        self.fit_requested = true;
        self.tex_dirty = true;
        let spectra_note = if self.spectra_tof_us.is_some() {
            " TOF axis found (Spectra.txt)."
        } else {
            ""
        };
        let (detector, orientation) = {
            let s = self.stack.as_ref().expect("stack just set");
            (s.detector.summary(), s.orientation)
        };
        self.status = format!(
            "Loaded {n} image(s), {w}×{h} px, {detector}: {orientation}.{spectra_note}"
        );
    }

    // ----- correction job ----------------------------------------------------

    fn start_correction_job(&mut self, ctx: &egui::Context, bin: usize) {
        let Some(stack) = &self.stack else { return };
        if let Err(e) = self.params.validate() {
            self.status = format!("Cannot run: {e}");
            return;
        }
        if let Some(e) = &self.mask_error {
            self.status = format!("Cannot run: mask error — {e}");
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let rx = start_correction(
            stack.clone(),
            self.params,
            bin,
            self.mask.clone(),
            cancel.clone(),
            ctx.clone(),
        );
        self.corr_job = Some(CorrJob {
            rx,
            stage: "Starting…".to_owned(),
            fraction: 0.0,
            cancel,
        });
        self.run_log.clear();
        let mode = if bin > 1 {
            format!("preview, {bin}×{bin} binned, ")
        } else {
            String::new()
        };
        let masked = match &self.mask {
            Some(m) => format!(", {} of {} px", mask::count(m), m.len()),
            None => String::new(),
        };
        self.status = format!("Correction running ({mode}{}{masked})…", self.params.summary());
    }

    fn poll_correction(&mut self) {
        let mut done = None;
        if let Some(job) = &mut self.corr_job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    CorrectionMsg::Progress { stage, fraction } => {
                        job.stage = stage;
                        job.fraction = fraction;
                    }
                    CorrectionMsg::Log(line) => self.run_log.push(line),
                    CorrectionMsg::Done(res) => done = Some(res),
                }
            }
        }
        let Some(res) = done else { return };
        self.corr_job = None;
        match res {
            Ok(out) => {
                let (h, w) = out.integrated_mean.dim();
                let preview_note = if out.is_preview() {
                    format!(" — PREVIEW at {0}×{0} binning", out.bin)
                } else {
                    String::new()
                };
                let rank = out
                    .report
                    .rank
                    .map(|r| format!("{r} material(s)"))
                    .unwrap_or_else(|| "number of materials unknown".to_owned());
                let chi2 = out
                    .report
                    .reduced_chi2
                    .map(|c| format!(", reduced chi-square {c:.2}"))
                    .unwrap_or_default();
                // Warnings are not spelled out here: the status bar shows a
                // button that opens the log, with them listed on top.
                self.status = format!(
                    "Correction done in {:.1} s ({rank}{chi2}){preview_note}.",
                    out.elapsed_seconds
                );
                self.result = Some(ResultState {
                    frames: Arc::new(out.frames),
                    integrated_mean: out.integrated_mean,
                    binned_raw: out.binned_raw,
                    bin: out.bin,
                    elapsed_seconds: out.elapsed_seconds,
                    params: self.params,
                    report: out.report,
                    artifacts: out.artifacts,
                    mask: out.mask.map(Arc::new),
                    n_pixels: out.n_pixels,
                    mask_description: self.mask_spec.describe(),
                });
                self.diag_tex = None;
                self.pixel_marker = None;
                self.pixel_profiles = None;
                // The notebook's default profile region: the second quarter
                // of the image in both directions.
                self.region = Some(
                    Region {
                        left: w / 4,
                        right: 2 * w / 4,
                        top: h / 4,
                        bottom: 2 * h / 4,
                    }
                    .clamped(w, h),
                );
                self.profiles_dirty = true;
                self.view = View::Result;
                self.contrast_auto = true;
                self.tex_dirty = true;
            }
            Err(e) if e.contains("cancelled") => {
                self.status = "Correction cancelled.".to_owned();
            }
            Err(e) => {
                self.status = format!("Correction failed: {e}");
                self.show_log = true;
            }
        }
    }

    // ----- export job --------------------------------------------------------

    fn export_dialog(&mut self, ctx: &egui::Context) {
        let Some(result) = &self.result else { return };
        let Some(stack) = &self.stack else { return };
        if result.is_preview() {
            self.status =
                "Preview results are binned — run the full correction before exporting.".to_owned();
            return;
        }
        let input_dir = self.input_dir.clone().unwrap_or_default();
        let mut dialog = rfd::FileDialog::new()
            .set_title("Choose the folder that will receive the corrected images");
        if let Some(parent) = self.input_dir.as_ref().and_then(|p| p.parent()) {
            dialog = dialog.set_directory(parent);
        }
        let Some(output_dir) = dialog.pick_folder() else {
            return;
        };
        let (h, w) = result.dims();
        let provenance = Provenance {
            input_folder: input_dir.clone(),
            num_images: result.frames.len(),
            image_width: w,
            image_height: h,
            params: result.params,
            bin: result.bin,
            elapsed_seconds: result.elapsed_seconds,
            report: result.report.clone(),
            mbirtorch_commit: self.mbirtorch_commit.clone(),
            mask: result.mask.as_ref().map(|m| MaskInfo {
                selected: result.n_pixels,
                total: m.len(),
                description: result.mask_description.clone(),
            }),
        };
        let rx = start_export(
            output_dir,
            input_dir,
            result.frames.clone(),
            stack.sources.clone(),
            stack.orientation,
            provenance,
            result.artifacts.clone(),
            ctx.clone(),
        );
        self.export_job = Some(ExportJob {
            rx,
            done: 0,
            total: result.frames.len(),
        });
        self.status = "Exporting corrected images…".to_owned();
    }

    /// Save the region profiles (plus TOF/λ columns when available) as CSV.
    fn save_profiles_csv(&mut self) {
        let Some((uncorrected, corrected)) = self.profiles.clone() else {
            return;
        };
        let mut dialog = rfd::FileDialog::new()
            .add_filter("CSV", &["csv"])
            .set_file_name("profiles.csv")
            .set_title("Save the region profiles as CSV");
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else {
            return;
        };
        // The detector offset is applied to the exported TOF column, so the
        // CSV matches the plotted axes.
        let tof: Option<Vec<f64>> = self
            .spectra_tof_us
            .as_ref()
            .map(|tof| tof.iter().map(|&t| t + self.offset_us).collect());
        let lambda: Option<Vec<f64>> = tof.as_ref().map(|tof| {
            tof.iter()
                .map(|&t| spectra::tof_us_to_lambda_angstroms(t, self.distance_m))
                .collect()
        });
        match crate::export::write_profiles_csv(
            &path,
            &uncorrected,
            &corrected,
            tof.as_deref(),
            lambda.as_deref(),
        ) {
            Ok(()) => self.status = format!("Profiles saved to {}", path.display()),
            Err(e) => self.status = format!("CSV save failed: {e:#}"),
        }
    }

    fn poll_export(&mut self) {
        let mut done = None;
        if let Some(job) = &mut self.export_job {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    ExportMsg::Progress { done, total } => {
                        job.done = done;
                        job.total = total;
                    }
                    ExportMsg::Done(res) => done = Some(res),
                }
            }
        }
        let Some(res) = done else { return };
        self.export_job = None;
        match res {
            Ok(folder) => {
                self.status = format!("Corrected images exported to {}", folder.display());
                self.last_export = Some(folder);
            }
            Err(e) => self.status = format!("Export failed: {e}"),
        }
    }

    // ----- profiles ----------------------------------------------------------

    fn recompute_profiles(&mut self) {
        if !self.profiles_dirty {
            return;
        }
        self.profiles_dirty = false;
        self.profiles = None;
        self.pixel_profiles = None;
        let (Some(stack), Some(result)) = (&self.stack, &self.result) else {
            return;
        };
        let (h, w) = result.dims();
        let n = result.frames.len();

        if let Some(region) = self.region {
            let r = region.clamped(w, h);
            let mean_of = |frame: &Array2<f32>| -> f64 {
                frame
                    .slice(s![r.top..r.bottom, r.left..r.right])
                    .mapv(f64::from)
                    .mean()
                    .unwrap_or(0.0)
            };
            let uncorrected: Vec<f64> = (0..n)
                .filter_map(|i| result.raw_frame(stack, i))
                .map(mean_of)
                .collect();
            let corrected: Vec<f64> = result.frames.iter().map(mean_of).collect();
            self.profiles = Some((uncorrected, corrected));
        }

        if let Some((py, px)) = self.pixel_marker {
            if py < h && px < w {
                let uncorrected: Vec<f64> = (0..n)
                    .filter_map(|i| result.raw_frame(stack, i))
                    .map(|f| f64::from(f[(py, px)]))
                    .collect();
                let corrected: Vec<f64> = result
                    .frames
                    .iter()
                    .map(|f| f64::from(f[(py, px)]))
                    .collect();
                self.pixel_profiles = Some((uncorrected, corrected));
            }
        }
    }

    /// X-axis values for the profile plots, per the selected axis. The
    /// detector offset is added to the TOF values (and so shifts the
    /// wavelength axis too).
    fn x_values(&self, n: usize) -> Vec<f64> {
        match (self.x_axis, &self.spectra_tof_us) {
            (XAxis::TofUs, Some(tof)) => {
                tof.iter().take(n).map(|&t| t + self.offset_us).collect()
            }
            (XAxis::LambdaAngstrom, Some(tof)) => tof
                .iter()
                .take(n)
                .map(|&t| {
                    spectra::tof_us_to_lambda_angstroms(t + self.offset_us, self.distance_m)
                })
                .collect(),
            _ => (0..n).map(|i| i as f64).collect(),
        }
    }

    // ----- textures ----------------------------------------------------------

    /// The images of the current view as owned copies: `(left, right,
    /// right_is_difference)` — right is `None` in the Profiles view (the
    /// plot takes its place).
    fn display_images(&self) -> (Option<Array2<f32>>, Option<Array2<f32>>, bool) {
        match self.view {
            View::Raw => (
                self.stack
                    .as_ref()
                    .and_then(|s| s.frames.get(self.frame_idx))
                    .cloned(),
                self.integrated_raw.clone(),
                false,
            ),
            View::Result => {
                let (Some(stack), Some(result)) = (&self.stack, &self.result) else {
                    return (None, None, false);
                };
                let corrected = result.frames.get(self.frame_idx).cloned();
                let raw = result.raw_frame(stack, self.frame_idx).cloned();
                match self.right_pane {
                    RightPane::Raw => (corrected, raw, false),
                    RightPane::Difference => {
                        let diff = match (&corrected, &raw) {
                            (Some(c), Some(r)) if c.dim() == r.dim() => Some(c - r),
                            _ => None,
                        };
                        (corrected, diff, true)
                    }
                }
            }
            View::Profiles => (
                self.result.as_ref().map(|r| r.integrated_mean.clone()),
                None,
                false,
            ),
        }
    }

    fn pane_titles(&self) -> (String, String) {
        let preview = self
            .result
            .as_ref()
            .filter(|r| r.is_preview())
            .map(|r| format!("  [PREVIEW {0}×{0} binned]", r.bin))
            .unwrap_or_default();
        match self.view {
            View::Raw => (
                format!("Raw image #{}", self.frame_idx),
                "Integrated image (sum)".to_owned(),
            ),
            View::Result => (
                format!("Corrected image #{}{preview}", self.frame_idx),
                match self.right_pane {
                    RightPane::Raw => format!("Uncorrected image #{}{preview}", self.frame_idx),
                    RightPane::Difference => {
                        format!("Corrected − raw #{}{preview}", self.frame_idx)
                    }
                },
            ),
            View::Profiles => (
                format!("Integrated corrected image (mean){preview}"),
                String::new(),
            ),
        }
    }

    fn ensure_textures(&mut self, ctx: &egui::Context) {
        if !self.tex_dirty {
            return;
        }
        let lut = self.colormap.lut();

        let (left, right, right_is_diff) = self.display_images();

        // Contrast range follows the left image while on auto.
        let range = left.as_ref().map(finite_range);
        if let Some((lo, hi)) = range {
            self.data_min = lo;
            self.data_max = hi;
            if self.contrast_auto {
                self.vmin = lo;
                self.vmax = hi;
            }
        }
        let (vmin, vmax) = (self.vmin, self.vmax);
        let mut left_color = left.as_ref().map(|img| colorize(img, vmin, vmax, &lut));
        let mut right_color = right.as_ref().map(|img| {
            let (lo, hi) = if right_is_diff {
                // Symmetric range around 0: structure in the difference
                // stands out regardless of sign.
                let (lo, hi) = finite_range(img);
                let a = lo.abs().max(hi.abs()).max(1e-12);
                (-a, a)
            } else {
                match self.view {
                    // Raw view: the integrated image has its own scale;
                    // Result view: corrected and raw share the contrast.
                    View::Raw => finite_range(img),
                    _ => (vmin, vmax),
                }
            };
            colorize(img, lo, hi, &lut)
        });
        if self.show_mask
            && let Some(m) = self.display_mask()
        {
            if let Some(c) = left_color.as_mut() {
                tint_excluded(c, &m);
            }
            if let Some(c) = right_color.as_mut()
                && !right_is_diff
            {
                tint_excluded(c, &m);
            }
        }
        self.pane_cache = (left, right);

        self.tex_left =
            left_color.map(|c| ctx.load_texture("pane_left", c, TextureOptions::NEAREST));
        self.tex_right =
            right_color.map(|c| ctx.load_texture("pane_right", c, TextureOptions::NEAREST));

        // Colorbar strip: 1×256, high values at the top.
        let mut cbuf = vec![0u8; 256 * 4];
        for j in 0..256 {
            let [r, g, b] = lut[255 - j];
            cbuf[j * 4] = r;
            cbuf[j * 4 + 1] = g;
            cbuf[j * 4 + 2] = b;
            cbuf[j * 4 + 3] = 255;
        }
        let cbar = egui::ColorImage::from_rgba_unmultiplied([1, 256], &cbuf);
        self.cbar_tex = Some(ctx.load_texture("colorbar", cbar, TextureOptions::LINEAR));

        self.tex_dirty = false;
    }

    // ----- IPTS / dialog start folder -----------------------------------------

    /// Pre-select an IPTS (`--ipts`): the open dialogs start in its `shared`
    /// folder.
    pub fn set_ipts(&mut self, ipts: Option<u32>) {
        self.selected_ipts = ipts;
    }

    /// Where the *Open Folder…* / *Open Files…* dialogs start: the selected
    /// IPTS's `shared` folder (the IPTS root when it has no `shared`), or the
    /// VENUS root when no IPTS is selected.
    fn dialog_start_dir(&self) -> PathBuf {
        let root = PathBuf::from(crate::run_lookup::VENUS_ROOT);
        let Some(ipts) = self.selected_ipts else {
            return root;
        };
        let ipts_dir = root.join(format!("IPTS-{ipts}"));
        let shared = ipts_dir.join("shared");
        if shared.is_dir() {
            shared
        } else if ipts_dir.is_dir() {
            ipts_dir
        } else {
            root
        }
    }

    /// List the `IPTS-*` folders of the VENUS root on a background thread
    /// (the root is on a network file system — the toolbar must not wait).
    fn start_ipts_scan(&mut self, ctx: &egui::Context) {
        if self.ipts_list.is_some() || self.ipts_scan.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        self.ipts_scan = Some(rx);
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let mut found: Vec<u32> = std::fs::read_dir(crate::run_lookup::VENUS_ROOT)
                .into_iter()
                .flatten()
                .flatten()
                .filter_map(|e| {
                    e.file_name()
                        .to_str()?
                        .strip_prefix("IPTS-")?
                        .parse::<u32>()
                        .ok()
                })
                .collect();
            found.sort_unstable_by(|a, b| b.cmp(a));
            found.dedup();
            let _ = tx.send(found);
            ctx.request_repaint();
        });
    }

    /// Toolbar "IPTS" drop-down: the experiments found under the VENUS root,
    /// newest first, with a filter box; "none" goes back to the VENUS root.
    fn ipts_combo(&mut self, ui: &mut egui::Ui) {
        self.start_ipts_scan(ui.ctx());
        if let Some(rx) = &self.ipts_scan
            && let Ok(list) = rx.try_recv()
        {
            self.ipts_list = Some(list);
            self.ipts_scan = None;
        }
        let selected_text = match self.selected_ipts {
            Some(n) => format!("IPTS-{n}"),
            None => "IPTS: none".to_owned(),
        };
        egui::ComboBox::from_id_salt("ipts_combo")
            .selected_text(selected_text)
            .width(150.0)
            .show_ui(ui, |ui| {
                ui.set_min_width(180.0);
                match &self.ipts_list {
                    None => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Scanning {} …", crate::run_lookup::VENUS_ROOT));
                        });
                        ui.ctx()
                            .request_repaint_after(std::time::Duration::from_millis(200));
                    }
                    Some(list) => {
                        let list = list.clone();
                        ui.horizontal(|ui| {
                            ui.label("Filter:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.ipts_filter)
                                    .desired_width(80.0),
                            );
                        });
                        let filter = self.ipts_filter.trim().to_owned();
                        if ui
                            .selectable_label(self.selected_ipts.is_none(), "none (VENUS root)")
                            .clicked()
                        {
                            self.selected_ipts = None;
                        }
                        egui::ScrollArea::vertical()
                            .max_height(260.0)
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                let mut any = false;
                                for n in list {
                                    if !filter.is_empty() && !n.to_string().contains(&filter) {
                                        continue;
                                    }
                                    any = true;
                                    if ui
                                        .selectable_label(
                                            self.selected_ipts == Some(n),
                                            format!("IPTS-{n}"),
                                        )
                                        .clicked()
                                    {
                                        self.selected_ipts = Some(n);
                                    }
                                }
                                if !any {
                                    ui.label("No IPTS matches the filter.");
                                }
                            });
                    }
                }
            })
            .response
            .on_hover_text(format!(
                "Experiment the open dialogs start in: {}/IPTS-<n>/shared (none: {})",
                crate::run_lookup::VENUS_ROOT,
                crate::run_lookup::VENUS_ROOT
            ));
    }

    // ----- dialogs -----------------------------------------------------------

    fn open_files_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new()
            .add_filter("Images", loader::SUPPORTED_EXTENSIONS)
            .set_title("Open TIFF image(s)")
            .set_directory(self.dialog_start_dir())
            .pick_files()
        {
            self.start_load(files, ctx);
        }
    }

    /// The "🕒 Recent" drop-down: the last [`crate::recent::MAX_RECENT`]
    /// dataset folders, most recent first. Folders that no longer exist are
    /// shown disabled.
    fn recent_menu(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.set_min_width(280.0);
        for dir in self.recent.clone() {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| dir.display().to_string());
            let exists = dir.is_dir();
            let label = if exists {
                name
            } else {
                format!("{name}  (missing)")
            };
            if ui
                .add_enabled(exists, egui::Button::new(label).wrap_mode(egui::TextWrapMode::Extend))
                .on_hover_text(dir.display().to_string())
                .on_disabled_hover_text(format!("{} no longer exists", dir.display()))
                .clicked()
            {
                match loader::list_supported_in_dir(&dir) {
                    Ok(files) => self.start_load(files, ctx),
                    Err(e) => self.status = format!("{e:#}"),
                }
                ui.close();
            }
        }
    }

    fn open_folder_dialog(&mut self, ctx: &egui::Context) {
        if let Some(dir) = rfd::FileDialog::new()
            .set_title("Open the folder containing the images to correct")
            .set_directory(self.dialog_start_dir())
            .pick_folder()
        {
            match loader::list_supported_in_dir(&dir) {
                Ok(files) => self.start_load(files, ctx),
                Err(e) => self.status = format!("{e:#}"),
            }
        }
    }

    // ----- drag & drop ---------------------------------------------------------

    /// Files or folders dropped onto the window are loaded exactly as the
    /// *Open Folder…* / *Open Files…* buttons would: a dropped folder loads
    /// its images (looking into subfolders when it holds none itself),
    /// dropped TIFF / `.npy` files load as the stack. Only one folder is
    /// taken per drop — the stack is one dataset. Ignored while a job runs.
    /// While something is dragged over the window, a dark overlay says what
    /// can be dropped.
    fn handle_drops(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let rect = ctx.content_rect();
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop_overlay"),
            ));
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha(170));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop a folder of TIFF images, or TIFF / .npy files",
                egui::FontId::proportional(22.0),
                Color32::WHITE,
            );
        }
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        if dropped.is_empty() {
            return;
        }
        if self.loading.is_some() || self.corr_job.is_some() || self.export_job.is_some() {
            self.status = "Wait for the running job to finish (or cancel it) first.".to_owned();
            return;
        }
        let mut folders: Vec<PathBuf> = Vec::new();
        let mut files: Vec<PathBuf> = Vec::new();
        let mut rejected: Vec<String> = Vec::new();
        for path in dropped {
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .unwrap_or_default();
            if path.is_dir() {
                folders.push(path);
            } else if path.is_file() && loader::SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
                files.push(path);
            } else {
                rejected.push(
                    path.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| path.display().to_string()),
                );
            }
        }
        let mut notes: Vec<String> = Vec::new();
        if !rejected.is_empty() {
            notes.push(format!(
                "ignored (not a folder, TIFF or .npy): {}",
                rejected.join(", ")
            ));
        }
        if let Some(dir) = folders.first() {
            if folders.len() > 1 || !files.is_empty() {
                notes.push(format!(
                    "one dataset at a time — loading only {}",
                    dir.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| dir.display().to_string())
                ));
            }
            match loader::list_supported_in_dir(dir) {
                Ok(paths) => {
                    self.start_load(paths, ctx);
                }
                Err(e) => notes.insert(0, format!("{e:#}")),
            }
        } else if !files.is_empty() {
            files.sort();
            files.dedup();
            self.start_load(files, ctx);
        }
        if !notes.is_empty() {
            let notes = notes.join("; ");
            // Keep the "Loading…" message when a load did start.
            if self.loading.is_some() {
                self.status = format!("{} ({notes})", self.status);
            } else {
                self.status = notes;
            }
        }
    }

    // ----- run number lookup ---------------------------------------------------

    fn open_run_dialog(&mut self) {
        self.run_lookup.open = true;
        self.run_lookup.focus = true;
    }

    /// Locate `run` on a background thread; the dialog shows the progress
    /// and, for a Timepix run, the raw / autoreduce choice.
    fn start_run_lookup(&mut self, run: u32, ctx: &egui::Context) {
        let (tx, rx) = std::sync::mpsc::channel();
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(run_lookup::resolve(run));
            ctx.request_repaint();
        });
        self.run_lookup.state = RunState::Busy { run, rx };
        self.run_lookup.input = run.to_string();
        self.run_lookup.open = true;
        self.status = format!("Locating run {run}…");
    }

    /// Collect a finished lookup: a non-Timepix run loads its raw data right
    /// away, a Timepix run waits in the dialog for the raw / autoreduce
    /// choice. Then start the `--run-number` lookup once the window is up.
    fn poll_run_lookup(&mut self, ctx: &egui::Context) {
        if let RunState::Busy { run, rx } = &self.run_lookup.state {
            let run = *run;
            let result = match rx.try_recv() {
                Ok(r) => Some(r),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => {
                    Some(Err(anyhow::anyhow!("the lookup thread stopped unexpectedly")))
                }
            };
            match result {
                None => {}
                Some(Ok(info)) if info.is_timepix() => {
                    self.status = format!(
                        "Run {run} ({}, {}): Timepix data — pick raw or autoreduce in the dialog.",
                        info.ipts,
                        info.detector_text()
                    );
                    self.run_lookup.state = RunState::Found(info);
                    self.run_lookup.open = true;
                }
                Some(Ok(info)) => self.load_run_stack(&info, RunChoice::Raw, ctx),
                Some(Err(e)) => {
                    let error = format!("{e:#}");
                    self.status = format!("Run {run}: {error}");
                    self.run_lookup.state = RunState::Failed { run, error };
                }
            }
        }
        if matches!(self.run_lookup.state, RunState::Idle)
            && !self.run_lookup.open
            && self.loading.is_none()
            && let Some(run) = self.pending_run.take()
        {
            self.start_run_lookup(run, ctx);
        }
    }

    /// Load the chosen data of a located run and close the dialog. The
    /// detector offset recorded with the run replaces the current one, and
    /// the run's detector (`BL10:Exp:Det`) decides the orientation.
    fn load_run_stack(&mut self, info: &RunInfo, choice: RunChoice, ctx: &egui::Context) {
        let run = info.run;
        // The run's experiment becomes the dialogs' starting point.
        if let Some(n) = info
            .ipts
            .strip_prefix("IPTS-")
            .and_then(|n| n.parse::<u32>().ok())
        {
            self.selected_ipts = Some(n);
        }
        let (files, which) = match choice {
            RunChoice::Raw => (info.raw_files(), "raw"),
            RunChoice::Autoreduce => (info.autoreduce_files(), "autoreduce"),
        };
        let mut head = format!("Run {run} ({}, {})", info.ipts, info.detector_text());
        if let Some(offset) = info.offset_us {
            self.offset_us = offset;
            head.push_str(&format!(", offset {offset:.3} µs"));
        }
        let files = match files {
            Ok(files) if !files.is_empty() => files,
            Ok(_) => {
                let looked_in = match choice {
                    RunChoice::Raw => &info.image_dir,
                    RunChoice::Autoreduce => &info.autoreduce_dir,
                };
                let error = format!(
                    "no loadable image found for this run (looked in {})",
                    looked_in.display()
                );
                self.status = format!("{head}: {error}");
                self.run_lookup.state = RunState::Failed { run, error };
                self.run_lookup.open = true;
                return;
            }
            Err(e) => {
                let error = format!("{e:#}");
                self.status = format!("{head}: {error}");
                self.run_lookup.state = RunState::Failed { run, error };
                self.run_lookup.open = true;
                return;
            }
        };
        let n = files.len();
        if self.start_load_from(files, info.detector.clone(), ctx) {
            self.status = format!("{head}: loading {which} data ({n} file(s))…");
        }
        self.run_lookup.state = RunState::Idle;
        self.run_lookup.open = false;
    }

    /// The "Locate a run" window: run number field, lookup progress or
    /// error, and for a Timepix run the raw / autoreduce buttons.
    fn run_dialog(&mut self, ctx: &egui::Context) {
        if !self.run_lookup.open {
            return;
        }
        let mut open = true;
        let mut lookup: Option<u32> = None;
        let mut parse_error: Option<String> = None;
        let mut choice: Option<RunChoice> = None;
        let busy = matches!(self.run_lookup.state, RunState::Busy { .. });
        egui::Window::new("Locate a run")
            .id(egui::Id::new("run_dialog"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .pivot(egui::Align2::CENTER_CENTER)
            .default_pos(ctx.content_rect().center())
            .show(ctx, |ui| {
                ui.set_min_width(440.0);
                ui.horizontal(|ui| {
                    ui.label("Run number:");
                    let edit = ui.add_enabled(
                        !busy,
                        egui::TextEdit::singleline(&mut self.run_lookup.input)
                            .desired_width(100.0)
                            .hint_text("e.g. 23640"),
                    );
                    if self.run_lookup.focus {
                        edit.request_focus();
                        self.run_lookup.focus = false;
                    }
                    let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    let go = ui
                        .add_enabled(!busy, egui::Button::new("🔍 Locate"))
                        .on_hover_text("Find the NeXus file of this run and the folder of its images")
                        .clicked();
                    if go || enter {
                        match self.run_lookup.input.trim().parse::<u32>() {
                            Ok(run) if run > 0 => lookup = Some(run),
                            _ => {
                                parse_error = Some(format!(
                                    "“{}” is not a run number (a positive integer)",
                                    self.run_lookup.input.trim()
                                ))
                            }
                        }
                    }
                });
                ui.small(format!(
                    "Looks for {}/IPTS-*/nexus/VENUS_<run>.nxs.h5 and reads the detector, \
                     the image folder and the detector offset from it.",
                    run_lookup::VENUS_ROOT
                ));
                ui.separator();
                match &self.run_lookup.state {
                    RunState::Idle => {
                        ui.label("Type a run number and press Enter.");
                    }
                    RunState::Busy { run, .. } => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(format!("Locating run {run}…"));
                        });
                    }
                    RunState::Failed { run, error } => {
                        // `run == 0` marks an unparsable run number field.
                        let text = if *run == 0 {
                            error.clone()
                        } else {
                            format!("Run {run}: {error}")
                        };
                        ui.colored_label(ui.visuals().error_fg_color, text);
                    }
                    RunState::Found(info) => {
                        ui.label(format!(
                            "Run {}: {} — {}",
                            info.run,
                            info.ipts,
                            info.detector_text()
                        ));
                        ui.small(info.nexus.display().to_string());
                        match info.offset_us {
                            Some(v) => ui.small(format!(
                                "Detector offset from the run: {v:.3} µs ({}/average_value)",
                                run_lookup::OFFSET_LOG
                            )),
                            None => ui.small(format!(
                                "No detector offset in the run ({} log missing): the current \
                                 offset is kept",
                                run_lookup::OFFSET_LOG
                            )),
                        };
                        ui.add_space(4.0);
                        ui.label("Timepix detector — which data do you want to load?");
                        let raw = info.raw.first();
                        ui.horizontal(|ui| {
                            let text = match raw {
                                Some(s) => format!("Raw ({})", s.size_text()),
                                None => "Raw".to_owned(),
                            };
                            let b = ui.add_enabled(info.raw_loadable(), egui::Button::new(text));
                            match raw {
                                Some(s) => {
                                    if b.on_hover_text(s.path.display().to_string())
                                        .on_disabled_hover_text(format!(
                                            "{}\nThe raw Timepix frames are .fits files, which this \
                                             program cannot load — use the autoreduce TIFFs",
                                            s.path.display()
                                        ))
                                        .clicked()
                                    {
                                        choice = Some(RunChoice::Raw);
                                    }
                                }
                                None => {
                                    ui.small(format!("not found: {}", info.image_dir.display()));
                                }
                            }
                        });
                        let auto = info.autoreduce.as_ref();
                        ui.horizontal(|ui| {
                            let text = match auto {
                                Some(s) => format!("Autoreduce ({})", s.size_text()),
                                None => "Autoreduce".to_owned(),
                            };
                            let b = ui.add_enabled(info.autoreduce_loadable(), egui::Button::new(text));
                            match auto {
                                Some(s) => {
                                    if b.on_hover_text(s.path.display().to_string())
                                        .on_disabled_hover_text(format!(
                                            "{}\nNo TIFF image in the autoreduce folder yet",
                                            s.path.display()
                                        ))
                                        .clicked()
                                    {
                                        choice = Some(RunChoice::Autoreduce);
                                    }
                                }
                                None => {
                                    ui.small(format!("not found: {}", info.autoreduce_dir.display()));
                                }
                            }
                        });
                    }
                }
            });
        if let Some(err) = parse_error {
            self.run_lookup.state = RunState::Failed { run: 0, error: err };
        }
        if let Some(run) = lookup {
            self.start_run_lookup(run, ctx);
        }
        if let Some(choice) = choice
            && let RunState::Found(info) = &self.run_lookup.state
        {
            let info = info.clone();
            self.load_run_stack(&info, choice, ctx);
        }
        if !open {
            // Closing the window forgets a located run / error, but a lookup
            // in flight keeps running so the status bar reports its outcome.
            if !busy {
                self.run_lookup.state = RunState::Idle;
            }
            self.run_lookup.open = false;
        }
    }

    // ----- config file (HDF5) -------------------------------------------------

    /// The current settings as a [`crate::config::AppConfig`] snapshot.
    fn current_config(&self) -> crate::config::AppConfig {
        crate::config::AppConfig {
            params: self.params,
            distance_m: self.distance_m,
            offset_us: self.offset_us,
            colormap: self.colormap,
            log_y: self.log_y,
            contrast_auto: self.contrast_auto,
            vmin: self.vmin,
            vmax: self.vmax,
            region: self
                .region
                .map(|r| [r.left, r.right, r.top, r.bottom]),
            input_folder: self.input_dir.clone(),
            mask_rects: self.mask_spec.rects.clone(),
            mask_range: self.mask_spec.range,
            mask_file: self.mask_spec.file.as_ref().map(|f| f.path.clone()),
        }
    }

    /// Apply a loaded config: correction parameters, physical axis, display
    /// settings, and profile region (clamped to the loaded stack, if any).
    /// The recorded input folder is informational only — no data is loaded.
    fn apply_config(&mut self, cfg: crate::config::AppConfig) {
        self.params = cfg.params;
        self.distance_m = cfg.distance_m;
        self.offset_us = cfg.offset_us;
        self.colormap = cfg.colormap;
        self.log_y = cfg.log_y;
        self.contrast_auto = cfg.contrast_auto;
        if !cfg.contrast_auto {
            self.vmin = cfg.vmin;
            self.vmax = cfg.vmax;
        }
        if let Some([left, right, top, bottom]) = cfg.region {
            let mut region = Region { left, right, top, bottom };
            if let Some(stack) = &self.stack {
                region = region.clamped(stack.width, stack.height);
            }
            self.region = Some(region);
        }
        // The mask definition; a mask file is re-read with the loaded
        // stack's orientation (it needs a stack to line up with).
        self.mask_spec.rects = cfg.mask_rects.clone();
        self.mask_spec.range = cfg.mask_range;
        if let Some(range) = cfg.mask_range {
            self.mask_range_values = range;
        }
        self.mask_spec.file = None;
        if let Some(path) = &cfg.mask_file {
            match &self.stack {
                Some(stack) => match mask::load_file(path, stack.detector) {
                    Ok(pixels) => {
                        self.mask_spec.file = Some(MaskFile {
                            path: path.clone(),
                            pixels: Arc::new(pixels),
                        });
                    }
                    Err(e) => self.mask_error = Some(format!("mask file of the config: {e:#}")),
                },
                None => {
                    self.mask_error = Some(format!(
                        "the config names a mask file ({}): load the data first, then load the config again",
                        path.display()
                    ));
                }
            }
        }
        self.mask_dirty = true;
        self.tex_dirty = true;
        self.profiles_dirty = true;
    }

    fn save_config_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Save the current settings as an HDF5 config file")
            .add_filter("HDF5 config", &["h5", "hdf5"])
            .set_file_name("dehydration_hydration_config.h5");
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.save_file() else { return };
        match crate::config::save(&path, &self.current_config()) {
            Ok(()) => self.status = format!("Settings saved to {}", path.display()),
            Err(e) => self.status = format!("Saving the config failed: {e:#}"),
        }
    }

    fn load_config_dialog(&mut self) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Load settings from an HDF5 config file")
            .add_filter("HDF5 config", &["h5", "hdf5"]);
        if let Some(dir) = &self.input_dir {
            dialog = dialog.set_directory(dir);
        }
        let Some(path) = dialog.pick_file() else { return };
        match crate::config::load(&path) {
            Ok(cfg) => {
                self.apply_config(cfg);
                self.status = format!("Settings loaded from {}", path.display());
            }
            Err(e) => self.status = format!("Loading the config failed: {e:#}"),
        }
    }

    // ----- UI: toolbar, parameters, status -----------------------------------

    fn toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            // Far right: the mbirtorch/About button; everything else flows in
            // from the left inside the nested layout.
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button("ℹ mbirtorch")
                    .on_hover_text("Algorithm provenance — the mbirtorch hsnt branch this beta runs")
                    .clicked()
                {
                    self.show_about = true;
                }
                let has_plots = self
                    .result
                    .as_ref()
                    .is_some_and(|r| r.artifacts.iter().any(|a| a.name.ends_with(".png")));
                if ui
                    .add_enabled(has_plots, egui::Button::new("📈 Diagnostics"))
                    .on_hover_text(
                        "The component spectra and maps plotted by mbirtorch for the current \
                         result, and its data checks",
                    )
                    .on_disabled_hover_text("Available once a correction has run")
                    .clicked()
                {
                    self.show_diagnostics = !self.show_diagnostics;
                }
                let has_log = !self.run_log.is_empty();
                if ui
                    .add_enabled(has_log, egui::Button::new("📜 Log"))
                    .on_hover_text(
                        "Everything the mbirtorch command line printed during the last \
                         correction",
                    )
                    .on_disabled_hover_text("No correction has run yet")
                    .clicked()
                {
                    self.show_log = !self.show_log;
                }
                ui.separator();
                ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                    self.toolbar_left(ui);
                });
            });
        });
    }

    /// The left-flowing part of the toolbar (everything except the
    /// right-aligned mbirtorch button).
    fn toolbar_left(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            let busy = self.loading.is_some();
            let ctx = ui.ctx().clone();
            self.ipts_combo(ui);
            if ui
                .add_enabled(!busy, egui::Button::new("📁 Open Folder…"))
                .clicked()
            {
                self.open_folder_dialog(&ctx);
            }
            if ui
                .add_enabled(!busy, egui::Button::new("📂 Open Files…"))
                .clicked()
            {
                self.open_files_dialog(&ctx);
            }
            ui.add_enabled_ui(!busy && !self.recent.is_empty(), |ui| {
                ui.menu_button("🕒 Recent", |ui| {
                    self.recent_menu(ui, &ctx);
                });
            });
            if ui
                .add_enabled(!busy, egui::Button::new("🔍 Run number…"))
                .on_hover_text(
                    "Locate a run's images from its run number: finds its NeXus file, \
                     then the raw data — and, for a Timepix run, lets you pick the raw \
                     or the autoreduce version (also sets the detector and its offset)",
                )
                .clicked()
            {
                self.open_run_dialog();
            }
            ui.add_enabled_ui(!busy, |ui| self.detector_combo(ui, &ctx));
            ui.menu_button("⚙ Config", |ui| {
                if ui
                    .button("💾 Save settings to HDF5…")
                    .on_hover_text(
                        "Write the correction parameters, physical axis, display \
                         settings, and profile region to a .h5 config file",
                    )
                    .clicked()
                {
                    self.save_config_dialog();
                }
                if ui
                    .button("📂 Load settings from HDF5…")
                    .on_hover_text("Apply the settings from a previously saved .h5 config file")
                    .clicked()
                {
                    self.load_config_dialog();
                }
            });

            ui.separator();

            let has_result = self.result.is_some();
            let mut changed = false;
            changed |= ui
                .selectable_value(&mut self.view, View::Raw, "Raw data")
                .changed();
            ui.add_enabled_ui(has_result, |ui| {
                changed |= ui
                    .selectable_value(&mut self.view, View::Result, "Corrected vs raw")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.view, View::Profiles, "Profiles")
                    .changed();
            });

            if self.view == View::Result {
                ui.separator();
                ui.label("vs:");
                changed |= ui
                    .selectable_value(&mut self.right_pane, RightPane::Raw, "Raw")
                    .changed();
                changed |= ui
                    .selectable_value(&mut self.right_pane, RightPane::Difference, "Difference")
                    .on_hover_text(
                        "corrected − raw on a symmetric color range: structure here is \
                         what the correction removed (or invented)",
                    )
                    .changed();
            }

            let n_frames = self.stack.as_ref().map(|s| s.n_frames()).unwrap_or(0);
            if matches!(self.view, View::Raw | View::Result) && n_frames > 1 {
                ui.separator();
                ui.label("Image:");
                let max = n_frames - 1;
                let source = self
                    .stack
                    .as_ref()
                    .and_then(|s| s.sources.get(self.frame_idx.min(max)))
                    .and_then(|p| p.file_name())
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                changed |= ui
                    .add(egui::Slider::new(&mut self.frame_idx, 0..=max))
                    .on_hover_text(source)
                    .changed();
            }
            if changed {
                self.frame_idx = self.frame_idx.min(n_frames.saturating_sub(1));
                self.tex_dirty = true;
            }

            ui.separator();

            ui.label("Contrast:");
            let range = self.data_min..=self.data_max;
            let speed = (self.data_max - self.data_min).max(1.0) / 200.0;
            let r1 = ui.add(
                egui::DragValue::new(&mut self.vmin)
                    .speed(speed)
                    .range(range.clone()),
            );
            let r2 = ui.add(egui::DragValue::new(&mut self.vmax).speed(speed).range(range));
            if r1.changed() || r2.changed() {
                self.contrast_auto = false;
                self.tex_dirty = true;
            }
            if ui
                .button("Auto")
                .on_hover_text("Track the displayed image's full range")
                .clicked()
            {
                self.contrast_auto = true;
                self.tex_dirty = true;
            }

            ui.separator();

            let mut cmap_changed = false;
            egui::ComboBox::from_id_salt("colormap")
                .selected_text(format!("Colormap: {}", self.colormap.label()))
                .show_ui(ui, |ui| {
                    for c in Colormap::ALL {
                        cmap_changed |= ui
                            .selectable_value(&mut self.colormap, c, c.label())
                            .changed();
                    }
                });
            if cmap_changed {
                self.tex_dirty = true;
            }

            ui.separator();
            ui.label("Zoom:");
            if ui.button("−").clicked() {
                self.scale = (self.scale / 1.25).max(0.02);
            }
            if ui.button("+").clicked() {
                self.scale = (self.scale * 1.25).min(64.0);
            }
            if ui.button("Fit").clicked() {
                self.fit_requested = true;
            }
            ui.label(format!("{:.0}%", self.scale * 100.0));

            ui.separator();
            crate::theme::toggle_button(ui);
            crate::zoom::toggle_button(ui);
        });
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(stack) = &self.stack {
            ui.heading("Data set");
            ui.add_space(2.0);
            if let Some(dir) = &self.input_dir {
                ui.label(
                    egui::RichText::new(dir.display().to_string())
                        .small()
                        .weak(),
                )
                .on_hover_text("Folder the images were loaded from");
            }
            let (n, h, w) = (stack.n_frames(), stack.height, stack.width);
            ui.label(format!("{n} images of {w}×{h} px"));
            ui.label(format!(
                "In memory: {} (f32); the exchange file for mbirtorch is as large",
                fmt_bytes(n as u64 * (h * w) as u64 * 4),
            ))
            .on_hover_text(format!(
                "The stack is handed to mbirtorch through a scratch HDF5 file in {} \
                 (override with {}), removed after the run",
                hsnt_cli::scratch_root().display(),
                hsnt_cli::SCRATCH_ENV_VAR
            ));
            if stack.nonfinite_fixed > 0 {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!(
                        "{} NaN/Inf pixel(s) replaced by 0 on load",
                        stack.nonfinite_fixed
                    ),
                );
            }
            if let (Some(first), Some(last)) = (stack.sources.first(), stack.sources.last()) {
                let name = |p: &std::path::Path| {
                    p.file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default()
                };
                ui.label(
                    egui::RichText::new(format!("{} … {}", name(first), name(last)))
                        .small()
                        .weak(),
                )
                .on_hover_text("First and last image file of the stack");
            }
            ui.add_space(6.0);
            ui.separator();
        }

        if let Some(stack) = self.stack.clone() {
            self.mask_section(ui, &stack);
            ui.add_space(6.0);
            ui.separator();
        }

        ui.heading("Correction parameters");
        ui.label(
            egui::RichText::new("mbirtorch.hsnt denoise — maximum-likelihood factorization")
                .small()
                .weak(),
        );
        ui.add_space(4.0);

        let running = self.corr_job.is_some();
        ui.add_enabled_ui(!running, |ui| {
            egui::ComboBox::from_label("Input type")
                .selected_text(self.params.input_type.label())
                .show_ui(ui, |ui| {
                    for t in InputType::ALL {
                        ui.selectable_value(&mut self.params.input_type, t, t.label());
                    }
                });
            ui.label(
                egui::RichText::new(
                    "transmission = normalized data (the usual case); attenuation = \
                     −log(transmission); auto infers it from the values",
                )
                .small()
                .weak(),
            );
            ui.add_space(6.0);

            // Number of materials (hsnt rank): estimated by the CLI, or given.
            let mut auto = self.params.rank == Rank::Auto;
            if let Rank::Fixed(n) = self.params.rank {
                self.fixed_rank = n;
            }
            ui.horizontal(|ui| {
                ui.label("Number of materials");
                if ui
                    .selectable_label(auto, "auto")
                    .on_hover_text(
                        "Estimate the number of materials from the data by likelihood-ratio \
                         tests (at full resolution and on pooled pixels)",
                    )
                    .clicked()
                {
                    auto = true;
                }
                if ui
                    .selectable_label(!auto, "fixed")
                    .on_hover_text("Give the number of materials yourself")
                    .clicked()
                {
                    auto = false;
                }
                ui.add_enabled(
                    !auto,
                    egui::DragValue::new(&mut self.fixed_rank).range(1..=50).speed(0.1),
                );
            });
            ui.label(
                egui::RichText::new(
                    "Number of distinct materials in the field of view (the hsnt rank); \
                     the fitted components span the materials' spectra",
                )
                .small()
                .weak(),
            );
            if auto {
                ui.add(egui::Slider::new(&mut self.params.max_rank, 1..=12).text("Max materials"))
                    .on_hover_text("Largest number of materials the estimate considers");
            }
            self.params.rank = if auto {
                Rank::Auto
            } else {
                Rank::Fixed(self.fixed_rank.max(1))
            };
            ui.add_space(6.0);

            egui::ComboBox::from_label("Spectra")
                .selected_text(self.params.spectra.label())
                .show_ui(ui, |ui| {
                    for s in Spectra::ALL {
                        ui.selectable_value(&mut self.params.spectra, s, s.label())
                            .on_hover_text(spectra_help(s));
                    }
                })
                .response
                .on_hover_text(spectra_help(self.params.spectra));

            // Dose: optional.
            let mut has_dose = self.params.dose.is_some();
            if let Some(d) = self.params.dose {
                self.dose_value = d;
            }
            ui.horizontal(|ui| {
                ui.checkbox(&mut has_dose, "Dose known").on_hover_text(
                    "Open-beam counts per pixel and bin. When given, the fit reports a \
                     reduced chi-square against the Poisson noise; required by the \
                     'support' spectra",
                );
                ui.add_enabled(
                    has_dose,
                    egui::DragValue::new(&mut self.dose_value)
                        .range(1e-6..=1e12)
                        .speed(1.0)
                        .suffix(" counts"),
                );
            });
            self.params.dose = has_dose.then_some(self.dose_value);
            if self.params.spectra == Spectra::Support && !has_dose {
                ui.colored_label(ui.visuals().warn_fg_color, "The 'support' spectra need the dose.");
            }
            ui.add_space(6.0);

            egui::ComboBox::from_label("Device")
                .selected_text(self.params.device.label())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.params.device, Device::Auto, "auto");
                    ui.selectable_value(&mut self.params.device, Device::Cpu, "cpu");
                    for n in 0..4u8 {
                        ui.selectable_value(
                            &mut self.params.device,
                            Device::Cuda(n),
                            Device::Cuda(n).label(),
                        );
                    }
                })
                .response
                .on_hover_text("auto = the first CUDA GPU when there is one, else the CPU");

            ui.collapsing("Advanced (solver)", |ui| {
                egui::ComboBox::from_label("Mode")
                    .selected_text(self.params.mode.label())
                    .show_ui(ui, |ui| {
                        for m in SolveMode::ALL {
                            ui.selectable_value(&mut self.params.mode, m, m.label());
                        }
                    })
                    .response
                    .on_hover_text(
                        "full: the whole solve on the device; stream: by chunks of pixels \
                         (less memory, polish passes); auto: full when the device has the \
                         memory",
                    );
                ui.horizontal(|ui| {
                    ui.label("Max steps");
                    ui.add(egui::DragValue::new(&mut self.params.max_steps).range(1..=100_000))
                        .on_hover_text("Solver step cap of a full solve (default 1000)");
                });
                ui.horizontal(|ui| {
                    ui.label("Rel. tol.");
                    ui.add(
                        egui::DragValue::new(&mut self.params.rel_tol)
                            .range(0.0..=1.0)
                            .speed(0.0)
                            .custom_formatter(|v, _| format!("{v:.0e}")),
                    )
                    .on_hover_text(
                        "Relative loss change per step; the solve stops after five steps in \
                         a row below it (default 1e-8; type a value such as 1e-6)",
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Max passes");
                    ui.add(egui::DragValue::new(&mut self.params.max_passes).range(0..=200))
                        .on_hover_text(
                            "Stream mode: polish passes over the data after the fit on a \
                             pixel subsample (default 5; more passes trade time for SNR)",
                        );
                });
                egui::ComboBox::from_label("Compile")
                    .selected_text(self.params.compile.label())
                    .show_ui(ui, |ui| {
                        for c in Compile::ALL {
                            ui.selectable_value(&mut self.params.compile, c, c.label());
                        }
                    })
                    .response
                    .on_hover_text(
                        "torch.compile of the solver kernels: auto compiles on CUDA for \
                         large data only",
                    );
            });
        });

        ui.add_space(8.0);
        let ctx = ui.ctx().clone();
        if let Some(job) = &self.corr_job {
            ui.add(
                egui::ProgressBar::new(job.fraction)
                    .animate(true)
                    .text(job.stage.clone()),
            );
            if let Some(last) = self.run_log.last() {
                ui.label(
                    egui::RichText::new(hsnt_cli::strip_log_prefix(last))
                        .small()
                        .weak(),
                )
                .on_hover_text(
                    "Last line of the mbirtorch log (📜 Log in the toolbar shows all of it)",
                );
            }
            ui.horizontal(|ui| {
                if ui.button("✖ Cancel").clicked() {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                if ui.button("📜 Log").clicked() {
                    self.show_log = !self.show_log;
                }
            });
        } else {
            let can_run = self.stack.is_some() && self.loading.is_none();
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_run, egui::Button::new("▶ Perform correction"))
                    .on_hover_text(
                        "Runs mbirtorch-hsnt denoise on the whole stack (dehydration + \
                         rehydration)",
                    )
                    .clicked()
                {
                    self.start_correction_job(&ctx, 1);
                }
                if ui
                    .add_enabled(can_run, egui::Button::new("⚡ Preview"))
                    .on_hover_text(format!(
                        "Fast preview on {PREVIEW_BIN}×{PREVIEW_BIN}-binned pixels (~{}× \
                         fewer pixels) for parameter tuning; previews cannot be exported",
                        PREVIEW_BIN * PREVIEW_BIN
                    ))
                    .clicked()
                {
                    self.start_correction_job(&ctx, PREVIEW_BIN);
                }
            });
        }

        if let Some(result) = &self.result {
            ui.add_space(10.0);
            ui.separator();
            if result.is_preview() {
                ui.heading(format!("Result — PREVIEW ({0}×{0} binned)", result.bin));
            } else {
                ui.heading("Result");
            }
            let r = &result.report;
            match r.rank {
                Some(rank) => {
                    ui.label(format!("Number of materials: {rank}")).on_hover_text(if r.rank_note.is_empty() {
                        "number of materials (the hsnt rank)"
                    } else {
                        r.rank_note.as_str()
                    });
                }
                None => {
                    ui.label("Number of materials: not reported");
                }
            }
            if !r.rank_note.is_empty() {
                ui.label(egui::RichText::new(&r.rank_note).small().weak());
            }
            let mut solve = format!(
                "{} solve",
                if r.mode.is_empty() { "?" } else { r.mode.as_str() }
            );
            if let Some(n) = r.steps {
                solve.push_str(&format!(", {n} steps"));
            }
            if let Some(t) = r.solve_seconds {
                solve.push_str(&format!(", {t:.1} s"));
            }
            if let Some(l) = r.loss {
                solve.push_str(&format!(", loss {l:.6}"));
            }
            ui.label(solve);
            match r.reduced_chi2 {
                Some(chi2) => {
                    let text = format!("Reduced chi-square {chi2:.3}");
                    let verdict = r.chi2_verdict().unwrap_or("");
                    if (0.5..=2.0).contains(&chi2) {
                        ui.label(text).on_hover_text(verdict);
                    } else {
                        ui.colored_label(ui.visuals().warn_fg_color, text)
                            .on_hover_text(verdict);
                    }
                    ui.label(egui::RichText::new(verdict).small().weak());
                }
                None => {
                    if let Some(res) = r.relative_residual {
                        ui.label(format!("Relative residual in transmission {res:.4}"))
                            .on_hover_text(
                                "No dose given, so no chi-square against the Poisson noise",
                            );
                    }
                }
            }
            if !r.warnings.is_empty() {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    format!("{} warning(s) from mbirtorch — see 📜 Log", r.warnings.len()),
                )
                .on_hover_text(r.warnings.join("\n"));
            }
            ui.label(format!(
                "{} · spectra {} · device {}",
                result.params.input_type.label(),
                result.params.spectra.label(),
                result.params.device.label()
            ));
            ui.label(format!("Computed in {:.1} s", result.elapsed_seconds));
            if let Some(m) = &result.mask {
                ui.label(format!(
                    "Mask: {} of {} pixels ({:.1}%) solved; the others keep their raw values",
                    result.n_pixels,
                    m.len(),
                    100.0 * result.n_pixels as f64 / m.len().max(1) as f64
                ))
                .on_hover_text(&result.mask_description);
            }
            if self.params != result.params {
                ui.colored_label(
                    ui.visuals().warn_fg_color,
                    "Parameters changed since this result — run again to apply.",
                );
            }

            ui.add_space(8.0);
            if let Some(job) = &self.export_job {
                let frac = if job.total > 0 {
                    job.done as f32 / job.total as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("Exporting {} / {}", job.done, job.total)),
                );
            } else {
                let exportable = !result.is_preview();
                if ui
                    .add_enabled(exportable, egui::Button::new("💾 Export corrected images…"))
                    .on_hover_text(
                        "Choose an output folder; the corrected stack is written as \
                         32-bit float TIFFs (plus correction_config.json recording the \
                         parameters) into a new subfolder named after the input folder",
                    )
                    .on_disabled_hover_text(
                        "Preview results are binned — run the full correction to export",
                    )
                    .clicked()
                {
                    self.export_dialog(&ctx);
                }
            }
            if let Some(folder) = &self.last_export {
                ui.add_space(4.0);
                ui.label("Exported to:");
                ui.label(
                    egui::RichText::new(folder.display().to_string())
                        .small()
                        .weak(),
                );
            }
        }
    }

    fn status_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(&self.status);
            // The warnings of the last run: one click opens the log, with
            // them listed on top.
            let n_warnings = self.result.as_ref().map_or(0, |r| r.report.warnings.len());
            if n_warnings > 0
                && ui
                    .button(
                        egui::RichText::new(format!("⚠ {n_warnings} warning(s) — show log"))
                            .color(egui::Color32::from_rgb(230, 160, 30)),
                    )
                    .on_hover_text("Open the mbirtorch hsnt log, the warnings listed first")
                    .clicked()
            {
                self.show_log = true;
            }
            if let Some(stack) = &self.stack {
                ui.separator();
                ui.label(format!(
                    "{} images, {}×{} px",
                    stack.n_frames(),
                    stack.width,
                    stack.height
                ));
            }
            if let Some((x, y, v)) = self.cursor {
                ui.separator();
                ui.label(format!("({x}, {y}) = {v:.4}"));
            }
        });
    }

    // ----- UI: image viewers -------------------------------------------------

    fn viewer(&mut self, ui: &mut egui::Ui) {
        if let Some(job) = &self.loading {
            let frac = if job.total > 0 {
                job.done as f32 / job.total as f32
            } else {
                0.0
            };
            ui.centered_and_justified(|ui| {
                ui.add_sized(
                    [320.0, 24.0],
                    egui::ProgressBar::new(frac)
                        .show_percentage()
                        .text(format!("⏳ Loading {} / {} files", job.done, job.total)),
                );
            });
            return;
        }
        if self.stack.is_none() {
            ui.centered_and_justified(|ui| {
                ui.label("No images loaded — use 'Open Folder…' to select the data to correct.");
            });
            return;
        }
        // The viewers adapt to the window, but their fixed parts (title rows,
        // control rows, minimum viewport/plot sizes) can outgrow a short
        // window (small displays, the large-text mode). The panel height is
        // measured before entering the scroll area — inside it the available
        // height is unbounded — and the minimum sizes in the viewers are what
        // make the scroll bar appear.
        let panel_h = ui.available_height();
        egui::ScrollArea::vertical()
            .id_salt("central_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| match self.view {
                View::Raw | View::Result => self.dual_viewer(ui, panel_h),
                View::Profiles => self.profiles_view(ui, panel_h),
            });
    }

    /// Applies a Ctrl+wheel (or pinch) zoom reported over an image.
    /// `content` is the point under the cursor in *unscaled* content
    /// coordinates of the scroll area (image pixels, plus a whole image
    /// width for the right pane of the dual viewer), `offset` the scroll
    /// area's current offset. The next frame scrolls so that point stays
    /// under the cursor.
    fn apply_wheel_zoom(
        &mut self,
        ctx: &egui::Context,
        content: egui::Vec2,
        factor: f32,
        offset: egui::Vec2,
    ) {
        let old = self.scale;
        let new = (old * factor).clamp(0.02, 64.0);
        if new == old {
            return;
        }
        // The content shifts by its distance to the origin times the scale
        // change.
        self.viewer_scroll = Some((offset + content * (new - old)).max(egui::Vec2::ZERO));
        self.scale = new;
        ctx.request_repaint();
    }

    /// Two images side by side (raw + integrated, or corrected + raw /
    /// difference) with a shared zoom and the colorbar of the
    /// contrast-controlled left pane.
    fn dual_viewer(&mut self, ui: &mut egui::Ui, panel_h: f32) {
        // Dimensions come from the displayed image, not the stack: preview
        // results are spatially binned.
        let Some((h, w)) = self.pane_cache.0.as_ref().map(|img| img.dim()) else {
            return;
        };
        let (title_left, title_right) = self.pane_titles();

        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(title_left).strong());
            ui.label(" | ");
            ui.label(egui::RichText::new(title_right).strong());
        });

        // Height left for the image panes: the measured panel height minus
        // the title row, never below a usable viewport.
        let view_h =
            (panel_h - ui.min_rect().height() - ui.spacing().item_spacing.y).max(160.0);

        // Mask rectangles are drawn (and shown) on the integrated image, the
        // right pane of the raw view. The drag is tracked in local state and
        // committed after the panes are laid out.
        let draw_mask = self.mask_draw && self.view == View::Raw;
        let rects: Vec<MaskRect> = if self.view == View::Raw && (self.show_mask || draw_mask) {
            self.mask_spec.rects.clone()
        } else {
            Vec::new()
        };
        let rect_mode = self.mask_rect_mode;
        let mut drag = self.mask_drag;
        let mut committed: Option<MaskRect> = None;

        ui.horizontal_top(|ui| {
            let avail = ui.available_size();
            let view_w = (avail.x - COLORBAR_WIDTH - ui.spacing().item_spacing.x).max(50.0);
            if self.fit_requested && w > 0 && h > 0 {
                let gap = ui.spacing().item_spacing.x;
                let s = ((view_w - gap) / (2.0 * w as f32)).min(view_h / h as f32);
                self.scale = s.clamp(0.02, 64.0);
                self.fit_requested = false;
            }

            let panes = [
                (self.tex_left.clone(), self.pane_cache.0.clone()),
                (self.tex_right.clone(), self.pane_cache.1.clone()),
            ];
            ui.allocate_ui(egui::vec2(view_w, view_h), |ui| {
                ui.set_min_size(egui::vec2(view_w, view_h));
                let mut scroll = egui::ScrollArea::both().auto_shrink([false, false]);
                if let Some(offset) = self.viewer_scroll.take() {
                    scroll = scroll.scroll_offset(offset);
                }
                // A Ctrl+wheel zoom over a pane: (content point under the
                // cursor, zoom factor). Applied after the scroll area reports
                // its current offset, so next frame's scale and offset match.
                let mut wheel_zoom: Option<(egui::Vec2, f32)> = None;
                let out = scroll.show(ui, |ui| {
                    ui.horizontal_top(|ui| {
                        let scale = self.scale;
                        let size = egui::vec2(w as f32 * scale, h as f32 * scale);
                        let mut cursor = None;
                        for (k, (tex, img)) in panes.iter().enumerate() {
                            let Some(tex) = tex else { continue };
                            let mask_pane = k == 1;
                            let sense = if draw_mask && mask_pane {
                                Sense::click_and_drag()
                            } else {
                                Sense::hover()
                            };
                            let (rect, response) = ui.allocate_exact_size(size, sense);
                            let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
                            let painter = ui.painter_at(rect);
                            painter.image(tex.id(), rect, full_uv, Color32::WHITE);
                            let to_img = |p: Pos2| -> (f32, f32) {
                                ((p.x - rect.left()) / scale, (p.y - rect.top()) / scale)
                            };
                            let to_screen = |ix: f32, iy: f32| -> Pos2 {
                                Pos2::new(rect.left() + ix * scale, rect.top() + iy * scale)
                            };

                            if mask_pane {
                                for r in &rects {
                                    let color = match r.mode {
                                        RectMode::Include => MASK_INCLUDE_COLOR,
                                        RectMode::Exclude => MASK_EXCLUDE_COLOR,
                                    };
                                    painter.rect_stroke(
                                        Rect::from_min_max(
                                            to_screen(r.left as f32, r.top as f32),
                                            to_screen(r.right as f32, r.bottom as f32),
                                        ),
                                        egui::CornerRadius::ZERO,
                                        Stroke::new(1.5, color),
                                        egui::StrokeKind::Middle,
                                    );
                                }
                                if draw_mask {
                                    if response.hovered() {
                                        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                                    }
                                    if response.drag_started()
                                        && let Some(sp) = ui
                                            .ctx()
                                            .input(|i| i.pointer.press_origin())
                                            .or_else(|| response.interact_pointer_pos())
                                    {
                                        let p = to_img(sp);
                                        drag = Some([p.0, p.1, p.0, p.1]);
                                    }
                                    if (response.dragged() || response.drag_stopped())
                                        && let (Some(d), Some(sp)) =
                                            (drag.as_mut(), response.interact_pointer_pos())
                                    {
                                        let p = to_img(sp);
                                        d[2] = p.0;
                                        d[3] = p.1;
                                    }
                                    if let Some(d) = drag {
                                        painter.rect_stroke(
                                            Rect::from_two_pos(to_screen(d[0], d[1]), to_screen(d[2], d[3])),
                                            egui::CornerRadius::ZERO,
                                            Stroke::new(2.0, Color32::YELLOW),
                                            egui::StrokeKind::Middle,
                                        );
                                    }
                                    if response.drag_stopped()
                                        && let Some(d) = drag.take()
                                    {
                                        committed =
                                            MaskRect::from_corners((d[0], d[1]), (d[2], d[3]), w, h, rect_mode);
                                    }
                                }
                            }

                            if let (Some(p), Some(img)) = (response.hover_pos(), img) {
                                let (fx, fy) = to_img(p);
                                let ix = fx.floor() as i64;
                                let iy = fy.floor() as i64;
                                if ix >= 0 && iy >= 0 && (ix as usize) < w && (iy as usize) < h {
                                    cursor = Some((
                                        ix as usize,
                                        iy as usize,
                                        img[(iy as usize, ix as usize)],
                                    ));
                                }
                                // egui turns Ctrl (Cmd on macOS) + wheel into a
                                // zoom factor instead of a scroll; a pinch
                                // arrives the same way.
                                let factor = ui.input(|i| i.zoom_delta());
                                if factor != 1.0 {
                                    // The right pane sits one image width (plus
                                    // the gap, which does not scale) further in.
                                    let content = egui::vec2(fx + k as f32 * w as f32, fy);
                                    wheel_zoom = Some((content, factor));
                                }
                            }
                        }
                        self.cursor = cursor;
                    });
                });
                if let Some((content, factor)) = wheel_zoom {
                    self.apply_wheel_zoom(ui.ctx(), content, factor, out.state.offset);
                }
            });

            self.colorbar(ui, view_h);
        });

        self.mask_drag = drag;
        if let Some(r) = committed {
            self.mask_spec.rects.push(r);
            self.mask_dirty = true;
        }
    }

    /// Profiles view: the integrated corrected image with a draggable region
    /// on the left, the region's mean-intensity profiles on the right.
    fn profiles_view(&mut self, ui: &mut egui::Ui, panel_h: f32) {
        let Some((h, w)) = self.result.as_ref().map(|r| r.dims()) else {
            return;
        };
        let (title_left, _) = self.pane_titles();

        // Height the two columns may use: the measured panel height (inside
        // the scroll area the available height is unbounded), never below a
        // usable minimum.
        let body_h = (panel_h - ui.min_rect().height()).max(240.0);

        ui.horizontal_top(|ui| {
            let avail = ui.available_size();
            let img_w = (avail.x * 0.42).max(120.0);

            // ---- left: image + region ----
            ui.vertical(|ui| {
                ui.set_max_width(img_w);
                ui.label(egui::RichText::new(title_left).strong());
                ui.label(
                    "Drag to draw the region, drag inside it to move it, drag a handle to \
                     resize it. Click a pixel to plot its spectrum.",
                );
                if self.fit_requested && w > 0 && h > 0 {
                    let s = (img_w / w as f32).min((body_h - 60.0).max(50.0) / h as f32);
                    self.scale = s.clamp(0.02, 64.0);
                    self.fit_requested = false;
                }
                let mut scroll = egui::ScrollArea::both()
                    .id_salt("profile_img")
                    .auto_shrink([false, false])
                    .max_height((body_h - 40.0).max(120.0));
                if let Some(offset) = self.viewer_scroll.take() {
                    scroll = scroll.scroll_offset(offset);
                }
                let mut wheel_zoom: Option<(egui::Vec2, f32)> = None;
                let out = scroll.show(ui, |ui| {
                    wheel_zoom = self.profile_image(ui, w, h);
                });
                if let Some((content, factor)) = wheel_zoom {
                    self.apply_wheel_zoom(ui.ctx(), content, factor, out.state.offset);
                }
            });

            ui.separator();

            // ---- right: region fields + plot ----
            ui.vertical(|ui| {
                if let Some(region) = &mut self.region {
                    let mut r = *region;
                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Left:");
                        changed |= drag_usize(ui, &mut r.left, w);
                        ui.label("Right:");
                        changed |= drag_usize(ui, &mut r.right, w);
                        ui.label("Top:");
                        changed |= drag_usize(ui, &mut r.top, h);
                        ui.label("Bottom:");
                        changed |= drag_usize(ui, &mut r.bottom, h);
                    });
                    if changed {
                        *region = r.clamped(w, h);
                        self.profiles_dirty = true;
                    }
                }
                ui.horizontal(|ui| {
                    if let Some(region) = self.region {
                        ui.label(format!(
                            "Region (rows {}:{}, columns {}:{})",
                            region.top, region.bottom, region.left, region.right
                        ));
                        ui.separator();
                    }
                    ui.label("Y-axis:");
                    ui.selectable_value(&mut self.log_y, false, "Linear");
                    ui.selectable_value(&mut self.log_y, true, "Log")
                        .on_hover_text("log₁₀ scale; non-positive values are hidden");
                    ui.separator();
                    if ui
                        .add_enabled(self.profiles.is_some(), egui::Button::new("📄 Save CSV…"))
                        .on_hover_text(
                            "Write the plotted profiles (with TOF/wavelength columns when \
                             a Spectra.txt was found) to a CSV file",
                        )
                        .clicked()
                    {
                        self.save_profiles_csv();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("X-axis:");
                    ui.selectable_value(&mut self.x_axis, XAxis::Index, "Image index");
                    ui.add_enabled_ui(self.spectra_tof_us.is_some(), |ui| {
                        ui.selectable_value(&mut self.x_axis, XAxis::TofUs, "TOF (µs)")
                            .on_disabled_hover_text("No *_Spectra.txt found next to the images");
                        ui.selectable_value(
                            &mut self.x_axis,
                            XAxis::LambdaAngstrom,
                            "Wavelength (Å)",
                        )
                        .on_disabled_hover_text("No *_Spectra.txt found next to the images");
                    });
                    if self.x_axis == XAxis::LambdaAngstrom {
                        ui.label("L (m):");
                        ui.add(
                            egui::DragValue::new(&mut self.distance_m)
                                .speed(0.01)
                                .range(0.1..=100.0),
                        )
                        .on_hover_text("Source–detector distance for λ = h·t/(m_n·L)");
                    }
                    if self.x_axis != XAxis::Index {
                        ui.label("Detector offset:");
                        ui.add(
                            egui::DragValue::new(&mut self.offset_us)
                                .speed(0.5)
                                .suffix(" µs"),
                        )
                        .on_hover_text(
                            "Constant added to the TOF values of the spectra file \
                             (also shifts the wavelength axis)",
                        );
                    }
                    if let Some((py, px)) = self.pixel_marker {
                        ui.separator();
                        ui.label(format!("Pixel ({px}, {py})"));
                        if ui.small_button("✖").on_hover_text("Remove the pixel marker").clicked()
                        {
                            self.pixel_marker = None;
                            self.pixel_profiles = None;
                        }
                    }
                });
                if let Some((uncorrected, corrected)) = &self.profiles {
                    // The plot gets the height left below the control rows:
                    // inside the scroll area the available height is
                    // unbounded, so it must be sized explicitly — and its
                    // minimum is what makes the scroll bar appear.
                    let plot_h = (body_h
                        - ui.min_rect().height()
                        - ui.spacing().item_spacing.y)
                        .max(140.0);
                    // In log mode the plotted values are log10(y); the axis
                    // ticks and the cursor read-out convert back.
                    let log_y = self.log_y;
                    let xs = self.x_values(uncorrected.len().max(corrected.len()));
                    let series = |vals: &[f64]| -> PlotPoints {
                        vals.iter()
                            .enumerate()
                            .filter(|&(_, &v)| !log_y || v > 0.0)
                            .map(|(i, &v)| {
                                [
                                    xs.get(i).copied().unwrap_or(i as f64),
                                    if log_y { v.log10() } else { v },
                                ]
                            })
                            .collect()
                    };
                    let uncorr = series(uncorrected);
                    let corr = series(corrected);
                    let pixel_series = self
                        .pixel_profiles
                        .as_ref()
                        .map(|(u, c)| (series(u), series(c)));
                    let x_axis = self.x_axis;
                    let mut plot = Plot::new(("profiles_plot", log_y, x_axis.label()))
                        .height(plot_h)
                        .legend(Legend::default())
                        .x_axis_label(x_axis.label())
                        .y_axis_label(if log_y {
                            "Average intensity (log)"
                        } else {
                            "Average intensity"
                        })
                        .coordinates_formatter(
                            Corner::LeftBottom,
                            CoordinatesFormatter::new(move |p, _| {
                                let y = if log_y { 10f64.powf(p.y) } else { p.y };
                                let x = match x_axis {
                                    XAxis::Index => format!("image {:.0}", p.x),
                                    XAxis::TofUs => format!("{:.2} µs", p.x),
                                    XAxis::LambdaAngstrom => format!("{:.4} Å", p.x),
                                };
                                format!("{x}  —  intensity {y:.5}")
                            }),
                        );
                    if log_y {
                        plot = plot
                            .y_axis_formatter(|mark, _| fmt_axis(10f64.powf(mark.value)));
                    }
                    // Explicit colors (never egui_plot's automatic ones, which
                    // are handed out in insertion order while the legend is
                    // sorted by name) and the marker glyph in each legend
                    // entry, so the legend cannot be read the wrong way round.
                    plot.show(ui, |plot_ui| {
                        plot_ui.points(
                            Points::new("Uncorrected profile  ✕", uncorr)
                                .shape(MarkerShape::Cross)
                                .color(UNCORRECTED_COLOR)
                                .radius(3.0),
                        );
                        plot_ui.points(
                            Points::new("Corrected profile  ●", corr)
                                .shape(MarkerShape::Circle)
                                .color(CORRECTED_COLOR)
                                .radius(2.5),
                        );
                        if let Some((pu, pc)) = pixel_series {
                            plot_ui.points(
                                Points::new("Pixel uncorrected  ✱", pu)
                                    .shape(MarkerShape::Asterisk)
                                    .color(PIXEL_UNCORRECTED_COLOR)
                                    .radius(2.0),
                            );
                            plot_ui.points(
                                Points::new("Pixel corrected  ◆", pc)
                                    .shape(MarkerShape::Diamond)
                                    .color(PIXEL_CORRECTED_COLOR)
                                    .radius(2.0),
                            );
                        }
                    });
                }
            });
        });
    }

    /// The integrated corrected image with the profile region rectangle;
    /// dragging draws a new region.
    /// Returns a pending Ctrl+wheel zoom (image point under the cursor,
    /// zoom factor) for the caller to apply once the scroll offset is known.
    fn profile_image(
        &mut self,
        ui: &mut egui::Ui,
        w: usize,
        h: usize,
    ) -> Option<(egui::Vec2, f32)> {
        let tex = self.tex_left.clone()?;
        let size = egui::vec2(w as f32 * self.scale, h as f32 * self.scale);
        let (rect, response) = ui.allocate_exact_size(size, Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex.id(), rect, full_uv, Color32::WHITE);

        let scale = self.scale;
        let to_img =
            |p: Pos2| -> (f32, f32) { ((p.x - rect.left()) / scale, (p.y - rect.top()) / scale) };

        // Cursor read-out on the integrated corrected image.
        self.cursor = None;
        if let Some(p) = response.hover_pos() {
            let (ix, iy) = to_img(p);
            let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
            if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                if let Some(result) = &self.result {
                    self.cursor =
                        Some((xi as usize, yi as usize, result.integrated_mean[(yi as usize, xi as usize)]));
                }
            }
        }

        const HANDLE_HIT: f32 = 10.0;
        const HANDLE_SIZE: f32 = 9.0;
        let to_screen = |ix: f32, iy: f32| -> Pos2 {
            Pos2::new(rect.left() + ix * scale, rect.top() + iy * scale)
        };

        // Plain click (no drag): mark the pixel whose spectrum to plot.
        if response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                let (ix, iy) = to_img(p);
                let (xi, yi) = (ix.floor() as i64, iy.floor() as i64);
                if xi >= 0 && yi >= 0 && (xi as usize) < w && (yi as usize) < h {
                    let new = (yi as usize, xi as usize);
                    self.pixel_marker = if self.pixel_marker == Some(new) {
                        None // clicking the marked pixel clears it
                    } else {
                        Some(new)
                    };
                    self.profiles_dirty = true;
                }
            }
        }

        // A drag starts on a handle (resize), inside the region (move), or
        // anywhere else (draw a new region). Hit-test against the press
        // origin, not the current pointer position: by the time egui reports
        // drag_started the pointer has already moved past the drag threshold,
        // and a fast drag on a handle would miss it and draw a new region.
        if response.drag_started() {
            let press = ui
                .ctx()
                .input(|i| i.pointer.press_origin())
                .or_else(|| response.interact_pointer_pos());
            if let Some(sp) = press {
                let p = to_img(sp);
                let started = self.region.and_then(|r| {
                    let rectf = [
                        r.left as f32,
                        r.top as f32,
                        r.right as f32,
                        r.bottom as f32,
                    ];
                    if let Some(((hx, vy), _)) = r
                        .handles()
                        .into_iter()
                        .find(|(_, (ix, iy))| to_screen(*ix, *iy).distance(sp) <= HANDLE_HIT)
                    {
                        Some(RegionDrag {
                            mode: DragMode::Resize { hx, vy },
                            rect: rectf,
                        })
                    } else if r.contains(p.0, p.1) {
                        Some(RegionDrag {
                            mode: DragMode::Move { last: p },
                            rect: rectf,
                        })
                    } else {
                        None
                    }
                });
                self.region_drag = Some(started.unwrap_or(RegionDrag {
                    mode: DragMode::Draw,
                    rect: [p.0, p.1, p.0, p.1],
                }));
            }
        }

        if response.dragged() || response.drag_stopped() {
            if let (Some(drag), Some(sp)) =
                (self.region_drag.as_mut(), response.interact_pointer_pos())
            {
                let p = to_img(sp);
                match &mut drag.mode {
                    DragMode::Draw => {
                        drag.rect[2] = p.0;
                        drag.rect[3] = p.1;
                    }
                    DragMode::Move { last } => {
                        // Translate, keeping the region fully inside the image.
                        let (dx, dy) = (p.0 - last.0, p.1 - last.1);
                        let rw = drag.rect[2] - drag.rect[0];
                        let rh = drag.rect[3] - drag.rect[1];
                        let nx0 = (drag.rect[0] + dx).clamp(0.0, (w as f32 - rw).max(0.0));
                        let ny0 = (drag.rect[1] + dy).clamp(0.0, (h as f32 - rh).max(0.0));
                        drag.rect = [nx0, ny0, nx0 + rw, ny0 + rh];
                        *last = p;
                    }
                    DragMode::Resize { hx, vy } => {
                        if *hx == -1 {
                            drag.rect[0] = p.0;
                        } else if *hx == 1 {
                            drag.rect[2] = p.0;
                        }
                        if *vy == -1 {
                            drag.rect[1] = p.1;
                        } else if *vy == 1 {
                            drag.rect[3] = p.1;
                        }
                    }
                }

                // Commit the working rectangle to the integer region.
                let (x0, x1) = (drag.rect[0].min(drag.rect[2]), drag.rect[0].max(drag.rect[2]));
                let (y0, y1) = (drag.rect[1].min(drag.rect[3]), drag.rect[1].max(drag.rect[3]));
                let new = Region {
                    left: x0.round().max(0.0) as usize,
                    right: (x1.round().max(0.0) as usize).min(w),
                    top: y0.round().max(0.0) as usize,
                    bottom: (y1.round().max(0.0) as usize).min(h),
                }
                .clamped(w, h);
                // A fresh draw only takes effect once it grows beyond a click.
                let keep = match drag.mode {
                    DragMode::Draw => new.right > new.left + 1 || new.bottom > new.top + 1,
                    _ => true,
                };
                if keep && self.region != Some(new) {
                    self.region = Some(new);
                    self.profiles_dirty = true;
                }
            }
            if response.drag_stopped() {
                self.region_drag = None;
            }
        }

        // Hover cursor: resize arrows over a handle, grab inside the region.
        if !response.dragged() {
            if let (Some(hp), Some(r)) = (response.hover_pos(), self.region) {
                if let Some(((hx, vy), _)) = r
                    .handles()
                    .into_iter()
                    .find(|(_, (ix, iy))| to_screen(*ix, *iy).distance(hp) <= HANDLE_HIT)
                {
                    ui.ctx().set_cursor_icon(cursor_for_handle(hx, vy));
                } else {
                    let (ix, iy) = to_img(hp);
                    if r.contains(ix, iy) {
                        ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                    }
                }
            }
        }

        // Draw the region and its handles.
        if let Some(r) = self.region {
            let rr = Rect::from_min_max(
                to_screen(r.left as f32, r.top as f32),
                to_screen(r.right as f32, r.bottom as f32),
            );
            painter.rect_stroke(
                rr,
                egui::CornerRadius::ZERO,
                Stroke::new(2.0, Color32::RED),
                egui::StrokeKind::Middle,
            );
            for (_, (ix, iy)) in r.handles() {
                let hr = Rect::from_center_size(
                    to_screen(ix, iy),
                    egui::vec2(HANDLE_SIZE, HANDLE_SIZE),
                );
                painter.rect_filled(hr, egui::CornerRadius::ZERO, Color32::WHITE);
                painter.rect_stroke(
                    hr,
                    egui::CornerRadius::ZERO,
                    Stroke::new(1.0, Color32::BLACK),
                    egui::StrokeKind::Middle,
                );
            }
        }

        // Crosshair on the marked pixel.
        if let Some((py, px)) = self.pixel_marker {
            let c = to_screen(px as f32 + 0.5, py as f32 + 0.5);
            let arm = 7.0;
            for (a, b) in [
                (Pos2::new(c.x - arm, c.y), Pos2::new(c.x + arm, c.y)),
                (Pos2::new(c.x, c.y - arm), Pos2::new(c.x, c.y + arm)),
            ] {
                painter.line_segment([a, b], Stroke::new(3.0, Color32::BLACK));
                painter.line_segment([a, b], Stroke::new(1.5, Color32::YELLOW));
            }
        }

        // egui turns Ctrl (Cmd on macOS) + wheel into a zoom factor instead
        // of a scroll; a pinch arrives the same way.
        if let Some(p) = response.hover_pos() {
            let factor = ui.input(|i| i.zoom_delta());
            if factor != 1.0 {
                let (ix, iy) = to_img(p);
                return Some((egui::vec2(ix, iy), factor));
            }
        }
        None
    }

    /// Vertical colorbar for the current contrast range (left pane).
    fn colorbar(&self, ui: &mut egui::Ui, height: f32) {
        let Some(tex) = &self.cbar_tex else { return };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(COLORBAR_WIDTH, height), Sense::hover());
        let painter = ui.painter_at(rect);
        let font = egui::TextStyle::Small.resolve(ui.style());
        let text_color = ui.visuals().text_color();

        let pad = font.size * 0.6;
        let bar = Rect::from_min_max(
            Pos2::new(rect.left() + 2.0, rect.top() + pad),
            Pos2::new(rect.left() + 2.0 + 16.0, rect.bottom() - pad),
        );
        if bar.height() < 20.0 {
            return;
        }
        let full_uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
        painter.image(tex.id(), bar, full_uv, Color32::WHITE);
        painter.rect_stroke(
            bar,
            egui::CornerRadius::ZERO,
            Stroke::new(1.0, ui.visuals().widgets.noninteractive.fg_stroke.color),
            egui::StrokeKind::Outside,
        );

        let n_ticks = ((bar.height() / 60.0).floor() as usize + 2).clamp(2, 7);
        for i in 0..n_ticks {
            let t = i as f32 / (n_ticks - 1) as f32;
            let y = bar.bottom() - t * bar.height();
            let v = self.vmin + t * (self.vmax - self.vmin);
            painter.line_segment(
                [Pos2::new(bar.right(), y), Pos2::new(bar.right() + 4.0, y)],
                Stroke::new(1.0, text_color),
            );
            painter.text(
                Pos2::new(bar.right() + 6.0, y),
                egui::Align2::LEFT_CENTER,
                fmt_tick(v),
                font.clone(),
                text_color,
            );
        }
    }
}

fn drag_usize(ui: &mut egui::Ui, v: &mut usize, max: usize) -> bool {
    ui.add(egui::DragValue::new(v).speed(1).range(0..=max)).changed()
}

/// Finite min/max of an image (0..1 fallback for all-NaN data).
fn finite_range(img: &Array2<f32>) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &v in img.iter() {
        if v.is_finite() {
            lo = lo.min(v);
            hi = hi.max(v);
        }
    }
    if !lo.is_finite() || !hi.is_finite() {
        (0.0, 1.0)
    } else {
        (lo, hi)
    }
}

/// Map an image through the colormap LUT into an RGBA texture image.
fn colorize(img: &Array2<f32>, vmin: f32, vmax: f32, lut: &[[u8; 3]; 256]) -> egui::ColorImage {
    let (h, w) = (img.nrows(), img.ncols());
    let span = (vmax - vmin).max(1e-12);
    let mut buf = vec![0u8; w * h * 4];
    for (i, &v) in img.iter().enumerate() {
        let t = ((v - vmin) / span).clamp(0.0, 1.0);
        let idx = ((t * 255.0).round() as usize).min(255);
        let [r, g, b] = lut[idx];
        buf[i * 4] = r;
        buf[i * 4 + 1] = g;
        buf[i * 4 + 2] = b;
        buf[i * 4 + 3] = 255;
    }
    egui::ColorImage::from_rgba_unmultiplied([w, h], &buf)
}

/// Darken and redden the pixels the mask excludes (sizes must match, else
/// the image is left alone).
fn tint_excluded(img: &mut egui::ColorImage, mask: &Array2<bool>) {
    if img.size != [mask.ncols(), mask.nrows()] {
        return;
    }
    for (px, &keep) in img.pixels.iter_mut().zip(mask.iter()) {
        if !keep {
            let [r, g, b, _] = px.to_array();
            *px = Color32::from_rgb(
                (f32::from(r) * 0.3 + 110.0 * 0.7) as u8,
                (f32::from(g) * 0.3) as u8,
                (f32::from(b) * 0.3 + 40.0 * 0.7) as u8,
            );
        }
    }
}

/// Human-readable byte count (KB/MB/GB, decimal).
fn fmt_bytes(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut v = bytes as f64;
    let mut unit = 0;
    while v >= 1000.0 && unit < UNITS.len() - 1 {
        v /= 1000.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[unit])
    }
}

/// Compact tick label for the log y-axis: linear-space value of a tick.
fn fmt_axis(v: f64) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".to_owned()
    } else if a >= 100_000.0 || a < 0.001 {
        format!("{v:.1e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

/// Compact tick label: plain decimals in a comfortable range, scientific
/// notation for very large/small magnitudes.
fn fmt_tick(v: f32) -> String {
    let a = v.abs();
    if a == 0.0 {
        "0".to_owned()
    } else if a >= 100_000.0 || a < 0.001 {
        format!("{v:.2e}")
    } else if a >= 100.0 {
        format!("{v:.0}")
    } else if a >= 1.0 {
        format!("{v:.2}")
    } else {
        format!("{v:.4}")
    }
}

impl eframe::App for DehydrationApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_load();
        self.poll_run_lookup(&ctx);
        self.poll_correction();
        self.poll_export();
        if self.loading.is_some() || self.corr_job.is_some() || self.export_job.is_some() {
            ctx.request_repaint();
        }
        self.recompute_profiles();
        self.rebuild_mask();
        self.ensure_textures(&ctx);
        self.handle_drops(&ctx);

        egui::Panel::top("toolbar").show(ui, |ui| {
            self.toolbar(ui);
        });
        egui::Panel::bottom("status").show(ui, |ui| {
            self.status_bar(ui);
        });
        egui::Panel::left("params")
            .resizable(true)
            .default_size(300.0)
            .show(ui, |ui| {
                // Logos pinned to the bottom-left corner of the window: the
                // parameters scroll in the space above an explicitly reserved
                // logo row, and a spacer pushes the row to the very bottom
                // when the parameters are short.
                let logo_row_height = LOGO_HEIGHT + 24.0;
                let scroll_height = (ui.available_height() - logo_row_height).max(0.0);
                egui::ScrollArea::vertical()
                    .max_height(scroll_height)
                    .show(ui, |ui| {
                        self.params_panel(ui);
                    });
                ui.add_space((ui.available_height() - logo_row_height).max(0.0));
                ui.separator();
                self.logos_row(ui);
            });
        egui::CentralPanel::default().show(ui, |ui| {
            self.viewer(ui);
        });

        self.run_dialog(&ctx);
        self.about_modal(&ctx);
        self.log_window(&ctx);
        self.diagnostics_window(&ctx);
    }
}
