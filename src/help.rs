//! The illustrated step-by-step tutorial, opened from the `? Tutorial`
//! button of the toolbar: one page per step of the workflow, each with a
//! drawing of what the step does (painted with egui, so it follows the
//! theme) and the instructions. Previous / next buttons and the list on the
//! left move between the pages. It is a separate OS window, run by a second
//! process of the binary (`--instructions`), so it can sit next to the
//! application while the steps are followed.

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Overview,
    Window,
    Load,
    Mask,
    Correct,
    Review,
    Export,
    Settings,
}

impl Page {
    const ALL: [Page; 8] = [
        Page::Overview,
        Page::Window,
        Page::Load,
        Page::Mask,
        Page::Correct,
        Page::Review,
        Page::Export,
        Page::Settings,
    ];

    fn title(self) -> &'static str {
        match self {
            Page::Overview => "What the tool does",
            Page::Window => "The window",
            Page::Load => "1. Load data",
            Page::Mask => "2. Mask (optional)",
            Page::Correct => "3. Correct",
            Page::Review => "4. Review",
            Page::Export => "5. Export",
            Page::Settings => "Settings and command line",
        }
    }
}

/// The command-line flag that makes the binary show only this window.
pub const INSTRUCTIONS_FLAG: &str = "--instructions";

/// Launcher of the tutorial window from the application. The window is a
/// second process of the same binary (`--instructions`): a second GPU
/// surface in the application's process loses the device on the software-
/// rendered X servers of the analysis nodes (ThinLinc, no DRI3) and crashes
/// it, a separate process has its own device and window.
#[derive(Default)]
pub struct Help {
    child: Option<std::process::Child>,
}

impl Help {
    /// Open the window; returns Ok(false) when it is already open.
    pub fn launch(&mut self) -> Result<bool, String> {
        if let Some(child) = &mut self.child
            && child.try_wait().map_err(|e| e.to_string())?.is_none()
        {
            return Ok(false);
        }
        let exe = std::env::current_exe().map_err(|e| format!("cannot find the executable: {e}"))?;
        let child = std::process::Command::new(&exe)
            .arg(INSTRUCTIONS_FLAG)
            .stdin(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", exe.display()))?;
        self.child = Some(child);
        Ok(true)
    }
}

/// The window itself: the root of the `--instructions` process.
#[derive(Default)]
pub struct HelpWindow {
    page: usize,
}

impl HelpWindow {
    pub const TITLE: &'static str = "Step-by-step tutorial — Dehydration / Hydration Correction";
    pub const SIZE: [f32; 2] = [1000.0, 720.0];
}

impl eframe::App for HelpWindow {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        if self.contents(ui) {
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

impl HelpWindow {
    /// The navigation, the page and the bottom bar. Returns true when the
    /// Close button was clicked.
    fn contents(&mut self, ui: &mut egui::Ui) -> bool {
        let mut close = false;
        egui::Panel::left("help_nav").exact_size(210.0).resizable(false).show(ui, |ui| {
            ui.add_space(6.0);
            ui.weak("Step by step");
            ui.add_space(4.0);
            for (i, p) in Page::ALL.iter().enumerate() {
                if ui.selectable_label(self.page == i, p.title()).clicked() {
                    self.page = i;
                }
            }
        });
        egui::Panel::bottom("help_nav_buttons").show(ui, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                if ui.add_enabled(self.page > 0, egui::Button::new("◀ Previous")).clicked() {
                    self.page -= 1;
                }
                ui.label(format!("{} / {}", self.page + 1, Page::ALL.len()));
                if ui.add_enabled(self.page + 1 < Page::ALL.len(), egui::Button::new("Next ▶")).clicked() {
                    self.page += 1;
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let page = Page::ALL[self.page];
                ui.heading(page.title());
                ui.add_space(8.0);
                let w = ui.available_width().clamp(400.0, 720.0);
                let (resp, painter) = ui.allocate_painter(Vec2::new(w, 250.0), Sense::hover());
                let pal = Palette::from_visuals(ui.visuals());
                draw(page, &painter, resp.rect, &pal);
                ui.label(
                    egui::RichText::new("Schematic drawing; the real images use the chosen colormap.")
                        .small()
                        .weak(),
                );
                ui.add_space(8.0);
                ui.set_max_width(w);
                text(page, ui);
                ui.add_space(8.0);
            });
        });
        close
    }
}

// ----- text ----------------------------------------------------------------

fn bullet(ui: &mut egui::Ui, s: &str) {
    ui.horizontal_wrapped(|ui| {
        ui.label("•");
        ui.label(s);
    });
}

fn text(page: Page, ui: &mut egui::Ui) {
    match page {
        Page::Overview => {
            ui.label(
                "A hyperspectral (time-of-flight) stack has one image per wavelength bin, and each \
                 image has few counts: it is noisy. But a sample is made of a few materials only, so \
                 every pixel's spectrum is a mix of a few material spectra. The tool finds those \
                 spectra and how much of each sits in every pixel (dehydration), then rebuilds the \
                 stack from them (rehydration): the structure stays, most of the noise goes.",
            );
            ui.add_space(6.0);
            ui.monospace("stack X (pixels × bins)  ≈  maps W (pixels × materials) · spectra H (materials × bins)");
            ui.add_space(6.0);
            bullet(ui, "The fit is the maximum-likelihood factorization of mbirtorch's hsnt package (Harel Dor), run on the GPU: it uses the Poisson statistics of the counts.");
            bullet(ui, "The number of materials can be given, or estimated from the data.");
            bullet(ui, "Input: a normalized stack (transmission), as TIFF images in one folder. Output: the corrected stack as TIFF images with the same names.");
            bullet(ui, "The steps are the pages of this window: load data, mask (optional), correct, review, export.");
        }
        Page::Window => {
            bullet(ui, "Step bar (top): the five steps in order. A green check marks a step that is done, the line under each title gives its state (number of images, share of the image in the mask, 'running… 45%', number of materials found, …). The current step is highlighted.");
            bullet(ui, "Click a step to open it. A greyed step is not possible yet; hover it to read what is missing (no data, no result).");
            bullet(ui, "Left panel: the controls of the current step only. '◀ Back' and 'Next ▶' at its bottom move one step.");
            bullet(ui, "Toolbar (under the step bar): what the images show — view (Raw data / Corrected vs raw / Profiles), image slider, contrast, colormap, zoom. On the right: Log, Diagnostics, light / dark theme, text size, this tutorial and the mbirtorch information.");
            bullet(ui, "Images: Ctrl + mouse wheel zooms on the pointer; the status bar at the bottom gives the pixel under the pointer and its value.");
            bullet(ui, "When a correction is done the window opens the Review step by itself.");
        }
        Page::Load => {
            bullet(ui, "Open Folder…: a folder of TIFF images, one per TOF bin. Open Files…: a selection of files. Both dialogs start in the chosen IPTS (the list at the top of the panel) when one is set.");
            bullet(ui, "Run number…: type a run number; the tool finds its NeXus file, the detector, the image folder and the detector offset. A Timepix run asks for the raw or the autoreduce data — autoreduce is the one to load.");
            bullet(ui, "Recent: the last folders loaded. A folder or files can also be dropped on the window.");
            bullet(ui, "Detector: decides how the frames are oriented on load. 'auto' recognizes it from the path; force it if the data lives elsewhere.");
            bullet(ui, "Once loaded, the left pane shows one image of the stack (Image slider in the toolbar), the right pane the integrated image: the sum of all the images.");
            bullet(ui, "Load settings from HDF5…: applies the mask and the parameters saved in an earlier session.");
        }
        Page::Mask => {
            ui.label(
                "Optional. The mask chooses which pixels are corrected; the others are not sent to \
                 mbirtorch and keep their raw values. Use it to leave out what is not the sample: \
                 the open beam, a holder, a region filled with zeros. Skip the step to correct \
                 every pixel.",
            );
            ui.add_space(4.0);
            bullet(ui, "Draw shapes: pick Rectangle or Ellipse and drag on either image; pick Polygon and click the points, then double-click, press Enter or click the first point to close it (Esc cancels). A shape shows on both images.");
            bullet(ui, "New shapes: 'include' shapes (green) keep only what is inside them; 'exclude' shapes (red) remove what is inside. With several include shapes, a pixel inside any of them is kept.");
            bullet(ui, "After a shape is drawn the tool goes back to 'Select / move': click a shape to select it, drag it to move it, drag its handles to reshape it.");
            bullet(ui, "Change your mind: click the include / exclude button of a shape in the list, or right-click the shape on the image, to switch it. ✖ in the list, the Delete key or the right-click menu deletes it.");
            bullet(ui, "Select by brightness: keeps the pixels whose value in the integrated image lies between a minimum and a maximum. Raise the minimum to drop the dark surroundings, lower the maximum to drop the open beam.");
            bullet(ui, "Use a mask image: a TIFF or .npy the size of the frames, non-zero where the pixels are kept (saved here, or made with the Sample / Hyperspectral Masker).");
            bullet(ui, "The parts in use combine: a pixel must pass all of them. The pixels left out are shaded dark red, and the top of the panel says how much of the image will be corrected.");
        }
        Page::Correct => {
            bullet(ui, "Input type: 'transmission' for normalized data (the usual case), 'attenuation' for −log(transmission), 'auto' to let the tool decide from the values.");
            bullet(ui, "Number of materials: 'auto' estimates it from the data (Max materials bounds the search); 'fixed' takes your number. With a mask the estimate can come out low: give the number if it does.");
            bullet(ui, "Spectra: 'mle' is the default; 'unconstrained' removes a low-dose bias and is worth it with many pixels; 'support' also zeroes the background of the maps and needs the dose.");
            bullet(ui, "Dose known: open-beam counts per pixel and bin. With it the result reports a reduced chi-square: about 1 means the fit is at the noise level.");
            bullet(ui, "Device: 'auto' takes the first GPU, else the CPU. 'Advanced (solver)' holds the solver mode, step and tolerance limits; the defaults are fine to start.");
            bullet(ui, "⚡ Preview runs on 2×2-binned pixels: fast, to tune the parameters; a preview cannot be exported. ▶ Perform correction runs on the full stack.");
            bullet(ui, "While it runs: the progress bar follows mbirtorch (loading, rank estimate, solve, outputs), ✖ Cancel stops it, 📜 Log shows everything it printed. The log opens by itself if the run fails.");
        }
        Page::Review => {
            bullet(ui, "The panel sums up the result: number of materials (and how it was obtained), solve mode, steps and time, the reduced chi-square with its verdict (or the relative residual when no dose was given), and the warnings of mbirtorch.");
            bullet(ui, "Corrected vs raw: the corrected image next to the raw one, image by image (Image slider). 'Difference' shows corrected − raw: it should look like noise; structure there is something the correction removed or invented.");
            bullet(ui, "Profiles: the mean of a region along the stack, before (orange ×) and after (blue ●). Drag the box on the image to move it, its handles to resize it; click a pixel for its own spectrum. The x axis can be the image index, the TOF or the wavelength; 'Save profiles…' writes them as CSV.");
            bullet(ui, "📈 Diagnostics: the material spectra and maps plotted by mbirtorch, with its data checks.");
            bullet(ui, "Not satisfied? Go back to step 3 (or 2), change the parameters and run again: the step bar says 'parameters changed' when the result no longer matches them.");
        }
        Page::Export => {
            bullet(ui, "💾 Export corrected images…: choose an output folder. The tool creates dehydrated_hydrated_<input folder> inside it (numbered if it already exists).");
            bullet(ui, "It holds the corrected images (32-bit float TIFF, same names and orientation as the input), a copy of the *_Spectra.txt so the TOF axis follows the stack, and correction_config.json with every parameter and the fit summary.");
            bullet(ui, "Also written: the mbirtorch report, its log, the spectra and maps plots, the dehydrated file (hsnt_dehydrated.h5: maps and spectra), the material maps as images and, when a mask was used, mask.tif.");
            bullet(ui, "A preview result is binned and cannot be exported: run the full correction first.");
            bullet(ui, "💾 Save settings to HDF5…: keeps the mask, the parameters and the display settings, to load them again in step 1 or to use them in the workflow runner.");
        }
        Page::Settings => {
            bullet(ui, "The settings file (.h5) is saved in the Export step and loaded in the Load data step. It holds the correction parameters, the mask (shapes, brightness range, mask image path), the detector offset and distance, the display settings and the profile region.");
            bullet(ui, "The same binary runs without the window:");
            ui.monospace("dehydration_hydration FOLDER --run --output OUT_DIR --input-type transmission --materials auto");
            ui.monospace("dehydration_hydration --run-number 12345 --run --output OUT_DIR --materials 3 --dose 50");
            ui.monospace("dehydration_hydration FOLDER --run --output OUT_DIR --mask sample_mask.tif");
            ui.monospace("dehydration_hydration FOLDER --run --output OUT_DIR --mask-exclude-ellipse 256,256,40,40");
            bullet(ui, "--help lists every option (--spectra, --device, --mask-include / --mask-exclude and their -ellipse and -polygon forms, --mask-range, --bin, --offset, --detector, …).");
            bullet(ui, "Every session is logged in /SNS/VENUS/shared/log (data loaded, mask, parameters, runs, exports, errors) and shows up in the Log Browser.");
            bullet(ui, "The toolbar has the light / dark theme and the text size toggles, shared with the other VENUS tools.");
        }
    }
}

// ----- drawings ------------------------------------------------------------

struct Palette {
    text: Color32,
    weak: Color32,
    frame: Color32,
    frame_bg: Color32,
    red: Color32,
    blue: Color32,
    green: Color32,
    orange: Color32,
    accent: Color32,
    excluded: Color32,
}

impl Palette {
    fn from_visuals(v: &egui::Visuals) -> Self {
        Palette {
            text: v.text_color(),
            weak: v.weak_text_color(),
            frame: v.widgets.inactive.fg_stroke.color,
            frame_bg: if v.dark_mode { Color32::from_gray(28) } else { Color32::from_gray(235) },
            red: Color32::from_rgb(255, 70, 70),
            blue: Color32::from_rgb(60, 140, 255),
            green: Color32::from_rgb(70, 200, 90),
            orange: Color32::from_rgb(255, 140, 0),
            accent: v.selection.bg_fill,
            excluded: Color32::from_rgba_unmultiplied(120, 0, 30, 170),
        }
    }
}

/// Viridis-like ramp: 0 = dark purple, 1 = yellow.
fn ramp(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let stops = [(68.0, 1.0, 84.0), (59.0, 82.0, 139.0), (33.0, 145.0, 140.0), (94.0, 201.0, 98.0), (253.0, 231.0, 37.0)];
    let x = t * (stops.len() - 1) as f32;
    let i = (x.floor() as usize).min(stops.len() - 2);
    let f = x - i as f32;
    let (a, b) = (stops[i], stops[i + 1]);
    let l = |p: f32, q: f32| (p + (q - p) * f) as u8;
    Color32::from_rgb(l(a.0, b.0), l(a.1, b.1), l(a.2, b.2))
}

fn label(p: &egui::Painter, pos: Pos2, anchor: Align2, s: &str, color: Color32) {
    p.text(pos, anchor, s, FontId::proportional(12.0), color);
}

fn small(p: &egui::Painter, pos: Pos2, anchor: Align2, s: &str, color: Color32) {
    p.text(pos, anchor, s, FontId::proportional(10.5), color);
}

fn dashed(p: &egui::Painter, a: Pos2, b: Pos2, stroke: Stroke) {
    let v = b - a;
    let len = v.length();
    if len <= 0.0 {
        return;
    }
    let d = v / len;
    let (dash, gap) = (6.0, 4.0);
    let mut t = 0.0;
    while t < len {
        let e = (t + dash).min(len);
        p.line_segment([a + d * t, a + d * e], stroke);
        t += dash + gap;
    }
}

/// An "image" area: background and border.
fn frame(p: &egui::Painter, r: Rect, pal: &Palette) {
    p.rect_filled(r, 2.0, pal.frame_bg);
    p.rect_stroke(r, 2.0, Stroke::new(1.0, pal.frame), StrokeKind::Outside);
}

/// Repeatable noise in −0.5..0.5 for cell `(i, j)`.
fn noise(i: i32, j: i32, seed: u32) -> f32 {
    let mut h = (i as u32).wrapping_mul(374_761_393) ^ (j as u32).wrapping_mul(668_265_263) ^ seed.wrapping_mul(2_246_822_519);
    h = (h ^ (h >> 13)).wrapping_mul(1_274_126_177);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0 - 0.5
}

/// A radiograph of the tutorial's sample inside `img`: a bright open beam
/// around a darker disc that holds a still darker inclusion. `noise_amp`
/// > 0 gives the grainy look of a low-count image.
fn sample(p: &egui::Painter, img: Rect, noise_amp: f32, seed: u32) {
    let p = p.with_clip_rect(img);
    let cell = 5.0;
    let (nx, ny) = ((img.width() / cell).ceil() as i32, (img.height() / cell).ceil() as i32);
    let c = img.center();
    let r = 0.36 * img.width().min(img.height() * 1.25);
    let inc = Pos2::new(c.x + 0.35 * r, c.y - 0.3 * r);
    for j in 0..ny {
        for i in 0..nx {
            let q = Pos2::new(img.left() + (i as f32 + 0.5) * cell, img.top() + (j as f32 + 0.5) * cell);
            let mut v = 0.88;
            if (q - c).length() < r {
                v = 0.42;
            }
            if (q - inc).length() < 0.28 * r {
                v = 0.18;
            }
            v += noise_amp * noise(i, j, seed);
            p.rect_filled(
                Rect::from_min_size(Pos2::new(img.left() + i as f32 * cell, img.top() + j as f32 * cell), Vec2::splat(cell + 0.5)),
                0.0,
                ramp(v),
            );
        }
    }
}

/// A few frames stacked behind each other, the front one drawn by `front`.
fn stack(p: &egui::Painter, first: Rect, n: usize, pal: &Palette, front: impl Fn(&egui::Painter, Rect)) {
    for i in (0..n).rev() {
        let r = first.translate(Vec2::splat(7.0 * i as f32));
        frame(p, r, pal);
        if i == 0 {
            front(p, r.shrink(1.0));
        }
    }
}

/// A plot of `f` over 0..1 (y also in 0..1, bottom = 0) inside `r`.
fn curve(p: &egui::Painter, r: Rect, f: impl Fn(f32) -> f32, stroke: Stroke) {
    let n = 80;
    let pts: Vec<Pos2> = (0..=n)
        .map(|i| {
            let x = i as f32 / n as f32;
            Pos2::new(r.left() + x * r.width(), r.bottom() - f(x).clamp(0.0, 1.0) * r.height())
        })
        .collect();
    p.add(egui::Shape::line(pts, stroke));
}

fn axes(p: &egui::Painter, r: Rect, x_title: &str, y_title: &str, pal: &Palette) {
    let s = Stroke::new(1.0, pal.weak);
    p.line_segment([r.left_bottom(), r.right_bottom()], s);
    p.line_segment([r.left_bottom(), r.left_top()], s);
    small(p, r.center_bottom() + Vec2::new(0.0, 3.0), Align2::CENTER_TOP, x_title, pal.weak);
    small(p, r.left_top() + Vec2::new(-3.0, 0.0), Align2::RIGHT_TOP, y_title, pal.weak);
}

/// A rounded box with a title, as a step of a flow chart.
fn step_box(p: &egui::Painter, r: Rect, title: &str, sub: &str, pal: &Palette) {
    p.rect_filled(r, 6.0, pal.frame_bg);
    p.rect_stroke(r, 6.0, Stroke::new(1.0, pal.accent), StrokeKind::Outside);
    label(p, r.center() - Vec2::new(0.0, 7.0), Align2::CENTER_CENTER, title, pal.text);
    small(p, r.center() + Vec2::new(0.0, 9.0), Align2::CENTER_CENTER, sub, pal.weak);
}

fn arrow_between(p: &egui::Painter, a: Rect, b: Rect, pal: &Palette) {
    let from = Pos2::new(a.right() + 4.0, a.center().y);
    let to = Pos2::new(b.left() - 4.0, b.center().y);
    p.arrow(from, to - from, Stroke::new(1.5, pal.weak));
}

/// A spectrum with a Bragg edge at `edge` (0..1) and height `level`.
fn spectrum(x: f32, edge: f32, level: f32) -> f32 {
    let step = 1.0 / (1.0 + (-(x - edge) * 40.0).exp());
    level * (0.35 + 0.25 * x) + 0.3 * step * level
}

/// The handles of a selected shape: small white squares.
fn handles(p: &egui::Painter, r: Rect) {
    for c in [r.left_top(), r.center_top(), r.right_top(), r.left_center(), r.right_center(), r.left_bottom(), r.center_bottom(), r.right_bottom()] {
        let h = Rect::from_center_size(c, Vec2::splat(6.0));
        p.rect_filled(h, 1.0, Color32::WHITE);
        p.rect_stroke(h, 1.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Middle);
    }
}

fn draw(page: Page, p: &egui::Painter, area: Rect, pal: &Palette) {
    let area = area.shrink(4.0);
    match page {
        Page::Overview => draw_overview(p, area, pal),
        Page::Window => draw_window(p, area, pal),
        Page::Load => draw_load(p, area, pal),
        Page::Mask => draw_mask(p, area, pal),
        Page::Correct => draw_correct(p, area, pal),
        Page::Review => draw_review(p, area, pal),
        Page::Export => draw_export(p, area, pal),
        Page::Settings => draw_settings(p, area, pal),
    }
}

fn draw_overview(p: &egui::Painter, area: Rect, pal: &Palette) {
    // Noisy stack → maps × spectra → denoised stack.
    let iw = ((area.width() - 330.0) / 2.0).clamp(110.0, 170.0);
    let ih = iw * 0.8;
    let y = area.top() + 34.0;
    let raw = Rect::from_min_size(Pos2::new(area.left() + 4.0, y), Vec2::new(iw, ih));
    stack(p, raw, 4, pal, |p, r| sample(p, r, 0.55, 1));
    label(p, raw.center_top() - Vec2::new(0.0, 6.0), Align2::CENTER_BOTTOM, "noisy stack", pal.text);
    small(p, Pos2::new(raw.center().x + 10.0, raw.bottom() + 26.0), Align2::CENTER_TOP, "one image per wavelength bin", pal.weak);

    // Middle: the dehydrated form.
    let mx = raw.right() + 70.0;
    let mw = 170.0;
    let mid = Rect::from_min_size(Pos2::new(mx, area.top() + 10.0), Vec2::new(mw, area.height() - 40.0));
    p.rect_stroke(mid, 6.0, Stroke::new(1.0, pal.accent), StrokeKind::Outside);
    label(p, mid.center_top() + Vec2::new(0.0, 6.0), Align2::CENTER_TOP, "dehydrated: a few materials", pal.text);
    let map_w = (mw - 36.0) / 2.0;
    for (k, lvl) in [0.42f32, 0.18].iter().enumerate() {
        let m = Rect::from_min_size(Pos2::new(mid.left() + 12.0 + k as f32 * (map_w + 12.0), mid.top() + 30.0), Vec2::new(map_w, map_w * 0.8));
        frame(p, m, pal);
        let pc = p.with_clip_rect(m);
        pc.rect_filled(m.shrink(1.0), 0.0, ramp(0.05));
        let c = m.center();
        let r = 0.36 * m.width();
        if k == 0 {
            pc.circle_filled(c, r, ramp(0.75));
            pc.circle_filled(Pos2::new(c.x + 0.35 * r, c.y - 0.3 * r), 0.28 * r, ramp(0.05));
        } else {
            pc.circle_filled(Pos2::new(c.x + 0.35 * r, c.y - 0.3 * r), 0.28 * r, ramp(0.9));
        }
        small(p, m.center_bottom() + Vec2::new(0.0, 2.0), Align2::CENTER_TOP, &format!("map {}", k + 1), pal.weak);
        let _ = lvl;
    }
    let sp = Rect::from_min_max(Pos2::new(mid.left() + 22.0, mid.top() + 30.0 + map_w * 0.8 + 28.0), Pos2::new(mid.right() - 12.0, mid.bottom() - 20.0));
    axes(p, sp, "wavelength", "", pal);
    curve(p, sp, |x| spectrum(x, 0.35, 0.9), Stroke::new(2.0, pal.blue));
    curve(p, sp, |x| spectrum(x, 0.65, 0.6), Stroke::new(2.0, pal.orange));
    small(p, sp.right_top() + Vec2::new(0.0, -2.0), Align2::RIGHT_BOTTOM, "spectra", pal.weak);

    let out = Rect::from_min_size(Pos2::new(mid.right() + 70.0, y), Vec2::new(iw, ih));
    stack(p, out, 4, pal, |p, r| sample(p, r, 0.06, 2));
    label(p, out.center_top() - Vec2::new(0.0, 6.0), Align2::CENTER_BOTTOM, "corrected stack", pal.text);
    small(p, Pos2::new(out.center().x + 10.0, out.bottom() + 26.0), Align2::CENTER_TOP, "same structure, less noise", pal.weak);

    for (a, b, t) in [(raw.right() + 24.0, mid.left(), "dehydrate"), (mid.right(), out.left(), "rehydrate")] {
        let yy = raw.center().y;
        p.arrow(Pos2::new(a + 2.0, yy), Vec2::new(b - a - 8.0, 0.0), Stroke::new(2.0, pal.text));
        small(p, Pos2::new((a + b) / 2.0, yy - 4.0), Align2::CENTER_BOTTOM, t, pal.text);
    }
}

fn draw_window(p: &egui::Painter, area: Rect, pal: &Palette) {
    // A miniature of the window with its four areas named.
    let win = Rect::from_min_size(Pos2::new(area.left() + 4.0, area.top() + 4.0), Vec2::new((area.width() * 0.66).clamp(340.0, 480.0), area.height() - 12.0));
    frame(p, win, pal);
    let bar = Rect::from_min_size(win.left_top(), Vec2::new(win.width(), 34.0));
    let tool = Rect::from_min_size(bar.left_bottom(), Vec2::new(win.width(), 18.0));
    let status = Rect::from_min_max(Pos2::new(win.left(), win.bottom() - 16.0), win.right_bottom());
    let panel = Rect::from_min_max(tool.left_bottom(), Pos2::new(win.left() + 0.28 * win.width(), status.top()));
    let s = Stroke::new(1.0, pal.frame);
    p.line_segment([bar.left_bottom(), bar.right_bottom()], s);
    p.line_segment([tool.left_bottom(), tool.right_bottom()], s);
    p.line_segment([status.left_top(), status.right_top()], s);
    p.line_segment([panel.right_top(), panel.right_bottom()], s);

    // Step bar: five discs, two done, the third current.
    let n = 5;
    let dx = (bar.width() - 30.0) / n as f32;
    for k in 0..n {
        let c = Pos2::new(bar.left() + 22.0 + k as f32 * dx, bar.center().y);
        if k == 2 {
            p.rect_stroke(Rect::from_min_size(Pos2::new(c.x - 13.0, c.y - 13.0), Vec2::new(dx - 14.0, 26.0)), 5.0, Stroke::new(1.5, pal.accent), StrokeKind::Inside);
        }
        let (fill, txt) = if k < 2 { (Color32::from_rgb(60, 160, 90), "✔".to_owned()) } else { (pal.frame, (k + 1).to_string()) };
        p.circle_filled(c, 9.0, fill);
        p.text(c, Align2::CENTER_CENTER, txt, FontId::proportional(10.0), if k < 2 { Color32::WHITE } else { pal.frame_bg });
        let name = ["Load", "Mask", "Correct", "Review", "Export"][k];
        let tr = p.text(c + Vec2::new(13.0, 0.0), Align2::LEFT_CENTER, name, FontId::proportional(10.0), if k == 2 { pal.text } else { pal.weak });
        let (from, to) = (tr.right() + 4.0, c.x + dx - 12.0);
        if k + 1 < n && to > from {
            p.line_segment([Pos2::new(from, c.y), Pos2::new(to, c.y)], Stroke::new(1.5, if k < 2 { Color32::from_rgb(60, 160, 90) } else { pal.frame }));
        }
    }
    // Toolbar ticks, panel lines, two image panes.
    for k in 0..6 {
        p.rect_filled(Rect::from_min_size(tool.left_top() + Vec2::new(8.0 + 44.0 * k as f32, 5.0), Vec2::new(34.0, 8.0)), 2.0, pal.frame);
    }
    for k in 0..6 {
        p.rect_filled(Rect::from_min_size(panel.left_top() + Vec2::new(8.0, 10.0 + 16.0 * k as f32), Vec2::new(panel.width() - 16.0 - 12.0 * (k % 3) as f32, 7.0)), 2.0, pal.frame);
    }
    let nav = Rect::from_min_max(Pos2::new(panel.left() + 6.0, panel.bottom() - 18.0), Pos2::new(panel.right() - 6.0, panel.bottom() - 5.0));
    p.rect_stroke(nav, 3.0, Stroke::new(1.0, pal.accent), StrokeKind::Inside);
    small(p, nav.center(), Align2::CENTER_CENTER, "◀ Back      Next ▶", pal.text);
    let view = Rect::from_min_max(panel.right_top(), status.right_top()).shrink(8.0);
    let pw = (view.width() - 8.0) / 2.0;
    let ph = (pw * 0.8).min(view.height());
    let a = Rect::from_min_size(view.left_top(), Vec2::new(pw, ph));
    let b = a.translate(Vec2::new(pw + 8.0, 0.0));
    sample(p, a, 0.5, 3);
    sample(p, b, 0.05, 4);

    // Call-outs.
    let x = win.right() + 26.0;
    let notes = [
        (bar.center().y, "Step bar", "the five steps and their state"),
        (tool.center().y, "Toolbar", "view, contrast, zoom"),
        (panel.center().y, "Left panel", "the controls of the current step"),
        (panel.bottom() - 12.0, "Back / Next", "move one step"),
    ];
    let mut y = area.top() + 18.0;
    for (ty, title, sub) in notes {
        let from = Pos2::new(x - 4.0, y + 6.0);
        let to = Pos2::new(if title == "Left panel" || title == "Back / Next" { panel.right() - 4.0 } else { win.right() - 6.0 }, ty);
        p.line_segment([from, to], Stroke::new(1.0, pal.weak));
        p.circle_filled(to, 2.5, pal.weak);
        label(p, Pos2::new(x, y), Align2::LEFT_TOP, title, pal.text);
        small(p, Pos2::new(x, y + 15.0), Align2::LEFT_TOP, sub, pal.weak);
        y += 52.0;
    }
}

fn draw_load(p: &egui::Painter, area: Rect, pal: &Palette) {
    // Four ways in, one stack out, two panes shown.
    let bw = 150.0;
    let bh = 40.0;
    let ways = [("Open Folder…", "one TIFF per TOF bin"), ("Open Files…", "a selection of files"), ("Run number…", "found from the NeXus file"), ("Recent / drop", "last folders, drag and drop")];
    let total = ways.len() as f32 * (bh + 10.0) - 10.0;
    let y0 = area.center().y - total / 2.0;
    let mut boxes = Vec::new();
    for (i, (t, s)) in ways.iter().enumerate() {
        let r = Rect::from_min_size(Pos2::new(area.left() + 4.0, y0 + i as f32 * (bh + 10.0)), Vec2::new(bw, bh));
        step_box(p, r, t, s, pal);
        boxes.push(r);
    }
    let iw = ((area.width() - bw - 250.0) / 2.0).clamp(100.0, 150.0);
    let ih = iw * 0.8;
    let st = Rect::from_min_size(Pos2::new(area.left() + bw + 70.0, area.center().y - ih / 2.0 - 10.0), Vec2::new(iw, ih));
    for b in &boxes {
        let from = Pos2::new(b.right() + 4.0, b.center().y);
        let to = Pos2::new(st.left() - 6.0, st.center().y);
        p.arrow(from, to - from, Stroke::new(1.2, pal.weak));
    }
    stack(p, st, 4, pal, |p, r| sample(p, r, 0.55, 5));
    label(p, st.center_top() - Vec2::new(0.0, 6.0), Align2::CENTER_BOTTOM, "left pane: one image", pal.text);
    small(p, Pos2::new(st.center().x + 10.0, st.bottom() + 26.0), Align2::CENTER_TOP, "Image slider picks which", pal.weak);

    let sum = Rect::from_min_size(Pos2::new(st.right() + 90.0, st.top()), Vec2::new(iw, ih));
    frame(p, sum, pal);
    sample(p, sum.shrink(1.0), 0.08, 6);
    label(p, sum.center_top() - Vec2::new(0.0, 6.0), Align2::CENTER_BOTTOM, "right pane: integrated image", pal.text);
    small(p, sum.center_bottom() + Vec2::new(0.0, 6.0), Align2::CENTER_TOP, "the sum of all the images", pal.weak);
    let a = Pos2::new(st.right() + 28.0, st.center().y);
    p.arrow(a, Vec2::new(sum.left() - a.x - 6.0, 0.0), Stroke::new(2.0, pal.text));
    small(p, Pos2::new((a.x + sum.left()) / 2.0, a.y - 4.0), Align2::CENTER_BOTTOM, "Σ", pal.text);
}

fn draw_mask(p: &egui::Painter, area: Rect, pal: &Palette) {
    let iw = (area.width() * 0.42).clamp(200.0, 300.0);
    let ih = area.height() - 26.0;
    let img = Rect::from_min_size(Pos2::new(area.left() + 4.0, area.top() + 4.0), Vec2::new(iw, ih));
    frame(p, img, pal);
    let inner = img.shrink(1.0);
    sample(p, inner, 0.08, 7);
    let pc = p.with_clip_rect(inner);
    let c = inner.center();
    let r = 0.36 * inner.width().min(inner.height() * 1.25);
    // Include rectangle around the sample; everything outside is shaded.
    let inc = Rect::from_center_size(c, Vec2::new(2.5 * r, 2.3 * r));
    for o in [
        Rect::from_min_max(inner.left_top(), Pos2::new(inner.right(), inc.top())),
        Rect::from_min_max(Pos2::new(inner.left(), inc.bottom()), inner.right_bottom()),
        Rect::from_min_max(Pos2::new(inner.left(), inc.top()), Pos2::new(inc.left(), inc.bottom())),
        Rect::from_min_max(Pos2::new(inc.right(), inc.top()), Pos2::new(inner.right(), inc.bottom())),
    ] {
        pc.rect_filled(o, 0.0, pal.excluded);
    }
    pc.rect_stroke(inc, 0.0, Stroke::new(2.0, pal.green), StrokeKind::Middle);
    // Exclude ellipse on the inclusion, selected (handles).
    let ec = Pos2::new(c.x + 0.35 * r, c.y - 0.3 * r);
    let er = Vec2::new(0.36 * r, 0.34 * r);
    pc.add(egui::Shape::ellipse_filled(ec, er, pal.excluded));
    pc.add(egui::Shape::ellipse_stroke(ec, er, Stroke::new(2.0, pal.red)));
    handles(&pc, Rect::from_center_size(ec, er * 2.0));
    // A polygon being drawn, bottom left.
    let poly = [
        Pos2::new(c.x - 0.95 * r, c.y + 0.25 * r),
        Pos2::new(c.x - 0.45 * r, c.y + 0.35 * r),
        Pos2::new(c.x - 0.35 * r, c.y + 0.85 * r),
        Pos2::new(c.x - 0.8 * r, c.y + 0.9 * r),
    ];
    pc.add(egui::Shape::line(poly.to_vec(), Stroke::new(2.0, Color32::YELLOW)));
    pc.circle_stroke(poly[0], 4.0, Stroke::new(2.0, Color32::YELLOW));
    small(p, img.left_bottom() + Vec2::new(0.0, 4.0), Align2::LEFT_TOP, "yellow = being drawn, dark red = left out", pal.weak);

    // Right: the legend of the gestures, then the brightness selection.
    let x = img.right() + 24.0;
    let mut y = area.top() + 4.0;
    let rows = [
        (pal.green, "include", "keeps only what is inside"),
        (pal.red, "exclude", "removes what is inside"),
        (pal.text, "drag a shape", "moves it"),
        (pal.text, "drag a handle ▫", "reshapes it"),
        (pal.text, "right-click / list button", "switches include ↔ exclude"),
    ];
    for (color, a, b) in rows {
        label(p, Pos2::new(x, y), Align2::LEFT_TOP, a, color);
        small(p, Pos2::new(x + 150.0, y + 1.0), Align2::LEFT_TOP, b, pal.weak);
        y += 19.0;
    }
    y += 12.0;
    label(p, Pos2::new(x, y), Align2::LEFT_TOP, "Select by brightness", pal.text);
    let h = Rect::from_min_max(Pos2::new(x + 34.0, y + 36.0), Pos2::new((area.right() - 10.0).max(x + 120.0), area.bottom() - 22.0));
    axes(p, h, "value in the integrated image", "pixels", pal);
    // Histogram: dark surroundings, sample, open beam.
    let hist = |v: f32| {
        let g = |c: f32, w: f32, a: f32| a * (-((v - c) * (v - c)) / w).exp();
        g(0.12, 0.004, 0.5) + g(0.45, 0.012, 0.85) + g(0.86, 0.004, 0.65)
    };
    let (lo, hi) = (0.26f32, 0.68f32);
    let keep = Rect::from_min_max(Pos2::new(h.left() + lo * h.width(), h.top()), Pos2::new(h.left() + hi * h.width(), h.bottom()));
    p.rect_filled(keep, 0.0, pal.green.gamma_multiply(0.18));
    curve(p, h, hist, Stroke::new(2.0, pal.accent));
    for (v, t) in [(lo, "minimum"), (hi, "maximum")] {
        let xx = h.left() + v * h.width();
        dashed(p, Pos2::new(xx, h.top()), Pos2::new(xx, h.bottom()), Stroke::new(1.5, pal.green));
        small(p, Pos2::new(xx, h.top() - 2.0), Align2::CENTER_BOTTOM, t, pal.green);
    }
    small(p, Pos2::new(h.left() + 0.12 * h.width(), h.bottom() - 0.5 * h.height() - 12.0), Align2::CENTER_BOTTOM, "dark", pal.weak);
    small(p, Pos2::new(h.left() + 0.45 * h.width(), h.bottom() - 0.5 * h.height()), Align2::CENTER_CENTER, "sample: kept", pal.text);
    small(p, Pos2::new(h.left() + 0.86 * h.width(), h.bottom() - 0.65 * h.height() - 12.0), Align2::CENTER_BOTTOM, "open beam", pal.weak);
}

fn draw_correct(p: &egui::Painter, area: Rect, pal: &Palette) {
    // Parameters → Preview / Perform → progress → Review.
    let gap = 24.0;
    let bw = ((area.width() - 3.0 * gap - 8.0) / 4.0).clamp(110.0, 165.0);
    let bh = 54.0;
    let y = area.top() + 10.0;
    let b1 = Rect::from_min_size(Pos2::new(area.left() + 4.0, y), Vec2::new(bw, bh));
    let b2 = b1.translate(Vec2::new(bw + gap, 0.0));
    let b3 = b2.translate(Vec2::new(bw + gap, 0.0));
    let b4 = b3.translate(Vec2::new(bw + gap, 0.0));
    step_box(p, b1, "Parameters", "defaults to start with", pal);
    step_box(p, b2, "⚡ Preview", "2×2 binned, fast", pal);
    step_box(p, b3, "▶ Perform correction", "full stack", pal);
    step_box(p, b4, "4. Review", "opens by itself", pal);
    arrow_between(p, b1, b2, pal);
    arrow_between(p, b2, b3, pal);
    arrow_between(p, b3, b4, pal);
    // Loop back from the preview to the parameters.
    let (a, b) = (Pos2::new(b2.center().x, b2.bottom() + 4.0), Pos2::new(b1.center().x, b1.bottom() + 4.0));
    p.line_segment([a, a + Vec2::new(0.0, 14.0)], Stroke::new(1.2, pal.weak));
    p.line_segment([a + Vec2::new(0.0, 14.0), b + Vec2::new(0.0, 14.0)], Stroke::new(1.2, pal.weak));
    p.arrow(b + Vec2::new(0.0, 14.0), Vec2::new(0.0, -12.0), Stroke::new(1.2, pal.weak));
    small(p, Pos2::new((a.x + b.x) / 2.0, a.y + 16.0), Align2::CENTER_TOP, "tune and try again", pal.weak);

    // The parameters that matter.
    let ty = y + bh + 44.0;
    let rows = [
        ("Input type", "transmission (normalized data) · attenuation · auto"),
        ("Number of materials", "auto (estimated, up to Max materials) · fixed N"),
        ("Spectra", "mle (default) · unconstrained · support (needs the dose)"),
        ("Dose known", "open-beam counts per pixel and bin → reduced chi-square"),
        ("Device", "auto (GPU when there is one) · cpu · cuda:N"),
    ];
    for (i, (k, v)) in rows.iter().enumerate() {
        let yy = ty + 17.0 * i as f32;
        label(p, Pos2::new(area.left() + 8.0, yy), Align2::LEFT_TOP, k, pal.text);
        small(p, Pos2::new(area.left() + 160.0, yy + 1.0), Align2::LEFT_TOP, v, pal.weak);
    }
    // Progress bar.
    let pb = Rect::from_min_size(Pos2::new(area.left() + 8.0, area.bottom() - 26.0), Vec2::new((area.width() * 0.55).min(360.0), 16.0));
    p.rect_filled(pb, 4.0, pal.frame_bg);
    p.rect_filled(Rect::from_min_size(pb.left_top(), Vec2::new(pb.width() * 0.62, pb.height())), 4.0, pal.accent);
    p.rect_stroke(pb, 4.0, Stroke::new(1.0, pal.frame), StrokeKind::Outside);
    small(p, pb.center(), Align2::CENTER_CENTER, "loading → rank estimate → solve → outputs", pal.text);
    small(p, pb.right_center() + Vec2::new(12.0, 0.0), Align2::LEFT_CENTER, "✖ Cancel   📜 Log", pal.weak);
}

fn draw_review(p: &egui::Painter, area: Rect, pal: &Palette) {
    // Corrected | raw | difference, then the profiles.
    let iw = ((area.width() * 0.56 - 24.0) / 3.0).clamp(80.0, 130.0);
    let ih = iw * 0.8;
    let y = area.top() + 22.0;
    let a = Rect::from_min_size(Pos2::new(area.left() + 4.0, y), Vec2::new(iw, ih));
    let b = a.translate(Vec2::new(iw + 10.0, 0.0));
    let d = b.translate(Vec2::new(iw + 10.0, 0.0));
    sample(p, a, 0.06, 8);
    sample(p, b, 0.55, 9);
    {
        // Difference: noise only, no structure.
        let pc = p.with_clip_rect(d);
        let cell = 5.0;
        for j in 0..(d.height() / cell).ceil() as i32 {
            for i in 0..(d.width() / cell).ceil() as i32 {
                let v = 0.5 + 0.55 * noise(i, j, 10);
                let g = (60.0 + 150.0 * v) as u8;
                pc.rect_filled(Rect::from_min_size(Pos2::new(d.left() + i as f32 * cell, d.top() + j as f32 * cell), Vec2::splat(cell + 0.5)), 0.0, Color32::from_rgb(g, g, (g as f32 * 1.1).min(255.0) as u8));
            }
        }
    }
    for (r, t) in [(a, "corrected"), (b, "raw"), (d, "difference")] {
        p.rect_stroke(r, 0.0, Stroke::new(1.0, pal.frame), StrokeKind::Outside);
        label(p, r.center_top() - Vec2::new(0.0, 5.0), Align2::CENTER_BOTTOM, t, pal.text);
    }
    // Profile region on the corrected image.
    let reg = Rect::from_center_size(a.center() - Vec2::new(0.12 * iw, -0.08 * ih), Vec2::new(0.26 * iw, 0.26 * ih));
    p.rect_stroke(reg, 0.0, Stroke::new(1.5, Color32::WHITE), StrokeKind::Middle);
    small(p, a.center_bottom() + Vec2::new(0.0, 4.0), Align2::CENTER_TOP, "white box: profile region", pal.weak);
    small(p, d.center_bottom() + Vec2::new(0.0, 4.0), Align2::CENTER_TOP, "should look like noise", pal.weak);

    // Result summary lines under the images.
    let sy = a.bottom() + 30.0;
    let rows = ["Number of materials: 2  (estimated)", "Reduced chi-square 1.04 — at the noise level", "0 warning(s) from mbirtorch"];
    for (i, s) in rows.iter().enumerate() {
        small(p, Pos2::new(area.left() + 8.0, sy + 15.0 * i as f32), Align2::LEFT_TOP, s, if i == 1 { pal.text } else { pal.weak });
    }

    // Profiles plot.
    let px = d.right() + 44.0;
    let plot = Rect::from_min_max(Pos2::new(px, area.top() + 22.0), Pos2::new((area.right() - 8.0).max(px + 120.0), area.bottom() - 26.0));
    axes(p, plot, "image index / TOF / wavelength", "mean", pal);
    label(p, plot.center_top() - Vec2::new(0.0, 5.0), Align2::CENTER_BOTTOM, "Profiles of the region", pal.text);
    let n = 34;
    for i in 0..n {
        let x = i as f32 / (n - 1) as f32;
        let v = spectrum(x, 0.45, 0.8) + 0.16 * noise(i, 3, 11);
        let c = Pos2::new(plot.left() + x * plot.width(), plot.bottom() - v.clamp(0.02, 0.98) * plot.height());
        let s = Stroke::new(1.3, pal.orange);
        p.line_segment([c + Vec2::new(-3.0, -3.0), c + Vec2::new(3.0, 3.0)], s);
        p.line_segment([c + Vec2::new(-3.0, 3.0), c + Vec2::new(3.0, -3.0)], s);
    }
    curve(p, plot, |x| spectrum(x, 0.45, 0.8), Stroke::new(2.0, pal.blue));
    for i in (0..n).step_by(3) {
        let x = i as f32 / (n - 1) as f32;
        p.circle_filled(Pos2::new(plot.left() + x * plot.width(), plot.bottom() - spectrum(x, 0.45, 0.8) * plot.height()), 2.5, pal.blue);
    }
    small(p, plot.left_top() + Vec2::new(10.0, 4.0), Align2::LEFT_TOP, "× uncorrected", pal.orange);
    small(p, plot.left_top() + Vec2::new(10.0, 18.0), Align2::LEFT_TOP, "● corrected", pal.blue);
}

fn draw_export(p: &egui::Painter, area: Rect, pal: &Palette) {
    // Corrected stack → folder with files.
    let s0 = Rect::from_min_size(Pos2::new(area.left() + 10.0, area.top() + 40.0), Vec2::new(100.0, 80.0));
    stack(p, s0, 4, pal, |p, r| sample(p, r, 0.06, 12));
    small(p, Pos2::new(s0.center().x + 10.0, s0.bottom() + 26.0), Align2::CENTER_TOP, "corrected stack", pal.weak);
    let ax = s0.right() + 34.0;
    p.arrow(Pos2::new(ax, s0.center().y + 10.0), Vec2::new(50.0, 0.0), Stroke::new(2.0, pal.text));
    small(p, Pos2::new(ax + 25.0, s0.center().y - 2.0), Align2::CENTER_BOTTOM, "Export", pal.text);

    let fx = ax + 66.0;
    let fw = (area.right() - fx - 10.0).clamp(240.0, 460.0);
    let folder = Rect::from_min_size(Pos2::new(fx, area.top() + 20.0), Vec2::new(fw, area.height() - 30.0));
    let tab = Rect::from_min_size(folder.left_top() - Vec2::new(0.0, 12.0), Vec2::new(90.0, 14.0));
    p.rect_filled(tab, 3.0, pal.frame_bg);
    p.rect_stroke(tab, 3.0, Stroke::new(1.0, pal.accent), StrokeKind::Outside);
    p.rect_filled(folder, 4.0, pal.frame_bg);
    p.rect_stroke(folder, 4.0, Stroke::new(1.0, pal.accent), StrokeKind::Outside);
    label(p, folder.left_top() + Vec2::new(10.0, 8.0), Align2::LEFT_TOP, "dehydrated_hydrated_<input folder>/", pal.text);
    let files = [
        ("image_0000.tif … image_NNNN.tif", "corrected images (float32)"),
        ("*_Spectra.txt", "TOF axis, copied"),
        ("correction_config.json", "parameters, mask, fit"),
        ("hsnt_report.json, hsnt_log.txt", "mbirtorch report and log"),
        ("hsnt_spectra.png, hsnt_maps.png", "diagnostics plots"),
        ("hsnt_dehydrated.h5", "maps and spectra"),
        ("hsnt_map_<i>.tif", "material maps"),
        ("mask.tif", "the mask, if any"),
    ];
    for (i, (f, d)) in files.iter().enumerate() {
        let y = 32.0 + 19.0 * i as f32;
        small(p, folder.left_top() + Vec2::new(20.0, y), Align2::LEFT_TOP, f, pal.text);
        small(p, folder.left_top() + Vec2::new(20.0 + (fw * 0.5).min(210.0), y), Align2::LEFT_TOP, d, pal.weak);
    }
}

fn draw_settings(p: &egui::Painter, area: Rect, pal: &Palette) {
    let doc = Rect::from_min_size(Pos2::new(area.left() + 20.0, area.top() + 14.0), Vec2::new(190.0, area.height() - 28.0));
    p.rect_filled(doc, 4.0, pal.frame_bg);
    p.rect_stroke(doc, 4.0, Stroke::new(1.0, pal.accent), StrokeKind::Outside);
    label(p, doc.center_top() + Vec2::new(0.0, 8.0), Align2::CENTER_TOP, "settings.h5", pal.text);
    let keys = [
        "correction: input type, materials,",
        "   spectra, dose, device, solver",
        "mask: shapes, brightness range,",
        "   mask image path",
        "physical axis: distance, offset",
        "display: colormap, contrast",
        "region: profile box",
    ];
    for (i, k) in keys.iter().enumerate() {
        small(p, doc.left_top() + Vec2::new(12.0, 32.0 + 17.0 * i as f32), Align2::LEFT_TOP, k, pal.weak);
    }
    small(p, doc.center_bottom() - Vec2::new(0.0, 22.0), Align2::CENTER_BOTTOM, "saved in step 5 (Export)", pal.text);
    small(p, doc.center_bottom() - Vec2::new(0.0, 8.0), Align2::CENTER_BOTTOM, "loaded in step 1 (Load data)", pal.text);
    let targets = [
        ("This window", "Load settings from HDF5…"),
        ("Workflow runner", "the same file drives batches of runs"),
        ("Command line", "--run --output DIR, no window"),
    ];
    let bw = (area.right() - doc.right() - 90.0).clamp(170.0, 300.0);
    for (i, (t, s)) in targets.iter().enumerate() {
        let r = Rect::from_min_size(Pos2::new(doc.right() + 70.0, area.top() + 26.0 + 68.0 * i as f32), Vec2::new(bw, 46.0));
        step_box(p, r, t, s, pal);
        let from = Pos2::new(doc.right() + 4.0, doc.center().y);
        let to = Pos2::new(r.left() - 4.0, r.center().y);
        p.arrow(from, to - from, Stroke::new(1.5, pal.weak));
    }
}
