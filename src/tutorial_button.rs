//! The "? Tutorial" button of the toolbars, shared by every VENUS rust tool.
//!
//! A blue pill with a bold white "?" badge, so the help stands out from the
//! plain toolbar buttons; darker and outlined while the tutorial is open.
//!
//! This module is deliberately self-contained (egui + the bundled
//! `fonts/DejaVuSans-Bold.ttf`) so it can be copied verbatim into the other
//! tools' crates, like `theme.rs` and `zoom.rs`. Nothing to call at
//! start-up: the bold face is registered the first time the button is drawn.

use egui::{Align, Color32, FontFamily, FontId, Stroke};

const FONT_NAME: &str = "TutorialButton-DejaVuSans-Bold";

/// A family of its own, so the button does not depend on how each tool
/// sets up its fonts.
fn bold_family() -> FontFamily {
    FontFamily::Name("tutorial_button_bold".into())
}

/// The bold family once egui knows it; the regular one on the very first
/// frame (a font added now is only usable from the next frame).
fn family(ctx: &egui::Context) -> FontFamily {
    let bold = bold_family();
    if ctx.fonts(|f| f.families().contains(&bold)) {
        return bold;
    }
    ctx.add_font(egui::epaint::text::FontInsert::new(
        FONT_NAME,
        egui::FontData::from_static(include_bytes!("../fonts/DejaVuSans-Bold.ttf")),
        vec![egui::epaint::text::InsertFontFamily {
            family: bold,
            priority: egui::epaint::text::FontPriority::Highest,
        }],
    ));
    ctx.request_repaint();
    FontFamily::Proportional
}

/// Draw the button. `label` is the word after the "?" ("Tutorial", "Help");
/// `open` says whether the tutorial is showing right now (`false` for a
/// tutorial the tool cannot track, e.g. one opened in the browser).
pub fn show(ui: &mut egui::Ui, label: &str, open: bool) -> egui::Response {
    let dark = ui.visuals().dark_mode;
    let fill = match (dark, open) {
        (true, false) => Color32::from_rgb(45, 105, 190),
        (true, true) => Color32::from_rgb(30, 75, 145),
        (false, false) => Color32::from_rgb(30, 100, 190),
        (false, true) => Color32::from_rgb(20, 70, 140),
    };
    let outline = if open {
        ui.visuals().strong_text_color()
    } else {
        fill
    };
    let family = family(ui.ctx());
    let format = |size: f32| egui::TextFormat {
        font_id: FontId::new(size, family.clone()),
        color: Color32::WHITE,
        valign: Align::Center,
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob::default();
    job.append("?", 0.0, format(16.0));
    job.append(label, 7.0, format(13.0));
    ui.scope(|ui| {
        ui.spacing_mut().button_padding = egui::vec2(12.0, 4.0);
        ui.add(
            egui::Button::new(job)
                .fill(fill)
                .stroke(Stroke::new(1.5, outline))
                .corner_radius(14.0),
        )
    })
    .inner
}
