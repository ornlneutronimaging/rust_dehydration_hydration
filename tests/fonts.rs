//! Every non-ASCII glyph the UI shows must exist in the proportional font
//! stack — otherwise egui draws the "missing glyph" square (◻) instead.

use dehydration_hydration::theme;

/// Arrows and subscripts from the labels, plus every toolbar/button icon.
const UI_GLYPHS: &str = "→₁₀×µÅλ−≈·—–…☀🌙📁📂🕒🔍⚙💾✖▶⚡⏳📄ℹ";

/// egui's replacement character (drawn for glyphs no font has).
const REPLACEMENT: char = '◻';

/// The atlas rectangle (texture min/max) a single character lays out to.
fn uv_rect(ctx: &egui::Context, c: char) -> ([u16; 2], [u16; 2]) {
    let font_id = egui::FontId::proportional(14.0);
    ctx.fonts_mut(|f| {
        let galley = f.layout_no_wrap(c.to_string(), font_id, egui::Color32::WHITE);
        let uv = galley.rows[0].glyphs[0].uv_rect;
        (uv.min, uv.max)
    })
}

/// Characters that would be drawn as the replacement square.
fn missing(ctx: &egui::Context) -> Vec<char> {
    let _ = ctx.run_ui(Default::default(), |_| {});
    let square = uv_rect(ctx, REPLACEMENT);
    UI_GLYPHS
        .chars()
        .filter(|&c| uv_rect(ctx, c) == square)
        .collect()
}

#[test]
fn default_fonts_lack_the_arrow() {
    let ctx = egui::Context::default();
    let m = missing(&ctx);
    assert!(m.contains(&'→'), "missing with the default fonts: {m:?}");
    // The emoji icons have always rendered — the arrow is the odd one out.
    assert!(!m.contains(&'💾'), "missing with the default fonts: {m:?}");
}

#[test]
fn installed_fonts_cover_every_ui_glyph() {
    let ctx = egui::Context::default();
    theme::install_fonts(&ctx);
    assert_eq!(missing(&ctx), Vec::<char>::new());
}
