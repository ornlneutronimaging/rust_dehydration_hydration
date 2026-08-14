//! Dehydration/Hydration Correction — native port of the VENUS
//! dehydration_hydration notebook: load a stack of TIFF images, denoise it
//! with the NMF dehydrate/rehydrate algorithm (mbirjax.hsnt), inspect the
//! result, and export the corrected stack.

use dehydration_hydration::app::DehydrationApp;
use dehydration_hydration::loader;
use std::path::PathBuf;

const USAGE: &str = "\
dehydration_hydration — NMF dehydration/hydration denoising of a TIFF stack

USAGE:
  dehydration_hydration [OPTIONS] [INPUT ...]

ARGS:
  INPUT   TIFF file(s) or a folder of TIFF images (subfolders are searched
          when the folder itself has none). When omitted, the data can be
          opened from within the application.

OPTIONS:
  -h, --help    Show this help

The correction reproduces the dehydration_hydration notebook:
mbirjax.hsnt.hyper_denoise — M. S. N. Chowdhury et al., \"Fast Hyperspectral
Neutron Tomography\", IEEE Trans. Comput. Imaging 11, 663-677 (2025).
";

fn main() -> eframe::Result<()> {
    let mut inputs: Vec<PathBuf> = Vec::new();
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with('-') => {
                eprintln!("Error: unknown option: {s}\n\n{USAGE}");
                std::process::exit(2);
            }
            _ => inputs.push(PathBuf::from(a)),
        }
    }

    // Expand folders to the image files they contain, so errors (missing
    // folder, no images) surface on stderr before the GUI opens.
    let mut files: Vec<PathBuf> = Vec::new();
    for input in inputs {
        if input.is_dir() {
            match loader::list_supported_in_dir(&input) {
                Ok(found) => files.extend(found),
                Err(e) => {
                    eprintln!("Error: {e:#}");
                    std::process::exit(1);
                }
            }
        } else {
            files.push(input);
        }
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1440.0, 900.0])
            .with_title("VENUS Dehydration / Hydration Correction"),
        ..Default::default()
    };

    eframe::run_native(
        "VENUS Dehydration / Hydration Correction",
        native_options,
        Box::new(move |cc| {
            // Saved light/dark preference, shared by all the VENUS rust
            // tools (dark when none is saved); the toolbar has a toggle.
            cc.egui_ctx.set_theme(dehydration_hydration::theme::load());
            let mut app = DehydrationApp::new();
            if !files.is_empty() {
                app.start_load(files, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
