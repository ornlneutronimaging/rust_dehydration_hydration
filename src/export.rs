//! Exporting the corrected stack as 32-bit float TIFF files, one per input
//! image, keeping the input file names — the notebook's export step.

use anyhow::{Context, Result};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

/// `<output>/<input-folder-name>_dehydration_hydration_corrected`, suffixed
/// `_1`, `_2`, … when it already exists (the notebook's
/// `make_or_increment_folder_name`). The folder is created.
pub fn make_export_folder(output_dir: &Path, input_dir_name: &str) -> Result<PathBuf> {
    let base = output_dir.join(format!("{input_dir_name}_dehydration_hydration_corrected"));
    let mut candidate = base.clone();
    let mut i = 0;
    while candidate.exists() {
        i += 1;
        candidate = PathBuf::from(format!("{}_{i}", base.display()));
    }
    std::fs::create_dir_all(&candidate)
        .with_context(|| format!("create {}", candidate.display()))?;
    Ok(candidate)
}

/// Write one frame as a grayscale 32-bit float TIFF. `undo_display_transpose`
/// restores the on-disk orientation for frames the loader transposed (TIFF
/// input — same convention as rust_roi_selector).
pub fn write_f32_tiff(path: &Path, frame: &Array2<f32>, undo_display_transpose: bool) -> Result<()> {
    use tiff::encoder::{colortype::Gray32Float, TiffEncoder};

    let data = if undo_display_transpose {
        frame.t().as_standard_layout().into_owned()
    } else {
        frame.as_standard_layout().into_owned()
    };
    let (h, w) = (data.nrows(), data.ncols());
    let file =
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(file))
        .with_context(|| format!("init TIFF encoder for {}", path.display()))?;
    enc.write_image::<Gray32Float>(w as u32, h as u32, data.as_slice().unwrap())
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Output file name: the input file's stem with a `.tif` extension.
pub fn output_name(input: &Path) -> String {
    let stem = input
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "image".to_owned());
    format!("{stem}.tif")
}

pub enum ExportMsg {
    Progress { done: usize, total: usize },
    Done(Result<PathBuf, String>),
}

/// Write `frames` (paired with `sources` for the names) into a fresh export
/// folder on a background thread.
pub fn start_export(
    output_dir: PathBuf,
    input_dir_name: String,
    frames: std::sync::Arc<Vec<Array2<f32>>>,
    sources: Vec<PathBuf>,
    undo_display_transpose: bool,
    ctx: egui::Context,
) -> Receiver<ExportMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let run = || -> Result<PathBuf> {
            let folder = make_export_folder(&output_dir, &input_dir_name)?;
            let total = frames.len();
            for (i, frame) in frames.iter().enumerate() {
                let name = sources
                    .get(i)
                    .map(|p| output_name(p))
                    .unwrap_or_else(|| format!("image_{i:05}.tif"));
                write_f32_tiff(&folder.join(name), frame, undo_display_transpose)?;
                let _ = tx.send(ExportMsg::Progress {
                    done: i + 1,
                    total,
                });
                ctx.request_repaint();
            }
            Ok(folder)
        };
        let _ = tx.send(ExportMsg::Done(run().map_err(|e| format!("{e:#}"))));
        ctx.request_repaint();
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dehydration_export_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn export_folder_increments_on_collision() {
        let dir = tmp_dir("incr");
        let a = make_export_folder(&dir, "Run_1234").unwrap();
        let b = make_export_folder(&dir, "Run_1234").unwrap();
        let c = make_export_folder(&dir, "Run_1234").unwrap();
        assert!(a.ends_with("Run_1234_dehydration_hydration_corrected"));
        assert!(b.ends_with("Run_1234_dehydration_hydration_corrected_1"));
        assert!(c.ends_with("Run_1234_dehydration_hydration_corrected_2"));
    }

    #[test]
    fn f32_tiff_roundtrips_through_the_loader() {
        let dir = tmp_dir("tiff");
        let path = dir.join("img.tif");
        let frame = Array2::from_shape_fn((3, 5), |(y, x)| (y * 5 + x) as f32 * 0.5);
        // Written with the transpose undone, the loader (which transposes
        // TIFFs on read) must round-trip to the in-memory orientation.
        write_f32_tiff(&path, &frame, true).unwrap();
        let stack = crate::loader::load_paths(&[path]).unwrap();
        assert_eq!((stack.height, stack.width), (3, 5));
        assert_eq!(stack.frames[0], frame);
    }

    #[test]
    fn output_name_forces_tif_extension() {
        assert_eq!(output_name(Path::new("/a/b/img_0001.tiff")), "img_0001.tif");
        assert_eq!(output_name(Path::new("/a/b/img_0001.tif")), "img_0001.tif");
    }
}
