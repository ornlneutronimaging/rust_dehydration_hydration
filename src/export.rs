//! Exporting the corrected stack as 32-bit float TIFF files (one per input
//! image, keeping the input file names) together with a provenance file
//! (`correction_config.json`) recording exactly how they were produced, and
//! the by-products of the mbirtorch run (its JSON report, the dehydrated
//! maps + spectra file, the PNG plots and the log), and a copy of the input
//! folder's `*_Spectra.txt` so the exported stack keeps its TOF axis. Also:
//! CSV export of the profile plots.
//!
//! [`export_corrected`] is the synchronous, GUI-free core (also used by the
//! headless `--run` mode); [`start_export`] wraps it on a background thread
//! for the egui app.

use crate::correction::CorrectionParams;
use crate::hsnt_cli::{Artifact, HsntReport};
use anyhow::{Context, Result};
use detector_orientation::Orientation;
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

/// Everything `correction_config.json` records about a corrected stack.
pub struct Provenance {
    pub input_folder: PathBuf,
    pub num_images: usize,
    pub image_width: usize,
    pub image_height: usize,
    pub params: CorrectionParams,
    /// Spatial binning factor of the run (1 = full resolution).
    pub bin: usize,
    pub elapsed_seconds: f64,
    /// What the mbirtorch run reported.
    pub report: HsntReport,
    /// Commit + branch of the mbirtorch checkout that ran.
    pub mbirtorch_commit: Option<String>,
    /// The pixel mask, when one restricted the solve.
    pub mask: Option<MaskInfo>,
}

/// What the provenance records about the mask.
pub struct MaskInfo {
    /// Pixels handed to the solver (of the binned stack for a preview).
    pub selected: usize,
    pub total: usize,
    /// How the mask was defined (see `MaskSpec::describe`).
    pub description: String,
}

/// Prefix of the export folder name, in front of the input folder's name.
pub const EXPORT_PREFIX: &str = "dehydrated_hydrated_";

/// `<output>/dehydrated_hydrated_<input-folder-name>`, suffixed `_1`, `_2`, …
/// when it already exists (the notebook's `make_or_increment_folder_name`).
/// The input folder's name is kept whole so the export is easy to match
/// back to its run. The folder is created.
pub fn make_export_folder(output_dir: &Path, input_dir_name: &str) -> Result<PathBuf> {
    let base = output_dir.join(format!("{EXPORT_PREFIX}{input_dir_name}"));
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

/// Write one frame as a grayscale 32-bit float TIFF. `orientation` is the
/// one the loader applied (`ImageStack::orientation`); it is undone here so
/// the file comes out in the on-disk orientation of the input (same
/// convention as rust_roi_selector).
pub fn write_f32_tiff(path: &Path, frame: &Array2<f32>, orientation: Orientation) -> Result<()> {
    use tiff::encoder::{colortype::Gray32Float, TiffEncoder};

    let data = orientation.undo_view(frame.view());
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

/// Name of the export folder for an input folder: its own name (`images`
/// when the path has none) behind [`EXPORT_PREFIX`].
pub fn input_dir_name(input_dir: &Path) -> String {
    input_dir
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|n| !n.is_empty())
        .unwrap_or_else(|| "images".to_owned())
}

/// Copy the input folder's `*_Spectra.txt` (the TOF axis of the stack) into
/// the export folder, keeping its name. Nothing to do when there is none.
/// Returns the copied file's path.
pub fn copy_spectra_file(input_dir: &Path, folder: &Path) -> Result<Option<PathBuf>> {
    let Some(src) = crate::spectra::find_spectra_file(input_dir) else {
        return Ok(None);
    };
    let name = src.file_name().expect("find_spectra_file returns files");
    let dst = folder.join(name);
    std::fs::copy(&src, &dst)
        .with_context(|| format!("copy {} to {}", src.display(), dst.display()))?;
    Ok(Some(dst))
}

/// Write the corrected stack + provenance file + the run's by-products +
/// the input folder's spectra file into a fresh export folder under
/// `output_dir`, reporting `(files_done, files_total)`. Returns the created
/// folder.
#[allow(clippy::too_many_arguments)]
pub fn export_corrected(
    output_dir: &Path,
    input_dir: &Path,
    frames: &[Array2<f32>],
    sources: &[PathBuf],
    orientation: Orientation,
    provenance: &Provenance,
    artifacts: &[Artifact],
    progress: &mut dyn FnMut(usize, usize),
) -> Result<PathBuf> {
    let folder = make_export_folder(output_dir, &input_dir_name(input_dir))?;
    let total = frames.len();
    for (i, frame) in frames.iter().enumerate() {
        let name = sources
            .get(i)
            .map(|p| output_name(p))
            .unwrap_or_else(|| format!("image_{i:05}.tif"));
        write_f32_tiff(&folder.join(name), frame, orientation)?;
        progress(i + 1, total);
    }
    let json = provenance_json(provenance);
    std::fs::write(folder.join("correction_config.json"), json)
        .with_context(|| format!("write provenance in {}", folder.display()))?;
    for a in artifacts {
        std::fs::write(folder.join(&a.name), &a.bytes)
            .with_context(|| format!("write {} in {}", a.name, folder.display()))?;
    }
    copy_spectra_file(input_dir, &folder)?;
    Ok(folder)
}

/// The provenance file contents.
pub fn provenance_json(p: &Provenance) -> String {
    let doc = serde_json::json!({
        "tool": "rust_dehydration_hydration",
        "tool_version": env!("CARGO_PKG_VERSION"),
        "tool_variant": "beta — mbirtorch hsnt",
        "algorithm": "mbirtorch.hsnt denoise: maximum-likelihood (Poisson) factorization X = W·H of the attenuation, dehydrate + rehydrate",
        "mbirtorch_checkout": crate::hsnt_cli::MBIRTORCH_CHECKOUT,
        "mbirtorch_branch": crate::hsnt_cli::MBIRTORCH_BRANCH,
        "mbirtorch_commit": p.mbirtorch_commit,
        "created_utc": iso8601_utc_now(),
        "input_folder": p.input_folder.display().to_string(),
        "num_images": p.num_images,
        "image_width": p.image_width,
        "image_height": p.image_height,
        "parameters": {
            "input_type": p.params.input_type.label(),
            "rank": p.params.rank.label(),
            "max_rank": p.params.max_rank,
            "spectra": p.params.spectra.label(),
            "dose": p.params.dose,
            "device": p.params.device.label(),
            "mode": p.params.mode.label(),
            "max_steps": p.params.max_steps,
            "rel_tol": p.params.rel_tol,
            "max_passes": p.params.max_passes,
            "compile": p.params.compile.label(),
            "cli_args": p.params.cli_args(),
        },
        "result": p.report.to_json(),
        "mask": p.mask.as_ref().map(|m| serde_json::json!({
            "pixels_selected": m.selected,
            "pixels_total": m.total,
            "fraction": m.selected as f64 / m.total.max(1) as f64,
            "definition": m.description,
            "file": "mask.tif",
            "outside_mask": "pixels outside the mask were not sent to mbirtorch and keep their raw values"
        })),
        "spatial_binning": p.bin,
        "elapsed_seconds": (p.elapsed_seconds * 10.0).round() / 10.0,
        "files": {
            "hsnt_report.json": "the CLI's full report (checks, memory plan, rank search, fit)",
            "hsnt_dehydrated.h5": "subspace_data (maps) + subspace_basis (spectra): the dehydrated form",
            "hsnt_spectra.png": "component spectra plot",
            "hsnt_maps.png": "component maps plot",
            "hsnt_log.txt": "the CLI's log",
            "hsnt_map_<i>.tif": "component map i as a float32 image (0 outside the mask)",
            "mask.tif": "the pixel mask (255 = sent to the solver), only when one was set"
        }
    });
    let mut text = serde_json::to_string_pretty(&doc).unwrap_or_else(|_| doc.to_string());
    text.push('\n');
    text
}


/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`, from the system clock only
/// (no chrono dependency; civil-from-days per Howard Hinnant).
pub fn iso8601_utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let tod = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

/// Gregorian calendar date from days since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // day of era [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// Write the profile plot's data as CSV. `tof_us`/`lambda_a` are optional
/// extra axis columns (must match the profile length when present).
pub fn write_profiles_csv(
    path: &Path,
    uncorrected: &[f64],
    corrected: &[f64],
    tof_us: Option<&[f64]>,
    lambda_a: Option<&[f64]>,
) -> Result<()> {
    use std::io::Write;

    let mut out = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?,
    );
    write!(out, "image_index")?;
    if tof_us.is_some() {
        write!(out, ",tof_us")?;
    }
    if lambda_a.is_some() {
        write!(out, ",lambda_angstroms")?;
    }
    writeln!(out, ",uncorrected,corrected")?;
    for i in 0..uncorrected.len().min(corrected.len()) {
        write!(out, "{i}")?;
        if let Some(t) = tof_us {
            write!(out, ",{}", t.get(i).copied().unwrap_or(f64::NAN))?;
        }
        if let Some(l) = lambda_a {
            write!(out, ",{}", l.get(i).copied().unwrap_or(f64::NAN))?;
        }
        writeln!(out, ",{},{}", uncorrected[i], corrected[i])?;
    }
    Ok(())
}

pub enum ExportMsg {
    Progress { done: usize, total: usize },
    Done(Result<PathBuf, String>),
}

/// Run [`export_corrected`] on a background thread for the GUI.
#[allow(clippy::too_many_arguments)]
pub fn start_export(
    output_dir: PathBuf,
    input_dir: PathBuf,
    frames: std::sync::Arc<Vec<Array2<f32>>>,
    sources: Vec<PathBuf>,
    orientation: Orientation,
    provenance: Provenance,
    artifacts: Vec<Artifact>,
    ctx: egui::Context,
) -> Receiver<ExportMsg> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let progress_tx = tx.clone();
        let progress_ctx = ctx.clone();
        let mut progress = move |done: usize, total: usize| {
            let _ = progress_tx.send(ExportMsg::Progress { done, total });
            progress_ctx.request_repaint();
        };
        let result = export_corrected(
            &output_dir,
            &input_dir,
            &frames,
            &sources,
            orientation,
            &provenance,
            &artifacts,
            &mut progress,
        );
        let _ = tx.send(ExportMsg::Done(result.map_err(|e| format!("{e:#}"))));
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

    fn provenance() -> Provenance {
        Provenance {
            input_folder: PathBuf::from("/data/Run_1"),
            num_images: 2,
            image_width: 5,
            image_height: 3,
            params: CorrectionParams::default(),
            bin: 1,
            elapsed_seconds: 1.5,
            report: HsntReport {
                rank: Some(2),
                rank_note: "rank 2 given".to_owned(),
                reduced_chi2: Some(1.05),
                ..HsntReport::default()
            },
            mbirtorch_commit: Some("edb0bcb (hsnt)".to_owned()),
            mask: Some(MaskInfo { selected: 10, total: 15, description: "1 include rectangle(s)".to_owned() }),
        }
    }

    #[test]
    fn export_folder_increments_on_collision() {
        let dir = tmp_dir("incr");
        let a = make_export_folder(&dir, "Run_1234").unwrap();
        let b = make_export_folder(&dir, "Run_1234").unwrap();
        let c = make_export_folder(&dir, "Run_1234").unwrap();
        assert!(a.ends_with("dehydrated_hydrated_Run_1234"));
        assert!(b.ends_with("dehydrated_hydrated_Run_1234_1"));
        assert!(c.ends_with("dehydrated_hydrated_Run_1234_2"));
    }

    #[test]
    fn input_dir_name_keeps_the_folder_name() {
        assert_eq!(input_dir_name(Path::new("/data/IPTS-1/Run_1234")), "Run_1234");
        assert_eq!(input_dir_name(Path::new("/data/IPTS-1/Run_1234/")), "Run_1234");
        assert_eq!(input_dir_name(Path::new("")), "images");
    }

    #[test]
    fn f32_tiff_roundtrips_through_the_loader() {
        let dir = tmp_dir("tiff");
        let path = dir.join("img.tif");
        let frame = Array2::from_shape_fn((3, 5), |(y, x)| (y * 5 + x) as f32 * 0.5);
        // Written with the orientation undone, the loader (which re-orients
        // TIFFs on read) must round-trip to the in-memory orientation, for
        // every detector.
        use crate::loader::{Detector, Selection};
        for (d, o) in [
            (Detector::Timepix, Orientation::Transpose),
            (Detector::Ccd, Orientation::Rotate180),
            (Detector::Unknown, Orientation::Identity),
        ] {
            write_f32_tiff(&path, &frame, o).unwrap();
            let sel = Selection { manual: Some(d), ..Default::default() };
            let stack =
                crate::loader::load_paths_with_progress(&[path.clone()], sel, |_, _| {}).unwrap();
            assert_eq!(stack.orientation, o);
            assert_eq!((stack.height, stack.width), (3, 5));
            assert_eq!(stack.frames[0], frame, "{d:?}");
        }
    }

    #[test]
    fn output_name_forces_tif_extension() {
        assert_eq!(output_name(Path::new("/a/b/img_0001.tiff")), "img_0001.tif");
        assert_eq!(output_name(Path::new("/a/b/img_0001.tif")), "img_0001.tif");
    }

    #[test]
    fn export_writes_images_and_provenance() {
        let dir = tmp_dir("full");
        let input = dir.join("Run_1");
        std::fs::create_dir_all(&input).unwrap();
        std::fs::write(input.join("Run_1_Spectra.txt"), "1e-6,1\n2e-6,2\n").unwrap();
        let frames = vec![
            Array2::from_elem((3, 5), 1.0f32),
            Array2::from_elem((3, 5), 2.0f32),
        ];
        let sources = vec![input.join("a_0000.tiff"), input.join("a_0001.tiff")];
        let mut ticks = Vec::new();
        let folder = export_corrected(
            &dir,
            &input,
            &frames,
            &sources,
            Orientation::Transpose,
            &provenance(),
            &[Artifact { name: "hsnt_report.json".to_owned(), bytes: b"{}".to_vec() }],
            &mut |d, t| ticks.push((d, t)),
        )
        .unwrap();
        assert!(folder.ends_with("dehydrated_hydrated_Run_1"), "{}", folder.display());
        assert!(folder.join("a_0000.tif").is_file());
        assert!(folder.join("a_0001.tif").is_file());
        assert!(folder.join("hsnt_report.json").is_file());
        // The input folder's spectra file travels with the corrected stack.
        assert_eq!(
            std::fs::read_to_string(folder.join("Run_1_Spectra.txt")).unwrap(),
            "1e-6,1\n2e-6,2\n"
        );
        assert_eq!(ticks, vec![(1, 2), (2, 2)]);
        let json = std::fs::read_to_string(folder.join("correction_config.json")).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(doc["parameters"]["rank"], "auto");
        assert_eq!(doc["parameters"]["input_type"], "transmission");
        assert_eq!(doc["result"]["rank"], 2);
        assert_eq!(doc["result"]["reduced_chi2"], 1.05);
        assert_eq!(doc["mbirtorch_commit"], "edb0bcb (hsnt)");
        assert_eq!(doc["mask"]["pixels_selected"], 10);
        assert_eq!(doc["input_folder"], "/data/Run_1");
    }

    #[test]
    fn export_without_spectra_file_is_fine() {
        let dir = tmp_dir("nospectra");
        let input = dir.join("Run_2");
        std::fs::create_dir_all(&input).unwrap();
        let frames = vec![Array2::from_elem((2, 2), 1.0f32)];
        let folder = export_corrected(
            &dir,
            &input,
            &frames,
            &[input.join("b.tif")],
            Orientation::Identity,
            &provenance(),
            &[],
            &mut |_, _| {},
        )
        .unwrap();
        assert!(folder.join("b.tif").is_file());
        assert!(std::fs::read_dir(&folder)
            .unwrap()
            .flatten()
            .all(|e| !e.file_name().to_string_lossy().ends_with("_Spectra.txt")));
    }

    #[test]
    fn timestamp_looks_like_iso8601() {
        let t = iso8601_utc_now();
        assert_eq!(t.len(), 20, "{t}");
        assert!(t.ends_with('Z'));
        assert_eq!(&t[4..5], "-");
        assert_eq!(&t[10..11], "T");
        // Sanity: the year is in a plausible range.
        let year: i32 = t[0..4].parse().unwrap();
        assert!((2026..2100).contains(&year), "{t}");
    }

    #[test]
    fn civil_from_days_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1)); // leap-year checks
        assert_eq!(civil_from_days(19_723 + 59), (2024, 2, 29));
    }

    #[test]
    fn profiles_csv_includes_optional_axes() {
        let dir = tmp_dir("csv");
        let path = dir.join("profiles.csv");
        write_profiles_csv(
            &path,
            &[1.0, 2.0],
            &[0.9, 1.9],
            Some(&[6.08, 11.2]),
            Some(&[0.001, 0.002]),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.starts_with("image_index,tof_us,lambda_angstroms,uncorrected,corrected\n"));
        assert!(text.contains("0,6.08,0.001,1,0.9"));

        // Without axes: only three columns.
        let path2 = dir.join("plain.csv");
        write_profiles_csv(&path2, &[1.0], &[2.0], None, None).unwrap();
        assert!(std::fs::read_to_string(&path2)
            .unwrap()
            .starts_with("image_index,uncorrected,corrected\n"));
    }
}
