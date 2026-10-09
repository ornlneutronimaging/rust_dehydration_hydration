//! Run number → NeXus file → image stack of that run (same lookup as
//! rust_tiff_viewer, `run_lookup.rs` there).
//!
//! Mirrors what `/SNS/VENUS/shared/autoreduce/reduce_VENUS.py` and the
//! `find_run.sh` helper do:
//!
//! - NeXus file: `/SNS/VENUS/<IPTS>/nexus/VENUS_<run>.nxs.h5` — the IPTS is
//!   found by scanning every `/SNS/VENUS/IPTS-*` folder.
//! - detector: the `BL10:Exp:Det` DASlog (e.g. `MCP TPX1`, `Andor CCD
//!   iKon-XL`), which also decides the orientation the frames are loaded
//!   with (see [`crate::loader`]).
//! - detector offset: the `BL10:Det:TH:DSPT1:TIDelay` DASlog
//!   (`average_value`, in µs), applied to the profile plots' TOF axis.
//! - image folder: `/SNS/VENUS/<IPTS>/<folder_path>` where `folder_path` is
//!   the last `BL10:Exp:IM:ImageFilePath` DASlog value (fallback
//!   `BL10:Exp:IM:ConfigTpxFilePath`).
//! - **Timepix** runs: that folder holds the raw `.fits` frames of the run
//!   alone — which this program cannot read — and the autoreduce
//!   correction writes the `.tif` frames to
//!   `/SNS/VENUS/<IPTS>/shared/autoreduce/<folder_path>`.
//! - **other detectors** (CCD): the folder is shared by a whole series of
//!   runs, one TIFF per run named `*_Run_<run>_*.tiff`; the run's own
//!   image(s) are picked by name.

use crate::loader::SUPPORTED_EXTENSIONS;
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub const VENUS_ROOT: &str = "/SNS/VENUS";

/// A loadable stack (folder of images or a single image file) and its size.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub path: PathBuf,
    /// Image files this program can load (`.tif`/`.tiff`/`.npy`) directly
    /// inside a folder; 1 for a single file.
    pub n_images: usize,
    /// Raw `.fits` frames inside a folder (Timepix raw data), which cannot
    /// be loaded here — shown so the user understands why "Raw" is
    /// unavailable.
    pub n_fits: usize,
}

impl Stack {
    /// Whether there is something to load.
    pub fn is_loadable(&self) -> bool {
        self.n_images > 0
    }

    /// "770 files" / "file" / "312 .fits frames (not loadable)" for status
    /// messages and buttons.
    pub fn size_text(&self) -> String {
        if !self.path.is_dir() {
            "file".to_owned()
        } else if self.n_images > 0 {
            format!("{} files", self.n_images)
        } else if self.n_fits > 0 {
            format!("{} .fits frames, not loadable here", self.n_fits)
        } else {
            "no image".to_owned()
        }
    }

    /// The files to hand to the loader: the file itself, or every supported
    /// image of the folder (see [`crate::loader::list_supported_in_dir`]).
    pub fn files(&self) -> Result<Vec<PathBuf>> {
        if self.path.is_dir() {
            crate::loader::list_supported_in_dir(&self.path)
        } else {
            Ok(vec![self.path.clone()])
        }
    }
}

#[derive(Clone, Debug)]
pub struct RunInfo {
    pub run: u32,
    /// e.g. "IPTS-36967"
    pub ipts: String,
    pub nexus: PathBuf,
    /// `BL10:Exp:Det` value, when the NeXus file has it.
    pub detector: Option<String>,
    /// Detector offset in µs (`BL10:Det:TH:DSPT1:TIDelay/average_value`),
    /// when the NeXus file has it.
    pub offset_us: Option<f64>,
    /// Image folder relative to the IPTS root (whitespace-stripped DASlog
    /// value), e.g. "images/tpx1/raw/radiography/…/…_Run_23640_…_0".
    pub folder_path: String,
    /// `/SNS/VENUS/<IPTS>/<folder_path>`
    pub image_dir: PathBuf,
    /// The raw data: the image folder itself for Timepix runs; for other
    /// detectors the run's own image(s) inside it (or the folder when none is
    /// named after the run). Empty when nothing exists on disk.
    pub raw: Vec<Stack>,
    /// `/SNS/VENUS/<IPTS>/shared/autoreduce/<folder_path>` (Timepix runs).
    pub autoreduce_dir: PathBuf,
    /// The autoreduce stack, when its folder exists.
    pub autoreduce: Option<Stack>,
}

impl RunInfo {
    /// Timepix detectors ("MCP TPX1", "MCP TPX", "MCP TPX3", …): the only
    /// ones with an autoreduce version of the data.
    pub fn is_timepix(&self) -> bool {
        self.detector.as_deref().is_some_and(is_timepix_name)
    }

    pub fn detector_text(&self) -> &str {
        self.detector.as_deref().unwrap_or("unknown detector")
    }

    /// Whether any raw stack can be loaded.
    pub fn raw_loadable(&self) -> bool {
        self.raw.iter().any(Stack::is_loadable)
    }

    /// Whether the autoreduce stack exists and can be loaded.
    pub fn autoreduce_loadable(&self) -> bool {
        self.autoreduce.as_ref().is_some_and(Stack::is_loadable)
    }

    /// Every file of the raw data (all raw stacks, in order).
    pub fn raw_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for s in self.raw.iter().filter(|s| s.is_loadable()) {
            files.extend(s.files()?);
        }
        Ok(files)
    }

    /// Every file of the autoreduce data.
    pub fn autoreduce_files(&self) -> Result<Vec<PathBuf>> {
        match self.autoreduce.as_ref().filter(|s| s.is_loadable()) {
            Some(s) => s.files(),
            None => Ok(Vec::new()),
        }
    }

    /// What to load without asking: the autoreduce TIFFs of a Timepix run
    /// (the raw data is `.fits`), the raw image(s) otherwise — falling back
    /// to whichever is loadable. `Err` when nothing is.
    pub fn default_files(&self) -> Result<(Vec<PathBuf>, &'static str)> {
        let prefer_auto = self.is_timepix();
        let order: [(&'static str, bool); 2] = if prefer_auto {
            [("autoreduce", true), ("raw", false)]
        } else {
            [("raw", false), ("autoreduce", true)]
        };
        for (name, auto) in order {
            let files = if auto { self.autoreduce_files()? } else { self.raw_files()? };
            if !files.is_empty() {
                return Ok((files, name));
            }
        }
        bail!(
            "run {}: no loadable image found (raw: {}; autoreduce: {})",
            self.run,
            self.image_dir.display(),
            self.autoreduce_dir.display()
        )
    }
}

pub fn is_timepix_name(detector: &str) -> bool {
    let d = detector.to_ascii_uppercase();
    d.contains("TPX") || d.contains("TIMEPIX")
}

/// Scan `/SNS/VENUS/IPTS-*/nexus/` for `VENUS_<run>.nxs.h5`.
pub fn find_nexus(run: u32) -> Result<(String, PathBuf)> {
    let file_name = format!("VENUS_{run}.nxs.h5");
    let root = Path::new(VENUS_ROOT);
    let entries = std::fs::read_dir(root).with_context(|| format!("cannot read {VENUS_ROOT}"))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if !name.starts_with("IPTS-") {
            continue;
        }
        let candidate = entry.path().join("nexus").join(&file_name);
        if candidate.is_file() {
            return Ok((name, candidate));
        }
    }
    bail!("run {run}: {file_name} not found under any {VENUS_ROOT}/IPTS-*/nexus/ folder")
}

/// Locate the run and find its raw (and, for Timepix, autoreduce) stacks.
pub fn resolve(run: u32) -> Result<RunInfo> {
    let (ipts, nexus) = find_nexus(run)?;
    let Metadata { detector, folder_path, offset_us } = read_metadata(&nexus)?;
    let ipts_dir = Path::new(VENUS_ROOT).join(&ipts);
    let image_dir = ipts_dir.join(&folder_path);
    let autoreduce_dir = ipts_dir.join("shared/autoreduce").join(&folder_path);
    let timepix = detector.as_deref().is_some_and(is_timepix_name);

    let raw = if timepix {
        stack_at(&image_dir).into_iter().collect()
    } else {
        run_images_in(&image_dir, run)
    };
    let autoreduce = if timepix { stack_at(&autoreduce_dir) } else { None };

    Ok(RunInfo {
        run,
        ipts,
        nexus,
        detector,
        offset_us,
        folder_path,
        image_dir,
        raw,
        autoreduce_dir,
        autoreduce,
    })
}

/// The folder as a stack when it exists (even if empty, so the caller can
/// tell "folder exists but no images yet" from "no folder"), or the file as a
/// one-image stack.
fn stack_at(path: &Path) -> Option<Stack> {
    if path.is_dir() {
        let (n_images, n_fits) = count_images(path);
        Some(Stack { path: path.to_path_buf(), n_images, n_fits })
    } else if path.is_file() {
        Some(Stack { path: path.to_path_buf(), n_images: 1, n_fits: 0 })
    } else {
        None
    }
}

/// The images of a non-Timepix run: every loadable image file (or folder of
/// images) in `dir` whose name carries `Run_<run>_` / `Run_<run>.`, sorted
/// by name; the folder itself when none does but it holds images.
fn run_images_in(dir: &Path, run: u32) -> Vec<Stack> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<Stack> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .is_some_and(|n| names_run(&n.to_string_lossy(), run))
        })
        .filter(|p| p.is_dir() || is_loadable_file(p))
        .filter_map(|p| stack_at(&p))
        .collect();
    found.sort_by(|a, b| a.path.cmp(&b.path));
    if found.is_empty() && count_images(dir).0 > 0 {
        found.extend(stack_at(dir));
    }
    found
}

/// Whether a file/folder name belongs to `run` (`…_Run_23640_…`,
/// `…_run_23640.tiff`), without matching a longer run number.
pub fn names_run(name: &str, run: u32) -> bool {
    let name = name.to_ascii_lowercase();
    let key = format!("run_{run}");
    let mut from = 0;
    while let Some(pos) = name[from..].find(&key) {
        let end = from + pos + key.len();
        match name[end..].chars().next() {
            None | Some('_') | Some('.') | Some('-') | Some(' ') => return true,
            _ => from = end,
        }
    }
    false
}

/// The run number carried by a file or folder name (`…_Run_23640_…`,
/// `run_24671.tiff`), if any: the digits after `run_` (any case) up to a
/// `_`, `.`, `-`, a space or the end of the name. The first such run wins.
pub fn run_number_in(name: &str) -> Option<u32> {
    let lower = name.to_ascii_lowercase();
    let mut from = 0;
    while let Some(pos) = lower[from..].find("run_") {
        let start = from + pos + "run_".len();
        let digits = lower[start..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .count();
        let end = start + digits;
        let terminated = matches!(
            lower[end..].chars().next(),
            None | Some('_') | Some('.') | Some('-') | Some(' ')
        );
        if digits > 0 && terminated && let Ok(run) = lower[start..end].parse::<u32>() {
            return Some(run);
        }
        from = start;
    }
    None
}

/// The run an image path belongs to: from its file name (VENUS names every
/// frame `…_Run_<run>_…`), else from its folder's name (a Timepix autoreduce
/// folder is named after its run too).
pub fn run_number_of(path: &Path) -> Option<u32> {
    let name_of = |p: &Path| p.file_name().and_then(|n| run_number_in(&n.to_string_lossy()));
    name_of(path).or_else(|| path.parent().and_then(name_of))
}

/// The detector offset recorded with a run, for a stack that was loaded by
/// hand (see [`run_number_of`]): the run's NeXus file and its
/// [`OFFSET_LOG`] value in µs — `None` when the file has no such log.
pub fn run_offset(run: u32) -> Result<(PathBuf, Option<f64>)> {
    let (_ipts, nexus) = find_nexus(run)?;
    let file = hdf5::File::open(&nexus)
        .with_context(|| format!("cannot open {} as HDF5", nexus.display()))?;
    let offset_us = read_offset_us(&file).ok().flatten();
    Ok((nexus, offset_us))
}

fn ext_of(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase())
}

fn is_loadable_file(path: &Path) -> bool {
    path.is_file() && ext_of(path).is_some_and(|e| SUPPORTED_EXTENSIONS.contains(&e.as_str()))
}

fn is_fits_file(path: &Path) -> bool {
    path.is_file() && ext_of(path).is_some_and(|e| e == "fits")
}

/// Number of (loadable images, `.fits` frames) directly inside `dir`
/// (0 if missing).
pub fn count_images(dir: &Path) -> (usize, usize) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (0, 0);
    };
    let mut n_images = 0;
    let mut n_fits = 0;
    for entry in entries.flatten() {
        let p = entry.path();
        if is_loadable_file(&p) {
            n_images += 1;
        } else if is_fits_file(&p) {
            n_fits += 1;
        }
    }
    (n_images, n_fits)
}

// ----- NeXus reading (HDF5) --------------------------------------------------

/// DASlog for the detector offset (TOF delay), `average_value` in µs.
pub const OFFSET_LOG: &str = "BL10:Det:TH:DSPT1:TIDelay";

/// What [`resolve`] needs from the NeXus DASlogs.
struct Metadata {
    detector: Option<String>,
    folder_path: String,
    offset_us: Option<f64>,
}

fn read_metadata(nexus: &Path) -> Result<Metadata> {
    let file = hdf5::File::open(nexus)
        .with_context(|| format!("cannot open {} as HDF5", nexus.display()))?;

    let detector = read_strings(&file, "entry/DASlogs/BL10:Exp:Det/value_strings")
        .ok()
        .and_then(|v| v.into_iter().next())
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty());

    // Same field priority as reduce_VENUS.py: ImageFilePath (last entry),
    // then ConfigTpxFilePath (first entry).
    let folder_path = match read_strings(&file, "entry/DASlogs/BL10:Exp:IM:ImageFilePath/value") {
        Ok(v) if !v.is_empty() => v.last().unwrap().clone(),
        _ => read_strings(&file, "entry/DASlogs/BL10:Exp:IM:ConfigTpxFilePath/value")
            .with_context(|| {
                format!(
                    "{}: neither BL10:Exp:IM:ImageFilePath nor ConfigTpxFilePath found",
                    nexus.display()
                )
            })?
            .into_iter()
            .next()
            .with_context(|| format!("{}: BL10:Exp:IM:ConfigTpxFilePath is empty", nexus.display()))?,
    };
    let folder_path = folder_path.trim().trim_matches('/').to_owned();
    if folder_path.is_empty() {
        bail!("{}: the image folder DASlog is empty", nexus.display());
    }

    // Optional: an older file without the log just keeps the current offset.
    let offset_us = read_offset_us(&file).ok().flatten();

    Ok(Metadata { detector, folder_path, offset_us })
}

/// `entry/DASlogs/<OFFSET_LOG>/average_value` converted to µs from its
/// `units` attribute (`us` expected; `ns`, `ms` and `s` are converted, any
/// other unit is taken as µs). `None` for a missing, empty or non-finite
/// value.
fn read_offset_us(file: &hdf5::File) -> Result<Option<f64>> {
    let path = format!("entry/DASlogs/{OFFSET_LOG}/average_value");
    let ds = file
        .dataset(&path)
        .with_context(|| format!("dataset {path} not found"))?;
    let values = ds.read_raw::<f64>()?;
    let Some(&v) = values.first() else {
        return Ok(None);
    };
    if !v.is_finite() {
        return Ok(None);
    }
    let units = attr_string(&ds, "units").unwrap_or_default();
    Ok(Some(v * unit_to_us(&units)))
}

/// A scalar string attribute of a dataset, whatever its HDF5 string flavour.
fn attr_string(ds: &hdf5::Dataset, name: &str) -> Option<String> {
    use hdf5::types::TypeDescriptor::*;
    use hdf5::types::{FixedAscii, FixedUnicode, VarLenAscii, VarLenUnicode};

    let attr = ds.attr(name).ok()?;
    let td = attr.dtype().ok()?.to_descriptor().ok()?;
    let s = match td {
        VarLenAscii => attr.read_scalar::<VarLenAscii>().ok()?.to_string(),
        VarLenUnicode => attr.read_scalar::<VarLenUnicode>().ok()?.to_string(),
        FixedAscii(n) if n <= MAX_FIXED_STR => attr.read_scalar::<FixedAscii<MAX_FIXED_STR>>().ok()?.to_string(),
        FixedUnicode(n) if n <= MAX_FIXED_STR => {
            attr.read_scalar::<FixedUnicode<MAX_FIXED_STR>>().ok()?.to_string()
        }
        _ => return None,
    };
    Some(s)
}

/// Factor turning a value in `units` into microseconds.
pub fn unit_to_us(units: &str) -> f64 {
    match units.trim().to_ascii_lowercase().as_str() {
        "ns" | "nanosecond" | "nanoseconds" => 1e-3,
        "ms" | "millisecond" | "milliseconds" => 1e3,
        "s" | "sec" | "second" | "seconds" => 1e6,
        _ => 1.0,
    }
}

/// Fixed-length strings longer than this are unsupported (HDF5 only converts
/// fixed → fixed strings, so a fixed-size read buffer is required).
const MAX_FIXED_STR: usize = 4096;

/// Read a (possibly multi-dimensional) string dataset as a flat Vec<String>.
fn read_strings(file: &hdf5::File, path: &str) -> Result<Vec<String>> {
    use hdf5::types::TypeDescriptor::*;

    let ds = file
        .dataset(path)
        .with_context(|| format!("dataset {path} not found"))?;
    let td = ds.dtype()?.to_descriptor()?;
    let strings = match td {
        VarLenAscii => ds
            .read_raw::<hdf5::types::VarLenAscii>()?
            .iter()
            .map(|s| s.to_string())
            .collect(),
        VarLenUnicode => ds
            .read_raw::<hdf5::types::VarLenUnicode>()?
            .iter()
            .map(|s| s.to_string())
            .collect(),
        FixedAscii(n) if n <= MAX_FIXED_STR => ds
            .read_raw::<hdf5::types::FixedAscii<MAX_FIXED_STR>>()?
            .iter()
            .map(|s| s.to_string())
            .collect(),
        FixedUnicode(n) if n <= MAX_FIXED_STR => ds
            .read_raw::<hdf5::types::FixedUnicode<MAX_FIXED_STR>>()?
            .iter()
            .map(|s| s.to_string())
            .collect(),
        other => bail!("{path}: not a readable string dataset ({other:?})"),
    };
    Ok(strings)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "dehydration_hydration_run_{name}_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn timepix_names() {
        assert!(is_timepix_name("MCP TPX1"));
        assert!(is_timepix_name("MCP TPX"));
        assert!(is_timepix_name("MCP TPX3"));
        assert!(is_timepix_name("Timepix3"));
        assert!(!is_timepix_name("Andor CCD iKon-XL"));
    }

    #[test]
    fn offset_units() {
        assert_eq!(unit_to_us("us"), 1.0);
        assert_eq!(unit_to_us(" US "), 1.0);
        assert_eq!(unit_to_us(""), 1.0);
        assert_eq!(unit_to_us("ms"), 1e3);
        assert_eq!(unit_to_us("s"), 1e6);
        assert_eq!(unit_to_us("ns"), 1e-3);
    }

    #[test]
    fn run_name_matching_is_exact_on_the_number() {
        assert!(names_run("20260320_Run_15805_Compass_CT_Ang_0_000_1.tiff", 15805));
        assert!(names_run("20260622_Run_24671_DF_dc_2.tiff", 24671));
        assert!(names_run("run_24671.tiff", 24671));
        assert!(!names_run("20260320_Run_158051_Compass.tiff", 15805));
        assert!(!names_run("20260320_Run_1580_Compass.tiff", 15805));
        // A shorter run number embedded in a longer one still needs the
        // exact `Run_<run>_` somewhere.
        assert!(names_run("Run_158051_x_Run_15805_y.tiff", 15805));
    }

    #[test]
    fn run_number_is_read_out_of_a_name() {
        assert_eq!(
            run_number_in("20260613_Run_23640_LF99D_Rnd2_Coarsen_0_416C_0_000AngsMin_0_770_00000.tif"),
            Some(23640)
        );
        assert_eq!(run_number_in("20260320_Run_15805_Compass_CT_Ang_0_000_1.tiff"), Some(15805));
        assert_eq!(run_number_in("run_24671.tiff"), Some(24671));
        assert_eq!(run_number_in("Run_24671"), Some(24671));
        assert_eq!(run_number_in("Run_24671-ob.tif"), Some(24671));
        // A name without a run (a date is not one), or with a malformed one.
        assert_eq!(run_number_in("20260430_BPR_RT_0_464C_0_000AngsMin"), None);
        assert_eq!(run_number_in("run_abc_1.tif"), None);
        assert_eq!(run_number_in("Run_123x_1.tif"), None);
        // The first well-formed run wins; a malformed one is skipped.
        assert_eq!(run_number_in("Run_123x_Run_456_1.tif"), Some(456));
        assert_eq!(run_number_in("dehydrated_hydrated_Run_23640"), Some(23640));

        // From the path: the file name first, then the folder.
        assert_eq!(run_number_of(Path::new("/a/Run_1/x_Run_2_0.tif")), Some(2));
        assert_eq!(run_number_of(Path::new("/a/Run_1/x_00000.tif")), Some(1));
        assert_eq!(run_number_of(Path::new("/a/b/x_00000.tif")), None);
    }

    /// The offset lookup against real instrument data; skipped when
    /// /SNS/VENUS is not mounted.
    #[test]
    fn run_offset_real_run() {
        if !Path::new(VENUS_ROOT).is_dir() {
            eprintln!("skipping: {VENUS_ROOT} not mounted");
            return;
        }
        let (nexus, offset) = run_offset(23640).expect("offset of run 23640");
        assert!(nexus.ends_with("VENUS_23640.nxs.h5"), "{}", nexus.display());
        assert!(offset.is_some_and(|v| v.is_finite()), "{offset:?}");
        assert!(run_offset(0).is_err());
    }

    #[test]
    fn ccd_run_picks_its_own_file_out_of_the_series() {
        let dir = tmp_dir("ccd");
        for n in ["20260320_Run_15805_CT_Ang_0_1.tiff", "20260320_Run_15806_CT_Ang_180_2.tiff", "notes.txt"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let found = run_images_in(&dir, 15806);
        assert_eq!(found.len(), 1);
        assert!(found[0].path.ends_with("20260320_Run_15806_CT_Ang_180_2.tiff"));
        assert_eq!(found[0].n_images, 1);
        assert_eq!(found[0].size_text(), "file");
        assert_eq!(found[0].files().unwrap(), vec![found[0].path.clone()]);

        // No file named after the run: the folder itself (it has images).
        let found = run_images_in(&dir, 99999);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].path, dir);
        assert_eq!(found[0].n_images, 2);
        assert_eq!(found[0].size_text(), "2 files");
        assert_eq!(found[0].files().unwrap().len(), 2);

        // Missing folder: nothing.
        assert!(run_images_in(&dir.join("nope"), 15805).is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn folder_stack_counts_loadable_and_fits_separately() {
        let dir = tmp_dir("count");
        for n in ["a_00000.fits", "a_00001.fits", "a_Spectra.txt", "b.tif", "c.npy"] {
            std::fs::write(dir.join(n), b"").unwrap();
        }
        let s = stack_at(&dir).unwrap();
        assert_eq!((s.n_images, s.n_fits), (2, 2));
        assert!(s.is_loadable());
        assert_eq!(s.size_text(), "2 files");
        assert!(stack_at(&dir.join("missing")).is_none());

        // A raw Timepix folder: .fits only → not loadable, says so.
        let raw = tmp_dir("count_raw");
        std::fs::write(raw.join("a_00000.fits"), b"").unwrap();
        let s = stack_at(&raw).unwrap();
        assert!(!s.is_loadable());
        assert_eq!(s.size_text(), "1 .fits frames, not loadable here");
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&raw);
    }

    #[test]
    fn default_files_prefer_autoreduce_for_timepix() {
        let raw = tmp_dir("default_raw");
        let auto = tmp_dir("default_auto");
        std::fs::write(raw.join("a_00000.fits"), b"").unwrap();
        std::fs::write(auto.join("a_00000.tif"), b"").unwrap();
        let info = RunInfo {
            run: 1,
            ipts: "IPTS-1".into(),
            nexus: PathBuf::new(),
            detector: Some("MCP TPX1".into()),
            offset_us: None,
            folder_path: String::new(),
            image_dir: raw.clone(),
            raw: stack_at(&raw).into_iter().collect(),
            autoreduce_dir: auto.clone(),
            autoreduce: stack_at(&auto),
        };
        assert!(!info.raw_loadable());
        assert!(info.autoreduce_loadable());
        let (files, which) = info.default_files().unwrap();
        assert_eq!(which, "autoreduce");
        assert_eq!(files, vec![auto.join("a_00000.tif")]);

        let none = RunInfo { autoreduce: None, ..info.clone() };
        assert!(none.default_files().is_err());
        let _ = std::fs::remove_dir_all(&raw);
        let _ = std::fs::remove_dir_all(&auto);
    }

    /// End-to-end against real instrument data; skipped when /SNS/VENUS is
    /// not mounted (e.g. off-site).
    #[test]
    fn resolve_real_runs() {
        if !Path::new(VENUS_ROOT).is_dir() {
            eprintln!("skipping: {VENUS_ROOT} not mounted");
            return;
        }
        // Timepix run: raw .fits folder + autoreduce .tif folder.
        let info = resolve(23640).expect("resolve run 23640");
        assert_eq!(info.ipts, "IPTS-36967");
        assert_eq!(info.detector.as_deref(), Some("MCP TPX1"));
        assert!(info.is_timepix());
        assert!(info.offset_us.is_some_and(|v| v.is_finite()), "{:?}", info.offset_us);
        assert!(info.folder_path.starts_with("images/tpx1/"), "{}", info.folder_path);
        assert_eq!(info.raw.len(), 1);
        assert_eq!(info.raw[0].path, info.image_dir);
        let auto = info.autoreduce.as_ref().expect("autoreduce folder");
        assert!(auto.path.starts_with("/SNS/VENUS/IPTS-36967/shared/autoreduce/images/"));
        assert!(auto.n_images > 0);
        assert_eq!(info.default_files().unwrap().1, "autoreduce");

        // CCD run: one TIFF of the series, no autoreduce.
        let info = resolve(15805).expect("resolve run 15805");
        assert_eq!(info.detector.as_deref(), Some("Andor CCD iKon-XL"));
        assert!(!info.is_timepix());
        assert_eq!(info.raw.len(), 1);
        assert!(info.raw[0].path.is_file());
        assert!(names_run(&info.raw[0].path.to_string_lossy(), 15805));
        assert!(info.autoreduce.is_none());
        assert_eq!(info.default_files().unwrap().1, "raw");

        assert!(resolve(0).is_err());
    }
}
