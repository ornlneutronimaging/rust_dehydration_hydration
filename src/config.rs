//! Saving / loading the application settings as an HDF5 config file, so a
//! tuned configuration can be re-used later (or by other tools reading the
//! same file with h5py).
//!
//! File layout (all values are HDF5 attributes; strings are variable-length
//! UTF-8):
//!
//! ```text
//! /                    @tool, @tool_version, @format_version (u32),
//!                      @created_utc, @input_folder (informational, optional)
//! /correction          @engine ("mbirtorch.hsnt"),
//!                      @input_type ("auto"|"transmission"|"attenuation"),
//!                      @rank ("auto" or a number), @max_rank (u64),
//!                      @spectra ("mle"|"unconstrained"|"support"),
//!                      @dose (f64, -1 = unknown), @device (string),
//!                      @mode ("auto"|"full"|"stream"), @max_steps (u64),
//!                      @rel_tol (f64), @max_passes (u64),
//!                      @compile ("auto"|"on"|"off")
//! /physical_axis       @distance_m (f64), @detector_offset_us (f64)
//! /display             @colormap, @log_y (u8), @contrast_auto (u8),
//!                      @vmin (f64), @vmax (f64)
//! /region              @left, @right, @top, @bottom (u64, half-open px
//!                      bounds) — present only when a region is set
//! /mask                present only when a mask is defined:
//!                      @range_lo, @range_hi (f64) when the integrated-value
//!                      range is set; @file (path of a mask image) when one
//!                      was loaded (re-read on load, when it still exists);
//!                      dataset rects (n × 5 u64: left, right, top, bottom,
//!                      mode 0 = include / 1 = exclude)
//! ```
//!
//! Format version 2 (this beta). A version-1 file of the production tool
//! (NMF parameters `dataset_type`, `num_materials`, `safety_factor`,
//! `beta_loss`, `max_iter`) still loads: its dataset type becomes the input
//! type and its number of materials the rank; the NMF-only values are
//! ignored.
//!
//! Loading is tolerant of missing groups/attributes (defaults fill in) but
//! rejects files without a `format_version` root attribute and unknown enum
//! tokens, so a wrong file fails loudly instead of half-applying.

use crate::colormap::Colormap;
use crate::correction::CorrectionParams;
use crate::hsnt_cli::{Compile, Device, InputType, Rank, SolveMode, Spectra};
use crate::mask::{MaskRect, RectMode};
use crate::spectra;
use anyhow::{bail, Context, Result};
use hdf5::types::VarLenUnicode;
use std::path::{Path, PathBuf};

/// Bump when the file layout changes incompatibly.
pub const FORMAT_VERSION: u32 = 2;

/// The settings captured by (and restored from) a config file.
#[derive(Clone, PartialEq, Debug)]
pub struct AppConfig {
    pub params: CorrectionParams,
    /// Source–detector distance for the wavelength conversion (m).
    pub distance_m: f64,
    /// Detector offset added to the spectra TOF values (µs).
    pub offset_us: f64,
    pub colormap: Colormap,
    pub log_y: bool,
    pub contrast_auto: bool,
    pub vmin: f32,
    pub vmax: f32,
    /// Profile region as `[left, right, top, bottom]` (half-open px bounds).
    pub region: Option<[usize; 4]>,
    /// Folder the settings were tuned on — recorded for reference, not
    /// applied on load.
    pub input_folder: Option<PathBuf>,
    /// The mask definition: rectangles, integrated-value range, mask file.
    pub mask_rects: Vec<MaskRect>,
    pub mask_range: Option<(f32, f32)>,
    pub mask_file: Option<PathBuf>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            params: CorrectionParams::default(),
            distance_m: spectra::DEFAULT_DISTANCE_M,
            offset_us: 0.0,
            colormap: Colormap::Viridis,
            log_y: false,
            contrast_auto: true,
            vmin: 0.0,
            vmax: 1.0,
            region: None,
            input_folder: None,
            mask_rects: Vec::new(),
            mask_range: None,
            mask_file: None,
        }
    }
}

pub fn save(path: &Path, cfg: &AppConfig) -> Result<()> {
    let file = hdf5::File::create(path)
        .with_context(|| format!("create {}", path.display()))?;

    write_str(&file, "tool", "rust_dehydration_hydration")?;
    write_str(&file, "tool_version", env!("CARGO_PKG_VERSION"))?;
    write_str(&file, "created_utc", &crate::export::iso8601_utc_now())?;
    write_u32(&file, "format_version", FORMAT_VERSION)?;
    if let Some(dir) = &cfg.input_folder {
        write_str(&file, "input_folder", &dir.display().to_string())?;
    }

    let p = &cfg.params;
    let g = file.create_group("correction")?;
    write_str(&g, "engine", "mbirtorch.hsnt")?;
    write_str(&g, "input_type", p.input_type.label())?;
    write_str(&g, "rank", &p.rank.label())?;
    write_u64(&g, "max_rank", p.max_rank as u64)?;
    write_str(&g, "spectra", p.spectra.label())?;
    write_f64(&g, "dose", p.dose.unwrap_or(-1.0))?;
    write_str(&g, "device", &p.device.label())?;
    write_str(&g, "mode", p.mode.label())?;
    write_u64(&g, "max_steps", p.max_steps as u64)?;
    write_f64(&g, "rel_tol", p.rel_tol)?;
    write_u64(&g, "max_passes", p.max_passes as u64)?;
    write_str(&g, "compile", p.compile.label())?;

    let g = file.create_group("physical_axis")?;
    write_f64(&g, "distance_m", cfg.distance_m)?;
    write_f64(&g, "detector_offset_us", cfg.offset_us)?;

    let g = file.create_group("display")?;
    write_str(&g, "colormap", cfg.colormap.label())?;
    write_u8(&g, "log_y", cfg.log_y as u8)?;
    write_u8(&g, "contrast_auto", cfg.contrast_auto as u8)?;
    write_f64(&g, "vmin", cfg.vmin as f64)?;
    write_f64(&g, "vmax", cfg.vmax as f64)?;

    if let Some([left, right, top, bottom]) = cfg.region {
        let g = file.create_group("region")?;
        write_u64(&g, "left", left as u64)?;
        write_u64(&g, "right", right as u64)?;
        write_u64(&g, "top", top as u64)?;
        write_u64(&g, "bottom", bottom as u64)?;
    }

    if !cfg.mask_rects.is_empty() || cfg.mask_range.is_some() || cfg.mask_file.is_some() {
        let g = file.create_group("mask")?;
        if let Some((lo, hi)) = cfg.mask_range {
            write_f64(&g, "range_lo", lo as f64)?;
            write_f64(&g, "range_hi", hi as f64)?;
        }
        if let Some(path) = &cfg.mask_file {
            write_str(&g, "file", &path.display().to_string())?;
        }
        if !cfg.mask_rects.is_empty() {
            let mut rects = ndarray::Array2::<u64>::zeros((cfg.mask_rects.len(), 5));
            for (i, r) in cfg.mask_rects.iter().enumerate() {
                rects[(i, 0)] = r.left as u64;
                rects[(i, 1)] = r.right as u64;
                rects[(i, 2)] = r.top as u64;
                rects[(i, 3)] = r.bottom as u64;
                rects[(i, 4)] = (r.mode == RectMode::Exclude) as u64;
            }
            g.new_dataset_builder().with_data(&rects).create("rects")?;
        }
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<AppConfig> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("open {}", path.display()))?;
    if read_u32(&file, "format_version").is_none() {
        bail!(
            "{} does not look like a dehydration/hydration config file \
             (no format_version attribute)",
            path.display()
        );
    }

    let mut cfg = AppConfig::default();
    cfg.input_folder = read_str(&file, "input_folder").map(PathBuf::from);

    if let Ok(g) = file.group("correction") {
        let p = &mut cfg.params;
        // Version-1 (production NMF) names, mapped onto the new parameters.
        if let Some(s) = read_str(&g, "dataset_type") {
            p.input_type = token(&s, "dataset_type", InputType::parse)?;
        }
        if let Some(v) = read_u64(&g, "num_materials") {
            p.rank = Rank::Fixed((v as usize).max(1));
        }
        // Version-2 names.
        if let Some(s) = read_str(&g, "input_type") {
            p.input_type = token(&s, "input_type", InputType::parse)?;
        }
        if let Some(s) = read_str(&g, "rank") {
            p.rank = token(&s, "rank", Rank::parse)?;
        }
        if let Some(v) = read_u64(&g, "max_rank") {
            p.max_rank = (v as usize).max(1);
        }
        if let Some(s) = read_str(&g, "spectra") {
            p.spectra = token(&s, "spectra", Spectra::parse)?;
        }
        if let Some(v) = read_f64(&g, "dose") {
            p.dose = (v.is_finite() && v > 0.0).then_some(v);
        }
        if let Some(s) = read_str(&g, "device") {
            p.device = token(&s, "device", Device::parse)?;
        }
        if let Some(s) = read_str(&g, "mode") {
            p.mode = token(&s, "mode", SolveMode::parse)?;
        }
        if let Some(v) = read_u64(&g, "max_steps") {
            p.max_steps = (v as usize).max(1);
        }
        if let Some(v) = read_f64(&g, "rel_tol").filter(|v| v.is_finite() && *v >= 0.0) {
            p.rel_tol = v;
        }
        if let Some(v) = read_u64(&g, "max_passes") {
            p.max_passes = v as usize;
        }
        if let Some(s) = read_str(&g, "compile") {
            p.compile = token(&s, "compile", Compile::parse)?;
        }
    }

    if let Ok(g) = file.group("physical_axis") {
        if let Some(v) = read_f64(&g, "distance_m").filter(|v| v.is_finite() && *v > 0.0) {
            cfg.distance_m = v;
        }
        if let Some(v) = read_f64(&g, "detector_offset_us").filter(|v| v.is_finite()) {
            cfg.offset_us = v;
        }
    }

    if let Ok(g) = file.group("display") {
        if let Some(s) = read_str(&g, "colormap") {
            cfg.colormap = colormap_from(&s)?;
        }
        if let Some(v) = read_u8(&g, "log_y") {
            cfg.log_y = v != 0;
        }
        if let Some(v) = read_u8(&g, "contrast_auto") {
            cfg.contrast_auto = v != 0;
        }
        if let Some(v) = read_f64(&g, "vmin").filter(|v| v.is_finite()) {
            cfg.vmin = v as f32;
        }
        if let Some(v) = read_f64(&g, "vmax").filter(|v| v.is_finite()) {
            cfg.vmax = v as f32;
        }
    }

    if let Ok(g) = file.group("region") {
        if let (Some(l), Some(r), Some(t), Some(b)) = (
            read_u64(&g, "left"),
            read_u64(&g, "right"),
            read_u64(&g, "top"),
            read_u64(&g, "bottom"),
        ) {
            if l < r && t < b {
                cfg.region = Some([l as usize, r as usize, t as usize, b as usize]);
            }
        }
    }

    if let Ok(g) = file.group("mask") {
        if let (Some(lo), Some(hi)) = (read_f64(&g, "range_lo"), read_f64(&g, "range_hi"))
            && lo.is_finite()
            && hi.is_finite()
            && lo <= hi
        {
            cfg.mask_range = Some((lo as f32, hi as f32));
        }
        cfg.mask_file = read_str(&g, "file").map(PathBuf::from);
        if let Ok(ds) = g.dataset("rects") {
            let rects = ds.read_2d::<u64>().context("read the mask rectangles")?;
            if rects.ncols() != 5 {
                bail!("mask rectangles have {} columns, expected 5", rects.ncols());
            }
            for row in rects.rows() {
                let (l, r, t, b) = (row[0] as usize, row[1] as usize, row[2] as usize, row[3] as usize);
                if l < r && t < b {
                    cfg.mask_rects.push(MaskRect {
                        left: l,
                        right: r,
                        top: t,
                        bottom: b,
                        mode: if row[4] == 0 { RectMode::Include } else { RectMode::Exclude },
                    });
                }
            }
        }
    }
    Ok(cfg)
}

// ----- enum tokens -------------------------------------------------------

fn token<T>(s: &str, name: &str, parse: fn(&str) -> Option<T>) -> Result<T> {
    parse(s).with_context(|| format!("unknown {name} {s:?} in config file"))
}

fn colormap_from(s: &str) -> Result<Colormap> {
    Colormap::ALL
        .into_iter()
        .find(|c| c.label() == s)
        .with_context(|| format!("unknown colormap {s:?} in config file"))
}

// ----- attribute helpers ---------------------------------------------------

fn write_str(loc: &hdf5::Location, name: &str, value: &str) -> Result<()> {
    let v: VarLenUnicode = value
        .parse()
        .map_err(|e| anyhow::anyhow!("attribute {name}: {e}"))?;
    loc.new_attr::<VarLenUnicode>().create(name)?.write_scalar(&v)?;
    Ok(())
}

macro_rules! scalar_attr {
    ($write:ident, $read:ident, $ty:ty) => {
        fn $write(loc: &hdf5::Location, name: &str, value: $ty) -> Result<()> {
            loc.new_attr::<$ty>().create(name)?.write_scalar(&value)?;
            Ok(())
        }
        fn $read(loc: &hdf5::Location, name: &str) -> Option<$ty> {
            loc.attr(name).ok()?.read_scalar::<$ty>().ok()
        }
    };
}
scalar_attr!(write_u8, read_u8, u8);
scalar_attr!(write_u32, read_u32, u32);
scalar_attr!(write_u64, read_u64, u64);
scalar_attr!(write_f64, read_f64, f64);

fn read_str(loc: &hdf5::Location, name: &str) -> Option<String> {
    let v = loc.attr(name).ok()?.read_scalar::<VarLenUnicode>().ok()?;
    Some(v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("dh_config_{}_{name}.h5", std::process::id()))
    }

    #[test]
    fn round_trip_full() {
        let cfg = AppConfig {
            params: CorrectionParams {
                input_type: InputType::Attenuation,
                rank: Rank::Fixed(4),
                max_rank: 8,
                spectra: Spectra::Support,
                dose: Some(37.5),
                device: Device::Cuda(2),
                mode: SolveMode::Stream,
                max_steps: 250,
                rel_tol: 1e-6,
                max_passes: 12,
                compile: Compile::Off,
            },
            distance_m: 23.72,
            offset_us: 9600.0,
            colormap: Colormap::Turbo,
            log_y: true,
            contrast_auto: false,
            vmin: 0.125,
            vmax: 1.75,
            region: Some([10, 200, 20, 180]),
            input_folder: Some(PathBuf::from("/some/data/folder")),
            mask_rects: vec![
                MaskRect { left: 1, right: 9, top: 2, bottom: 8, mode: RectMode::Include },
                MaskRect { left: 3, right: 4, top: 3, bottom: 4, mode: RectMode::Exclude },
            ],
            mask_range: Some((0.5, 1.5)),
            mask_file: Some(PathBuf::from("/some/mask.tif")),
        };
        let path = tmp("full");
        save(&path, &cfg).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn round_trip_defaults() {
        let cfg = AppConfig::default();
        let path = tmp("defaults");
        save(&path, &cfg).unwrap();
        let loaded = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded, cfg);
        assert!(loaded.region.is_none());
        assert!(loaded.mask_rects.is_empty() && loaded.mask_range.is_none() && loaded.mask_file.is_none());
        assert_eq!(loaded.params.rank, Rank::Auto);
        assert!(loaded.params.dose.is_none());
    }

    #[test]
    fn loads_a_version1_production_file() {
        let path = tmp("v1");
        let file = hdf5::File::create(&path).unwrap();
        write_u32(&file, "format_version", 1).unwrap();
        let g = file.create_group("correction").unwrap();
        write_str(&g, "dataset_type", "attenuation").unwrap();
        write_u64(&g, "num_materials", 3).unwrap();
        write_f64(&g, "safety_factor", 16.0).unwrap();
        write_str(&g, "beta_loss", "frobenius").unwrap();
        write_u64(&g, "max_iter", 300).unwrap();
        drop(file);
        let cfg = load(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(cfg.params.input_type, InputType::Attenuation);
        assert_eq!(cfg.params.rank, Rank::Fixed(3));
        assert_eq!(cfg.params.max_steps, 1000, "NMF-only values are ignored");
    }

    #[test]
    fn rejects_foreign_h5() {
        let path = tmp("foreign");
        let file = hdf5::File::create(&path).unwrap();
        file.create_group("something_else").unwrap();
        drop(file);
        let err = load(&path).unwrap_err().to_string();
        std::fs::remove_file(&path).ok();
        assert!(err.contains("format_version"), "unexpected error: {err}");
    }
}
