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
//! /correction          @dataset_type ("attenuation"|"transmission"),
//!                      @num_materials (u64), @safety_factor (f64),
//!                      @beta_loss ("frobenius"|"kullback-leibler"),
//!                      @max_iter (u64)
//! /physical_axis       @distance_m (f64), @detector_offset_us (f64)
//! /display             @colormap, @log_y (u8), @contrast_auto (u8),
//!                      @vmin (f64), @vmax (f64)
//! /region              @left, @right, @top, @bottom (u64, half-open px
//!                      bounds) — present only when a region is set
//! ```
//!
//! Loading is tolerant of missing groups/attributes (defaults fill in) but
//! rejects files without a `format_version` root attribute and unknown enum
//! tokens, so a wrong file fails loudly instead of half-applying.

use crate::colormap::Colormap;
use crate::correction::CorrectionParams;
use crate::hsnt::DatasetType;
use crate::nmf::BetaLoss;
use crate::spectra;
use anyhow::{bail, Context, Result};
use hdf5::types::VarLenUnicode;
use std::path::{Path, PathBuf};

/// Bump when the file layout changes incompatibly.
pub const FORMAT_VERSION: u32 = 1;

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

    let g = file.create_group("correction")?;
    write_str(&g, "dataset_type", cfg.params.dataset_type.label())?;
    write_u64(&g, "num_materials", cfg.params.num_materials as u64)?;
    write_f64(&g, "safety_factor", cfg.params.safety_factor)?;
    write_str(&g, "beta_loss", cfg.params.beta_loss.label())?;
    write_u64(&g, "max_iter", cfg.params.max_iter as u64)?;

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
        if let Some(s) = read_str(&g, "dataset_type") {
            cfg.params.dataset_type = dataset_type_from(&s)?;
        }
        if let Some(v) = read_u64(&g, "num_materials") {
            cfg.params.num_materials = (v as usize).max(1);
        }
        if let Some(v) = read_f64(&g, "safety_factor").filter(|v| v.is_finite() && *v >= 1.0) {
            cfg.params.safety_factor = v;
        }
        if let Some(s) = read_str(&g, "beta_loss") {
            cfg.params.beta_loss = beta_loss_from(&s)?;
        }
        if let Some(v) = read_u64(&g, "max_iter") {
            cfg.params.max_iter = (v as usize).max(1);
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
    Ok(cfg)
}

// ----- enum tokens -------------------------------------------------------

fn dataset_type_from(s: &str) -> Result<DatasetType> {
    match s {
        "attenuation" => Ok(DatasetType::Attenuation),
        "transmission" => Ok(DatasetType::Transmission),
        _ => bail!("unknown dataset_type {s:?} in config file"),
    }
}

fn beta_loss_from(s: &str) -> Result<BetaLoss> {
    match s {
        "frobenius" => Ok(BetaLoss::Frobenius),
        "kullback-leibler" => Ok(BetaLoss::KullbackLeibler),
        _ => bail!("unknown beta_loss {s:?} in config file"),
    }
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
                dataset_type: DatasetType::Transmission,
                num_materials: 4,
                safety_factor: 8.0,
                beta_loss: BetaLoss::KullbackLeibler,
                max_iter: 250,
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
