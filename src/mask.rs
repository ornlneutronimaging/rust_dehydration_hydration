//! The pixel mask of a correction: which pixels of the stack are handed to
//! mbirtorch. The mask is the conjunction of up to three sources, all
//! optional — a mask image loaded from a file (nonzero = keep), a range on
//! the integrated image (keep the pixels whose summed intensity lies within
//! it), and rectangles drawn on the integrated image (include rectangles:
//! the pixel must lie in one of them; exclude rectangles: it must lie in
//! none). No source at all means every pixel. Pixels outside the mask are
//! not sent to the solver and keep their raw values in the corrected stack.
//!
//! Masks live in the oriented frame coordinates of the loaded stack (what
//! the window shows); mask files are read and written with the stack's
//! detector orientation, like the images themselves, so a mask made from
//! the same run's images by another tool lines up.

use anyhow::{bail, Context, Result};
use detector_orientation::{Orientation, Selection};
use ndarray::Array2;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RectMode {
    Include,
    Exclude,
}

impl RectMode {
    pub fn label(self) -> &'static str {
        match self {
            RectMode::Include => "include",
            RectMode::Exclude => "exclude",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "include" => Some(RectMode::Include),
            "exclude" => Some(RectMode::Exclude),
            _ => None,
        }
    }
}

/// A rectangle in pixels, half-open (`left..right`, `top..bottom`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MaskRect {
    pub left: usize,
    pub right: usize,
    pub top: usize,
    pub bottom: usize,
    pub mode: RectMode,
}

impl MaskRect {
    /// From two corners in any order, clamped to a `w`×`h` image; `None`
    /// when empty.
    pub fn from_corners(a: (f32, f32), b: (f32, f32), w: usize, h: usize, mode: RectMode) -> Option<Self> {
        let x0 = a.0.min(b.0).round().max(0.0) as usize;
        let x1 = (a.0.max(b.0).round().max(0.0) as usize).min(w);
        let y0 = a.1.min(b.1).round().max(0.0) as usize;
        let y1 = (a.1.max(b.1).round().max(0.0) as usize).min(h);
        (x1 > x0 && y1 > y0).then_some(Self { left: x0, right: x1, top: y0, bottom: y1, mode })
    }

    /// `x0,y0,x1,y1` (half-open, in pixels) as on the command line.
    pub fn parse(text: &str, mode: RectMode) -> Option<Self> {
        let v: Vec<usize> = text.split(',').map(|s| s.trim().parse::<usize>().ok()).collect::<Option<_>>()?;
        let [x0, y0, x1, y1] = v.as_slice() else { return None };
        (x1 > x0 && y1 > y0).then_some(Self { left: *x0, right: *x1, top: *y0, bottom: *y1, mode })
    }

    pub fn contains(&self, x: usize, y: usize) -> bool {
        x >= self.left && x < self.right && y >= self.top && y < self.bottom
    }

    pub fn label(&self) -> String {
        format!(
            "{} x {}..{}, y {}..{} ({}×{})",
            self.mode.label(),
            self.left,
            self.right,
            self.top,
            self.bottom,
            self.right - self.left,
            self.bottom - self.top
        )
    }
}

/// A mask image loaded from a file.
#[derive(Clone, Debug)]
pub struct MaskFile {
    pub path: PathBuf,
    /// true = keep.
    pub pixels: Arc<Array2<bool>>,
}

impl PartialEq for MaskFile {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.pixels.dim() == other.pixels.dim()
    }
}

/// How the mask is defined (what the GUI edits and the config saves).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct MaskSpec {
    pub rects: Vec<MaskRect>,
    /// Keep the pixels whose integrated (summed) intensity lies in `[lo, hi]`.
    pub range: Option<(f32, f32)>,
    pub file: Option<MaskFile>,
}

impl MaskSpec {
    /// No source: every pixel is used.
    pub fn is_empty(&self) -> bool {
        self.rects.is_empty() && self.range.is_none() && self.file.is_none()
    }

    /// The pixel mask for an `integrated` image of the stack's size; `None`
    /// when the spec is empty (all pixels). A file mask of another size is
    /// ignored, with an error.
    pub fn build(&self, integrated: &Array2<f32>) -> Result<Option<Array2<bool>>> {
        if self.is_empty() {
            return Ok(None);
        }
        let (h, w) = integrated.dim();
        let mut mask = match &self.file {
            Some(f) => {
                if f.pixels.dim() != (h, w) {
                    bail!(
                        "the mask file {} is {}×{} px, the stack {}×{}",
                        f.path.display(),
                        f.pixels.ncols(),
                        f.pixels.nrows(),
                        w,
                        h
                    );
                }
                (*f.pixels).clone()
            }
            None => Array2::from_elem((h, w), true),
        };
        if let Some((lo, hi)) = self.range {
            for (m, &v) in mask.iter_mut().zip(integrated.iter()) {
                if !(v >= lo && v <= hi) {
                    *m = false;
                }
            }
        }
        let includes: Vec<&MaskRect> = self.rects.iter().filter(|r| r.mode == RectMode::Include).collect();
        let excludes: Vec<&MaskRect> = self.rects.iter().filter(|r| r.mode == RectMode::Exclude).collect();
        if !includes.is_empty() || !excludes.is_empty() {
            for ((y, x), m) in mask.indexed_iter_mut() {
                if !*m {
                    continue;
                }
                let outside_includes = !includes.is_empty() && !includes.iter().any(|r| r.contains(x, y));
                if outside_includes || excludes.iter().any(|r| r.contains(x, y)) {
                    *m = false;
                }
            }
        }
        Ok(Some(mask))
    }

    /// One line saying what the mask is made of.
    pub fn describe(&self) -> String {
        if self.is_empty() {
            return "no mask (all pixels)".to_owned();
        }
        let mut parts = Vec::new();
        if let Some(f) = &self.file {
            parts.push(format!("file {}", f.path.display()));
        }
        if let Some((lo, hi)) = self.range {
            parts.push(format!("integrated value in [{lo}, {hi}]"));
        }
        let n_inc = self.rects.iter().filter(|r| r.mode == RectMode::Include).count();
        let n_exc = self.rects.len() - n_inc;
        if n_inc > 0 {
            parts.push(format!("{n_inc} include rectangle(s)"));
        }
        if n_exc > 0 {
            parts.push(format!("{n_exc} exclude rectangle(s)"));
        }
        parts.join(", ")
    }
}

/// Number of selected pixels.
pub fn count(mask: &Array2<bool>) -> usize {
    mask.iter().filter(|m| **m).count()
}

/// The mask of a `bin`×`bin` binned stack: a binned pixel is kept when at
/// least half of its source pixels are (ragged edges are dropped, like
/// [`crate::correction::bin_frames`]).
pub fn bin_mask(mask: &Array2<bool>, bin: usize) -> Array2<bool> {
    if bin <= 1 {
        return mask.clone();
    }
    let (h, w) = (mask.nrows() / bin, mask.ncols() / bin);
    Array2::from_shape_fn((h, w), |(y, x)| {
        let mut n = 0;
        for dy in 0..bin {
            for dx in 0..bin {
                n += mask[(y * bin + dy, x * bin + dx)] as usize;
            }
        }
        2 * n >= bin * bin
    })
}

/// Row-major `(y, x)` of every selected pixel.
pub fn positions(mask: &Array2<bool>) -> Vec<(usize, usize)> {
    mask.indexed_iter().filter(|(_, m)| **m).map(|(p, _)| p).collect()
}

/// Read a mask image (TIFF or `.npy`; nonzero = keep) with the stack's
/// detector orientation, so it lines up with the loaded frames.
pub fn load_file(path: &Path, detector: Selection) -> Result<Array2<bool>> {
    let stack = crate::loader::load_paths_with_progress(&[path.to_path_buf()], detector, |_, _| {})
        .with_context(|| format!("read the mask {}", path.display()))?;
    let Some(frame) = stack.frames.first() else {
        bail!("{} holds no image", path.display());
    };
    if stack.frames.len() > 1 {
        bail!("{} holds {} images; a mask is a single image", path.display(), stack.frames.len());
    }
    Ok(frame.mapv(|v| v != 0.0 && v.is_finite()))
}

/// Write a mask as an 8-bit TIFF (255 = keep, 0 = excluded) in the on-disk
/// orientation of the input images.
pub fn save_file(path: &Path, mask: &Array2<bool>, orientation: Orientation) -> Result<()> {
    use tiff::encoder::{colortype::Gray8, TiffEncoder};
    let as_u8 = mask.mapv(|m| if m { 255u8 } else { 0u8 });
    let data = orientation.undo_view(as_u8.view());
    let (h, w) = (data.nrows(), data.ncols());
    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = TiffEncoder::new(std::io::BufWriter::new(file))
        .with_context(|| format!("init TIFF encoder for {}", path.display()))?;
    let data = data.as_standard_layout();
    enc.write_image::<Gray8>(w as u32, h as u32, data.as_slice().expect("standard layout"))
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrated() -> Array2<f32> {
        Array2::from_shape_fn((4, 6), |(y, x)| (y * 6 + x) as f32)
    }

    #[test]
    fn empty_spec_means_all_pixels() {
        assert!(MaskSpec::default().build(&integrated()).unwrap().is_none());
        assert_eq!(MaskSpec::default().describe(), "no mask (all pixels)");
    }

    #[test]
    fn rectangles_include_then_exclude() {
        let spec = MaskSpec {
            rects: vec![
                MaskRect { left: 0, right: 3, top: 0, bottom: 4, mode: RectMode::Include },
                MaskRect { left: 1, right: 2, top: 1, bottom: 2, mode: RectMode::Exclude },
            ],
            ..Default::default()
        };
        let m = spec.build(&integrated()).unwrap().unwrap();
        assert!(m[(0, 0)] && m[(3, 2)]);
        assert!(!m[(0, 3)], "outside every include rectangle");
        assert!(!m[(1, 1)], "inside the exclude rectangle");
        assert_eq!(count(&m), 11);
        assert_eq!(positions(&m).len(), 11);
        assert!(spec.describe().contains("1 include rectangle(s), 1 exclude rectangle(s)"));
    }

    #[test]
    fn range_and_file_combine() {
        let file = Array2::from_shape_fn((4, 6), |(y, _)| y < 2);
        let spec = MaskSpec {
            range: Some((2.0, 20.0)),
            file: Some(MaskFile { path: PathBuf::from("m.tif"), pixels: Arc::new(file) }),
            ..Default::default()
        };
        let m = spec.build(&integrated()).unwrap().unwrap();
        // Rows 0-1 (file) with values 2..=20 (range): values 2..=11.
        assert_eq!(count(&m), 10);
        assert!(!m[(0, 1)] && m[(0, 2)] && m[(1, 5)] && !m[(2, 0)]);
        let wrong = MaskSpec {
            file: Some(MaskFile { path: PathBuf::from("m.tif"), pixels: Arc::new(Array2::from_elem((2, 2), true)) }),
            ..Default::default()
        };
        assert!(wrong.build(&integrated()).is_err());
    }

    #[test]
    fn binning_takes_the_majority() {
        let m = Array2::from_shape_fn((4, 4), |(y, x)| x < 1 || (y < 2 && x < 3));
        let b = bin_mask(&m, 2);
        assert_eq!(b.dim(), (2, 2));
        assert!(b[(0, 0)]); // 4 of 4
        assert!(b[(0, 1)]); // (0,2),(1,2) kept = 2 of 4 → kept
        assert!(b[(1, 0)]); // 2 of 4
        assert!(!b[(1, 1)]); // 0 of 4
    }

    #[test]
    fn rect_parsing_and_corners() {
        let r = MaskRect::parse("10, 20, 30, 40", RectMode::Exclude).unwrap();
        assert_eq!((r.left, r.top, r.right, r.bottom), (10, 20, 30, 40));
        assert!(MaskRect::parse("10,20,10,40", RectMode::Include).is_none());
        assert!(MaskRect::parse("a,b", RectMode::Include).is_none());
        let c = MaskRect::from_corners((5.4, 7.6), (1.2, 2.0), 4, 100, RectMode::Include).unwrap();
        assert_eq!((c.left, c.right, c.top, c.bottom), (1, 4, 2, 8));
        assert!(MaskRect::from_corners((1.0, 1.0), (1.2, 5.0), 10, 10, RectMode::Include).is_none());
    }

    #[test]
    fn mask_file_round_trip() {
        let dir = std::env::temp_dir().join(format!("dh_mask_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mask.tif");
        let m = Array2::from_shape_fn((3, 5), |(y, x)| (y + x) % 2 == 0);
        save_file(&path, &m, Orientation::Transpose).unwrap();
        let sel = Selection { manual: Some(detector_orientation::Detector::Timepix), ..Default::default() };
        let back = load_file(&path, sel).unwrap();
        assert_eq!(back, m);
        std::fs::remove_dir_all(&dir).ok();
    }
}
