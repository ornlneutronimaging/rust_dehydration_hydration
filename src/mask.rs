//! The pixel mask of a correction (`--mask`): which pixels of the stack go
//! into the NMF. Pixels outside the mask are not part of the factorization
//! and keep their input values in the corrected stack, so a region the
//! sample masker filled with 0 / NaN cannot pull the spectra toward a flat
//! opaque "material" (0 transmission → attenuation ln(1000) in every band).
//!
//! Masks live in the oriented frame coordinates of the loaded stack; mask
//! files are read and written with the stack's detector orientation, like
//! the images themselves, so the `<folder>_sample_mask.tif` the sample
//! masker writes beside a masked stack lines up with its frames.

use anyhow::{bail, Context, Result};
use detector_orientation::{Orientation, Selection};
use ndarray::Array2;
use std::path::Path;

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

    #[test]
    fn binning_keeps_majority_blocks() {
        let m = Array2::from_shape_fn((4, 5), |(y, x)| y < 2 || x == 0);
        let b = bin_mask(&m, 2);
        assert_eq!(b.dim(), (2, 2));
        assert!(b[(0, 0)] && b[(0, 1)], "top blocks fully kept");
        assert!(b[(1, 0)], "two of four kept: majority rule keeps it");
        assert!(!b[(1, 1)]);
        assert_eq!(count(&m), 12);
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
