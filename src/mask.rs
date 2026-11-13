//! The pixel mask of a correction: which pixels of the stack are handed to
//! mbirtorch. The mask is the conjunction of up to three sources, all
//! optional — a mask image loaded from a file (nonzero = keep), a range on
//! the integrated image (keep the pixels whose summed intensity lies within
//! it), and shapes drawn on the integrated image — rectangles, ellipses and
//! polygons (include shapes: the pixel must lie in one of them; exclude
//! shapes: it must lie in none). No source at all means every pixel. Pixels
//! outside the mask are not sent to the solver and keep their raw values in
//! the corrected stack.
//!
//! Masks live in the oriented frame coordinates of the loaded stack (what
//! the window shows); mask files are read and written with the stack's
//! detector orientation, like the images themselves, so a mask made from
//! the same run's images by another tool lines up.
//!
//! Shape coordinates are in pixel-edge units: pixel `(x, y)` covers
//! `x..x+1` × `y..y+1` and belongs to a shape when its center
//! `(x + 0.5, y + 0.5)` does.

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

impl From<MaskRect> for MaskShape {
    fn from(r: MaskRect) -> Self {
        MaskShape {
            geom: Geom::Rect([r.left as f32, r.top as f32, r.right as f32, r.bottom as f32]),
            mode: r.mode,
        }
    }
}

/// The outline of a mask shape.
#[derive(Clone, PartialEq, Debug)]
pub enum Geom {
    /// `[x0, y0, x1, y1]`, half-open.
    Rect([f32; 4]),
    /// The ellipse inscribed in the box `[x0, y0, x1, y1]`.
    Ellipse([f32; 4]),
    /// The vertices, in order (closed implicitly; even-odd interior).
    Polygon(Vec<[f32; 2]>),
}

/// The edges each of the 8 handles of a box drags: `(hx, vy)` with
/// hx/vy ∈ {-1, 0, 1} (left/none/right, top/none/bottom), in the order
/// [`MaskShape::handles`] returns them.
pub const BOX_HANDLES: [(i8, i8); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// A shape drawn on the integrated image, kept (`Include`) or dropped
/// (`Exclude`).
#[derive(Clone, PartialEq, Debug)]
pub struct MaskShape {
    pub geom: Geom,
    pub mode: RectMode,
}

impl MaskShape {
    pub fn rect(a: [f32; 2], b: [f32; 2], mode: RectMode) -> Self {
        MaskShape { geom: Geom::Rect([a[0], a[1], b[0], b[1]]), mode }
    }

    pub fn ellipse(a: [f32; 2], b: [f32; 2], mode: RectMode) -> Self {
        MaskShape { geom: Geom::Ellipse([a[0], a[1], b[0], b[1]]), mode }
    }

    pub fn polygon(points: Vec<[f32; 2]>, mode: RectMode) -> Self {
        MaskShape { geom: Geom::Polygon(points), mode }
    }

    /// `cx,cy,rx,ry` (center and radii, in pixels) as on the command line.
    pub fn parse_ellipse(text: &str, mode: RectMode) -> Option<Self> {
        let v = parse_numbers(text)?;
        let [cx, cy, rx, ry] = v.as_slice() else { return None };
        (*rx > 0.0 && *ry > 0.0).then(|| Self::ellipse([cx - rx, cy - ry], [cx + rx, cy + ry], mode))
    }

    /// `x0,y0,x1,y1,x2,y2,…` (at least three vertices) as on the command
    /// line.
    pub fn parse_polygon(text: &str, mode: RectMode) -> Option<Self> {
        let v = parse_numbers(text)?;
        if v.len() < 6 || v.len() % 2 != 0 {
            return None;
        }
        Some(Self::polygon(v.chunks(2).map(|c| [c[0], c[1]]).collect(), mode))
    }

    pub fn kind(&self) -> &'static str {
        match self.geom {
            Geom::Rect(_) => "rectangle",
            Geom::Ellipse(_) => "ellipse",
            Geom::Polygon(_) => "polygon",
        }
    }

    /// Bounding box `[x0, y0, x1, y1]` (ordered).
    pub fn bbox(&self) -> [f32; 4] {
        match &self.geom {
            Geom::Rect(b) | Geom::Ellipse(b) => {
                [b[0].min(b[2]), b[1].min(b[3]), b[0].max(b[2]), b[1].max(b[3])]
            }
            Geom::Polygon(pts) => {
                let mut b = [f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];
                for p in pts {
                    b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
                }
                b
            }
        }
    }

    /// Whether the point `(x, y)` (pixel-edge units) lies in the shape.
    pub fn contains(&self, x: f32, y: f32) -> bool {
        let [x0, y0, x1, y1] = self.bbox();
        match &self.geom {
            Geom::Rect(_) => x >= x0 && x < x1 && y >= y0 && y < y1,
            Geom::Ellipse(_) => {
                let (rx, ry) = ((x1 - x0) * 0.5, (y1 - y0) * 0.5);
                if rx <= 0.0 || ry <= 0.0 {
                    return false;
                }
                let (dx, dy) = ((x - (x0 + x1) * 0.5) / rx, (y - (y0 + y1) * 0.5) / ry);
                dx * dx + dy * dy <= 1.0
            }
            Geom::Polygon(pts) => {
                // Even-odd rule: count the edges a ray to the right crosses.
                let mut inside = false;
                let mut j = pts.len().wrapping_sub(1);
                for i in 0..pts.len() {
                    let (a, b) = (pts[i], pts[j]);
                    if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
                        inside = !inside;
                    }
                    j = i;
                }
                inside
            }
        }
    }

    pub fn translate(&mut self, dx: f32, dy: f32) {
        match &mut self.geom {
            Geom::Rect(b) | Geom::Ellipse(b) => {
                b[0] += dx;
                b[2] += dx;
                b[1] += dy;
                b[3] += dy;
            }
            Geom::Polygon(pts) => {
                for p in pts {
                    p[0] += dx;
                    p[1] += dy;
                }
            }
        }
    }

    /// Move by `(dx, dy)`, keeping the center of the shape on a `w`×`h`
    /// image so it cannot be dragged out of reach.
    pub fn translate_within(&mut self, dx: f32, dy: f32, w: usize, h: usize) {
        let [x0, y0, x1, y1] = self.bbox();
        let (cx, cy) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
        let nx = (cx + dx).clamp(0.0, w as f32);
        let ny = (cy + dy).clamp(0.0, h as f32);
        self.translate(nx - cx, ny - cy);
    }

    /// The points a shape is reshaped by: the 8 handles of the box of a
    /// rectangle or an ellipse (in [`BOX_HANDLES`] order), the vertices of a
    /// polygon.
    pub fn handles(&self) -> Vec<[f32; 2]> {
        match &self.geom {
            Geom::Rect(_) | Geom::Ellipse(_) => {
                let [x0, y0, x1, y1] = self.bbox();
                let (mx, my) = ((x0 + x1) * 0.5, (y0 + y1) * 0.5);
                BOX_HANDLES
                    .iter()
                    .map(|&(hx, vy)| {
                        [
                            match hx { -1 => x0, 1 => x1, _ => mx },
                            match vy { -1 => y0, 1 => y1, _ => my },
                        ]
                    })
                    .collect()
            }
            Geom::Polygon(pts) => pts.clone(),
        }
    }

    /// Drag handle `i` (as numbered by [`Self::handles`]) to `p`. The box of
    /// a rectangle or ellipse may cross itself during the drag: call
    /// [`Self::normalize`] when it ends.
    pub fn drag_handle(&mut self, i: usize, p: [f32; 2]) {
        match &mut self.geom {
            Geom::Rect(b) | Geom::Ellipse(b) => {
                let Some(&(hx, vy)) = BOX_HANDLES.get(i) else { return };
                match hx {
                    -1 => b[0] = p[0],
                    1 => b[2] = p[0],
                    _ => {}
                }
                match vy {
                    -1 => b[1] = p[1],
                    1 => b[3] = p[1],
                    _ => {}
                }
            }
            Geom::Polygon(pts) => {
                if let Some(v) = pts.get_mut(i) {
                    *v = p;
                }
            }
        }
    }

    /// Tidy a shape after an edit on a `w`×`h` image: order the box, snap a
    /// rectangle to whole pixels inside the image, drop repeated polygon
    /// vertices. Returns false when nothing usable is left (empty box,
    /// fewer than three vertices, or no overlap with the image).
    pub fn normalize(&mut self, w: usize, h: usize) -> bool {
        let (wf, hf) = (w as f32, h as f32);
        match &mut self.geom {
            Geom::Rect(b) => {
                let x0 = b[0].min(b[2]).round().clamp(0.0, wf);
                let x1 = b[0].max(b[2]).round().clamp(0.0, wf);
                let y0 = b[1].min(b[3]).round().clamp(0.0, hf);
                let y1 = b[1].max(b[3]).round().clamp(0.0, hf);
                *b = [x0, y0, x1, y1];
                x1 > x0 && y1 > y0
            }
            Geom::Ellipse(b) => {
                *b = [b[0].min(b[2]), b[1].min(b[3]), b[0].max(b[2]), b[1].max(b[3])];
                b[2] - b[0] >= 1.0 && b[3] - b[1] >= 1.0 && b[2] > 0.0 && b[3] > 0.0 && b[0] < wf && b[1] < hf
            }
            Geom::Polygon(pts) => {
                pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 0.5 && (a[1] - b[1]).abs() < 0.5);
                if pts.len() > 1 {
                    let (first, last) = (pts[0], pts[pts.len() - 1]);
                    if (first[0] - last[0]).abs() < 0.5 && (first[1] - last[1]).abs() < 0.5 {
                        pts.pop();
                    }
                }
                if pts.len() < 3 {
                    return false;
                }
                let b = MaskShape { geom: Geom::Polygon(pts.clone()), mode: RectMode::Include }.bbox();
                b[2] - b[0] >= 1.0 && b[3] - b[1] >= 1.0 && b[2] > 0.0 && b[3] > 0.0 && b[0] < wf && b[1] < hf
            }
        }
    }

    /// Call `f(x, y)` for every pixel of a `w`×`h` image inside the shape.
    pub fn for_each_pixel(&self, w: usize, h: usize, mut f: impl FnMut(usize, usize)) {
        let [x0, y0, x1, y1] = self.bbox();
        if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
            return;
        }
        let xs = x0.floor().max(0.0) as usize;
        let xe = (x1.ceil().max(0.0) as usize).min(w);
        let ys = y0.floor().max(0.0) as usize;
        let ye = (y1.ceil().max(0.0) as usize).min(h);
        for y in ys..ye {
            for x in xs..xe {
                if self.contains(x as f32 + 0.5, y as f32 + 0.5) {
                    f(x, y);
                }
            }
        }
    }

    pub fn label(&self) -> String {
        let [x0, y0, x1, y1] = self.bbox();
        match &self.geom {
            Geom::Rect(_) => format!(
                "rectangle x {x0:.0}..{x1:.0}, y {y0:.0}..{y1:.0} ({:.0}×{:.0})",
                x1 - x0,
                y1 - y0
            ),
            Geom::Ellipse(_) => format!(
                "ellipse center ({:.0}, {:.0}), {:.0}×{:.0}",
                (x0 + x1) * 0.5,
                (y0 + y1) * 0.5,
                x1 - x0,
                y1 - y0
            ),
            Geom::Polygon(pts) => format!(
                "polygon, {} points, x {x0:.0}..{x1:.0}, y {y0:.0}..{y1:.0}",
                pts.len()
            ),
        }
    }
}

fn parse_numbers(text: &str) -> Option<Vec<f32>> {
    text.split(',')
        .map(|s| s.trim().parse::<f32>().ok().filter(|v| v.is_finite()))
        .collect()
}

/// The shapes as JSON (the config file's `/mask@shapes`): a list of
/// `{"kind": "rectangle"|"ellipse"|"polygon", "mode": "include"|"exclude",
/// "points": [x0, y0, x1, y1, …]}` — the box of a rectangle or ellipse, the
/// vertices of a polygon.
pub fn shapes_to_json(shapes: &[MaskShape]) -> String {
    let list: Vec<serde_json::Value> = shapes
        .iter()
        .map(|s| {
            let points: Vec<f32> = match &s.geom {
                Geom::Rect(b) | Geom::Ellipse(b) => b.to_vec(),
                Geom::Polygon(pts) => pts.iter().flatten().copied().collect(),
            };
            serde_json::json!({"kind": s.kind(), "mode": s.mode.label(), "points": points})
        })
        .collect();
    serde_json::Value::Array(list).to_string()
}

pub fn shapes_from_json(text: &str) -> Result<Vec<MaskShape>> {
    let doc: serde_json::Value = serde_json::from_str(text).context("mask shapes: not JSON")?;
    let list = doc.as_array().context("mask shapes: not a list")?;
    let mut shapes = Vec::with_capacity(list.len());
    for item in list {
        let kind = item["kind"].as_str().unwrap_or_default();
        let mode = RectMode::parse(item["mode"].as_str().unwrap_or_default())
            .with_context(|| format!("mask shape: unknown mode {}", item["mode"]))?;
        let points: Vec<f32> = item["points"]
            .as_array()
            .context("mask shape: no points")?
            .iter()
            .map(|v| v.as_f64().map(|v| v as f32).filter(|v| v.is_finite()))
            .collect::<Option<_>>()
            .context("mask shape: points must be numbers")?;
        let geom = match (kind, points.as_slice()) {
            ("rectangle", &[x0, y0, x1, y1]) => Geom::Rect([x0, y0, x1, y1]),
            ("ellipse", &[x0, y0, x1, y1]) => Geom::Ellipse([x0, y0, x1, y1]),
            ("polygon", p) if p.len() >= 6 && p.len() % 2 == 0 => {
                Geom::Polygon(p.chunks(2).map(|c| [c[0], c[1]]).collect())
            }
            _ => bail!("mask shape: bad {kind:?} with {} coordinate(s)", points.len()),
        };
        shapes.push(MaskShape { geom, mode });
    }
    Ok(shapes)
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
    pub shapes: Vec<MaskShape>,
    /// Keep the pixels whose integrated (summed) intensity lies in `[lo, hi]`.
    pub range: Option<(f32, f32)>,
    pub file: Option<MaskFile>,
}

impl MaskSpec {
    /// No source: every pixel is used.
    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty() && self.range.is_none() && self.file.is_none()
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
        // Include shapes: the union of them is kept. Exclude shapes: each
        // one is dropped. Shapes are rasterized over their bounding box only.
        if self.shapes.iter().any(|s| s.mode == RectMode::Include) {
            let mut inside = Array2::from_elem((h, w), false);
            for s in self.shapes.iter().filter(|s| s.mode == RectMode::Include) {
                s.for_each_pixel(w, h, |x, y| inside[(y, x)] = true);
            }
            for (m, &i) in mask.iter_mut().zip(inside.iter()) {
                *m &= i;
            }
        }
        for s in self.shapes.iter().filter(|s| s.mode == RectMode::Exclude) {
            s.for_each_pixel(w, h, |x, y| mask[(y, x)] = false);
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
        for mode in [RectMode::Include, RectMode::Exclude] {
            let mut kinds: Vec<String> = Vec::new();
            for kind in ["rectangle", "ellipse", "polygon"] {
                let n = self.shapes.iter().filter(|s| s.mode == mode && s.kind() == kind).count();
                if n > 0 {
                    kinds.push(format!("{n} {} {kind}(s)", mode.label()));
                }
            }
            parts.extend(kinds);
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
            shapes: vec![
                MaskRect { left: 0, right: 3, top: 0, bottom: 4, mode: RectMode::Include }.into(),
                MaskRect { left: 1, right: 2, top: 1, bottom: 2, mode: RectMode::Exclude }.into(),
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
    fn ellipse_and_polygon_select_by_pixel_center() {
        let img = Array2::<f32>::zeros((20, 20));
        // A circle of radius 5 centered on (10, 10): π·25 ≈ 78.5 pixels.
        let circle = MaskShape::parse_ellipse("10,10,5,5", RectMode::Include).unwrap();
        let m = MaskSpec { shapes: vec![circle.clone()], ..Default::default() }.build(&img).unwrap().unwrap();
        assert!(m[(10, 10)] && m[(5, 9)] && m[(9, 14)]);
        assert!(!m[(5, 5)] && !m[(0, 0)] && !m[(10, 15)]);
        assert!((70..=88).contains(&count(&m)), "{} px", count(&m));
        // The same circle excluded: the complement.
        let hole = MaskShape { mode: RectMode::Exclude, ..circle };
        let c = MaskSpec { shapes: vec![hole], ..Default::default() }.build(&img).unwrap().unwrap();
        assert_eq!(count(&c) + count(&m), 400);

        // A right triangle (0,0) (10,0) (0,10): the pixels with x + y < 9.
        let tri = MaskShape::parse_polygon("0,0, 10,0, 0,10", RectMode::Include).unwrap();
        let t = MaskSpec { shapes: vec![tri], ..Default::default() }.build(&img).unwrap().unwrap();
        assert!(t[(0, 0)] && t[(0, 8)] && t[(8, 0)] && t[(4, 4)]);
        assert!(!t[(5, 5)] && !t[(0, 10)] && !t[(12, 3)]);
        assert_eq!(count(&t), 45);
        assert!(MaskShape::parse_polygon("0,0,10,0", RectMode::Include).is_none());
        assert!(MaskShape::parse_ellipse("5,5,0,2", RectMode::Include).is_none());
    }

    #[test]
    fn shapes_move_reshape_and_normalize() {
        let mut r: MaskShape = MaskRect { left: 2, right: 6, top: 3, bottom: 5, mode: RectMode::Include }.into();
        r.translate(1.4, -1.0);
        assert!(r.normalize(10, 10));
        assert_eq!(r.bbox(), [3.0, 2.0, 7.0, 4.0]);
        // Drag the right-bottom handle across the left-top corner: the box
        // is put back in order.
        r.drag_handle(7, [1.0, 0.0]);
        assert!(r.normalize(10, 10));
        assert_eq!(r.bbox(), [1.0, 0.0, 3.0, 2.0]);
        assert_eq!(r.handles().len(), 8);
        // Collapsed to nothing.
        r.drag_handle(7, [1.0, 0.0]);
        assert!(!r.normalize(10, 10));

        // A shape cannot be dragged out of reach: its center stays on the image.
        let mut e = MaskShape::parse_ellipse("5,5,2,2", RectMode::Exclude).unwrap();
        e.translate_within(100.0, -100.0, 10, 10);
        assert_eq!(e.bbox(), [8.0, -2.0, 12.0, 2.0]);
        assert!(e.normalize(10, 10));

        // Polygon: a vertex moves, the closing duplicate is dropped.
        let mut p = MaskShape::polygon(vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [4.0, 4.1], [0.0, 0.2]], RectMode::Include);
        assert!(p.normalize(10, 10));
        assert_eq!(p.handles().len(), 3);
        p.drag_handle(2, [8.0, 8.0]);
        assert_eq!(p.bbox(), [0.0, 0.0, 8.0, 8.0]);
        assert!(!MaskShape::polygon(vec![[0.0, 0.0], [4.0, 0.0]], RectMode::Include).normalize(10, 10));
    }

    #[test]
    fn shapes_json_round_trip() {
        let shapes = vec![
            MaskRect { left: 1, right: 9, top: 2, bottom: 8, mode: RectMode::Include }.into(),
            MaskShape::parse_ellipse("10.5,7.25,3,2", RectMode::Exclude).unwrap(),
            MaskShape::parse_polygon("0,0,10,0,0,10.5", RectMode::Include).unwrap(),
        ];
        let json = shapes_to_json(&shapes);
        assert_eq!(shapes_from_json(&json).unwrap(), shapes);
        assert!(shapes_from_json("[]").unwrap().is_empty());
        assert!(shapes_from_json(r#"[{"kind":"star","mode":"include","points":[1,2,3,4]}]"#).is_err());
        assert!(shapes_from_json(r#"[{"kind":"polygon","mode":"include","points":[1,2,3,4]}]"#).is_err());
        let spec = MaskSpec { shapes, ..Default::default() };
        assert_eq!(spec.describe(), "1 include rectangle(s), 1 include polygon(s), 1 exclude ellipse(s)");
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
