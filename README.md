# Dehydration / Hydration Correction

Native GUI (Rust, [egui](https://github.com/emilk/egui)) that reproduces the
VENUS **dehydration_hydration** notebook
(`python_notebooks/notebooks/dehydration_hydration.ipynb`): denoise a stack of
neutron images with the NMF **dehydrate / rehydrate** algorithm of
`mbirjax.hsnt.hyper_denoise`, compare the corrected and raw data, and export
the corrected stack as 32-bit float TIFFs.

Algorithm reference: M. S. N. Chowdhury, D. Yang, S. Tang,
S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman,
"Fast Hyperspectral Neutron Tomography," *IEEE Transactions on Computational
Imaging*, vol. 11, pp. 663–677, 2025.
[doi:10.1109/TCI.2025.3567854](https://doi.org/10.1109/TCI.2025.3567854) —
[mbirjax documentation](https://mbirjax.readthedocs.io/en/latest/usr_hsnt.html).

## Workflow (same as the notebook)

1. **Open Folder…** — select the folder containing the TIFF images to correct
   (when the folder has none, its subfolders are searched, like the notebook).
2. **Raw data** view — slide through the images next to the integrated (sum)
   image.
3. **Correction parameters** (left panel):
   - **Dataset type** — `attenuation` or `transmission`, where
     attenuation = −log(transmission). Default `attenuation`.
   - **Number of materials** — how many different materials the data set
     contains (1–10, default 2). The NMF subspace dimension is
     2 × this number (safety factor 2).
   - **Beta loss** — `frobenius` (coordinate-descent solver) or
     `kullback-leibler` (multiplicative-update solver). Default `frobenius`.
   - **Max iterations** — NMF solver cap (50–1000, default 300).
4. **▶ Perform correction** — runs on a background thread with progress and a
   cancel button; the image index is treated as the spectral axis, every
   pixel spectrum is projected onto a low-dimensional non-negative subspace
   (dehydration) and multiplied back (rehydration), discarding the noise
   outside the subspace.
5. **Corrected vs raw** view — side-by-side comparison with shared contrast.
6. **Profiles** view — drag a region on the integrated corrected image and
   compare the mean-intensity profiles of the corrected and uncorrected
   stacks across the image index. The region can be **moved** (drag inside
   it) and **resized** (drag one of its 8 handles) without redrawing; the
   plot shows the image index / intensity under the cursor in its corner and
   the y-axis can be toggled between linear and log scale.
7. **💾 Export corrected images…** — pick an output folder; the corrected
   stack is written as 32-bit float TIFFs (input file names kept) into a new
   subfolder `<input-folder>_dehydration_hydration_corrected` (suffixed `_1`,
   `_2`, … when it already exists).

The **ℹ mbirjax** button (top-right) shows the algorithm provenance: the
mbirjax version the implementation is a port of (0.7.2, tracked as a
constant in `src/app.rs` — bump it after diffing the denoising functions of
`mbirjax/hsnt.py` against the newer release) and the paper reference.

## Build & run

```bash
cargo build --release
# binary: target/release/dehydration_hydration

# or, rebuild-if-needed and run (needs a graphical session, e.g. ThinLinc):
./launch_dehydration_hydration.sh [folder-or-files...]
```

```bash
cargo test    # algorithm + IO unit tests, no display needed
```

## Implementation notes

- The NMF (NNDSVD initialization, coordinate-descent solver for the
  Frobenius loss, multiplicative updates for Kullback-Leibler) is a native
  Rust port of the scikit-learn `non_negative_factorization` path used by
  `mbirjax.hsnt`, parallelized with rayon — no Python, BLAS, or CUDA
  dependency.
- Large stacks are processed in batches of 2²⁷ elements exactly like the
  Python code (per-batch basis estimation, basis merging, then a fixed-basis
  projection of every batch).
- Deviations from the Python original, all inconsequential in practice: the
  randomized SVD and the batch shuffling are seeded (runs are reproducible
  where the Python ones are not), the automatic material-count estimation is
  not ported (the GUI always provides the number of materials), and the
  final reconstruction is computed in f64 before the f32 cast.
- TIFF frames are transposed on load for display (VENUS detector
  orientation, same convention as rust_roi_selector / rust_tiff_viewer) and
  transposed back on export, so exported files align with the input files on
  disk.
- Light/dark theme preference is shared with the other VENUS rust tools
  (`~/.config/venus_rust_tools/theme`).
