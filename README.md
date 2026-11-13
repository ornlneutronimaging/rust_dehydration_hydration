# Dehydration / Hydration Correction (mbirtorch hsnt)

The VENUS dehydration/hydration tool (load a stack, compare corrected vs raw,
profiles, export). The correction is the **dehydration/hydration of
mbirtorch's `hsnt` package by Harel Dor** — the
maximum-likelihood factorization X = W·H of the attenuation under the Poisson
statistics of the counts, with the number of components estimated by
likelihood-ratio tests when it is not given. It replaced the native Rust port
of the earlier least-squares NMF `hyper_denoise` (last commit of that tool:
`81579bd`).

**Production and beta.** The same repository is checked out twice:

| Checkout | Branch | Portal entry | Log file |
|---|---|---|---|
| `rust_dehydration_hydration` | `main` | Dehydration / Hydration Correction | `dehydration_hydration_<user>.log` |
| `rust_dehydration_hydration_development` | `mbirtorch_hsnt` | Dehydration / Hydration Correction (beta), Beta category | `dehydration_hydration_beta_<user>.log` |

New work goes into the beta checkout and reaches production by merging its
branch into `main`. A binary built in the `_development` checkout labels
itself "beta" (window title, About, provenance, log file) — see
`src/variant.rs`; the sources are otherwise the same.

Algorithm reference: M. S. N. Chowdhury, D. Yang, S. Tang,
S. V. Venkatakrishnan, H. Z. Bilheux, G. T. Buzzard, and C. A. Bouman,
"Fast Hyperspectral Neutron Tomography," *IEEE Transactions on Computational
Imaging*, vol. 11, pp. 663–677, 2025.
[doi:10.1109/TCI.2025.3567854](https://doi.org/10.1109/TCI.2025.3567854) —
the estimator here is Harel Dor's maximum-likelihood fit (mbirtorch
`docs/source/usr_hsnt.rst`), not the paper's NMF.

## How the correction runs

The correction is **not computed in Rust**. The loaded (and detector-oriented)
stack is written to a scratch HDF5 file in the hsnt layout (`data` of shape
rows × cols × images, float32, the image index being the spectral axis),
then

```bash
python -m mbirtorch.hsnt denoise <scratch>/input.h5 -o <scratch>/out \
    --input-type transmission --rank auto --max-rank 6 --spectra mle \
    --device auto --mode auto --max-steps 1000 --rel-tol 1e-8 --max-passes 5 \
    --compile auto [--dose D] -v
```

runs in the pixi environment of the sibling checkout
**`/SNS/VENUS/shared/software/git/mbirtorch_hsnt`** (clone of
cabouman/mbirtorch, **`main` checked out** — v0.1.2, which carries the hsnt
package since its squash on 2026-10-06; the remote `harel` =
harel55/mbirtorch, where the package was developed on branch `hsnt`, is kept
for reference; `pixi install -e cuda` → torch 2.14 + CUDA 13, editable
mbirtorch, so the checked-out branch is what runs). `<stem>_denoised.h5` is
read back into frames; the dehydrated file (maps + spectra), the JSON report,
the two PNG plots and the log are kept in memory and written into the export
folder next to the TIFFs. The scratch folder is removed after the run.

| Environment variable | Meaning |
|---|---|
| `DEHY_HSNT_PYTHON` | another interpreter with mbirtorch (default `mbirtorch_hsnt/.pixi/envs/cuda/bin/python`) |
| `DEHY_HSNT_SCRATCH` | folder for the exchange files (default: the system temporary folder, `$TMPDIR` or `/tmp`) |

The exchange file is as large as the stack in memory (512 × 512 × 5000
images ≈ 5 GB), and the denoised file the CLI writes is as large again.

Two safety nets around the subprocess: when the mode is `auto` and the full
solve runs out of GPU memory (the CLI's memory plan cannot know what other
programs hold on the GPU), the run is repeated in **stream mode**
automatically (noted in the log, the result panel and the provenance); and
the Python process is given a parent-death signal, so a killed GUI never
leaves a solve holding the GPU's memory. A 512 × 512 × 2786 stack needs about
28 GB for the compiled full solve on an A100 40 GB; with NeCTAR or another
correction on the same GPU, pick another `cuda:N` or let the retry stream.

To update the algorithm: `cd mbirtorch_hsnt && git pull origin main` (or
check out another branch, e.g. upstream `prerelease`, or `harel/hsnt` for
work not yet merged) — no rebuild of this program is needed. The
**ℹ mbirtorch** button shows the commit that is checked out. The hsnt
package of v0.1.2 (2026-10-09) is the `hsnt` branch at edb0bcb with
docstring edits only; its command line, solvers and outputs are the same.

## Workflow

The window follows the workflow in five steps. The **step bar** at the top
shows them in order, each with its state (a green ✔ once done, a line of
status under the title); the current step is highlighted and the **left
panel** holds its controls only. Click a step, or use **◀ Back** /
**Next ▶** at the bottom of the panel, to move; a step that is not possible
yet (no data, no result) is greyed out and says what is missing. The toolbar
under the step bar only sets what the image panes show: view, image,
contrast, colormap, zoom.

**? Tutorial** (right end of the step bar) opens an illustrated, step-by-step
tutorial in a window of its own — one page per step, with a drawing and the
instructions — that can stay open next to the tool while the steps are
followed (`dehydration_hydration --instructions` shows it alone).

1. **Load data** — **Open Folder…** / **Open Files…** / **🕒 Recent** / **🔍 Run number…**
   (`-r/--run-number N`, NeXus lookup: detector, image folder and detector
   offset; a Timepix run asks for raw or autoreduce data, autoreduce being
   the loadable one). Folders and TIFF / `.npy` files can be dragged onto the
   window. Files load in parallel; NaN/Inf pixels are zeroed. When the file
   (or folder) names carry a run number (`…_Run_<N>_…`), the detector
   offset is read from that run's NeXus file, unless `-t/--offset` gave one.
   The panes show the raw images next to the integrated image.
   **📂 Load settings from HDF5…** re-applies a saved mask and parameters.
2. **Mask** (optional) — restrict the solve to part of the image. Shapes
   are drawn on either pane — the raw image (use the Image slider to
   pick the frame that shows the feature best) or the integrated image;
   a shape appears on both:
   - **Rectangle** / **Ellipse**: drag. **Polygon**: click the points, then
     double-click, press Enter or click the first point to close it (Esc
     cancels).
   - **New shapes: include / exclude** — *include* keeps only the pixels
     inside the include shapes (inside any of them), *exclude* drops the
     pixels inside.
   - After a shape is drawn the tool goes back to **Select / move**: click
     a shape to select it, **drag it to move it**, drag its handles to
     reshape it (the box of a rectangle or ellipse, the points of a
     polygon).
   - **Switch include ↔ exclude at any time**: click the shape's colored
     mode button in the list, or right-click the shape on the image. ✖ in
     the list, the Delete key or the right-click menu removes it.

   The panel has one framed block per way of defining the mask, under a
   summary of how much of the image will be corrected:
   - **Draw shapes** — above.
   - **Select by brightness** — keep only the pixels whose value in the
     integrated image (the sum of all the images, right pane) lies between
     a **Minimum** and a **Maximum**: raise the minimum to drop the dark
     surroundings, lower the maximum to drop the open beam around the
     sample. The status bar gives the value under the cursor.
   - **Use a mask image** — **📂 Load mask image…** (TIFF or `.npy`,
     nonzero = kept, read with the stack's detector orientation — e.g. from
     the Sample / Hyperspectral Masker).

   The parts in use combine (image ∧ brightness ∧ shapes); the pixels left
   out are shaded dark red on the panes, **💾 Save mask as image…** writes
   the resulting mask as an 8-bit TIFF, **Clear the mask** goes back to all
   pixels. Only the selected
   pixels are written to the exchange file (as a pixels × bins table) and
   solved; the other pixels keep their raw values in the corrected stack.
   Note: on a masked run mbirtorch cannot pool neighbouring pixels in its
   rank estimate, so give the rank when the estimate looks low. The mask is
   saved with the settings.
3. **Correct** — the options of `mbirtorch-hsnt denoise`:
   - **Input type** — `transmission` (default: the normalized stacks this tool
     loads), `attenuation` (= −log(transmission)) or `auto` (inferred from the
     values; fails when non-negative values above 1.05 cannot be told apart).
   - **Number of materials** — `auto` (estimated from the data by
     likelihood-ratio tests, at full resolution and on pooled pixels;
     **Max materials** bounds the search, default 6) or `fixed` N. This is
     hsnt's *rank*: the fitted components span the materials' spectra.
   - **Spectra** — `mle` (default), `unconstrained` (removes a low-dose bias,
     worth it with many pixels), `support` (per-pixel component selection,
     zeroes the background of the maps; **needs the dose**).
   - **Dose known** — open-beam counts per pixel and bin. When given, the fit
     reports a **reduced chi-square** against the Poisson noise (≈1 = at the
     noise level).
   - **Device** — `auto` (first CUDA GPU, else CPU), `cpu`, `cuda:N`.
   - **Advanced (solver)** — mode `auto|full|stream`, max steps, rel. tol.,
     max passes (stream mode), compile `auto|on|off`.

   **▶ Perform correction** runs the CLI on a background thread; the
   progress bar follows the CLI's log (loading → rank estimate → solve →
   outputs → read-back), **✖ Cancel** kills the Python process. **⚡ Preview**
   runs on 2×2-binned pixels. **📜 Log** (toolbar, or next to the progress
   bar) shows everything the CLI printed; it opens by itself when a run
   fails. When the run is done the window moves on to the next step.
4. **Review** — the result: rank (and how it was obtained), solve mode /
   steps / time / loss, reduced chi-square with the CLI's verdict (or the
   relative residual when no dose was given), the CLI's warnings (the
   warnings line is a link: one click opens the log, the warnings first). **Look
   at** switches the panes between **Corrected vs raw** (right pane: raw or
   the difference), **Profiles** (region + single-pixel profiles, TOF /
   wavelength axes from `*_Spectra.txt`, detector offset, CSV export; the
   series have fixed colors — uncorrected orange ×, corrected blue ● — and
   the marker glyph is in the legend text) and the raw data.
   **📈 Diagnostics** shows the component spectra and maps plots, the data
   checks and the memory plan.
5. **Export** — **💾 Export corrected images…** writes 32-bit float TIFFs
   (input names kept) in
   `dehydrated_hydrated_<input-folder>[_N]`, plus a copy of the input
   folder's `*_Spectra.txt` (so the TOF axis follows the corrected stack),
   `correction_config.json` (parameters, CLI arguments, mbirtorch commit, the
   fit summary) and the run's by-products: `hsnt_report.json`,
   `hsnt_dehydrated.h5` (`subspace_data` maps, `subspace_basis` spectra,
   `mean_pixel_spectrum`), `hsnt_spectra.png`, `hsnt_maps.png`,
   `hsnt_log.txt`, the component maps as images `hsnt_map_<i>.tif`
   (float32, 0 outside the mask) and, for a masked run, `mask.tif`.
   **💾 Save settings to HDF5…** is here too.

The settings file (saved in the Export step, loaded in the Load data step)
is HDF5, format version 2:
`/correction` holds `input_type`, `rank`, `max_rank`, `spectra`, `dose`,
`device`, `mode`, `max_steps`, `rel_tol`, `max_passes`, `compile`; `/mask`
holds the range, the mask file path and the drawn shapes (`@shapes`, a JSON
list of rectangles, ellipses and polygons; the rectangles are also written
as the `rects` table older versions read). A
version-1 file of the former NMF tool still loads: its `dataset_type`
becomes the input type and its `num_materials` the (fixed) rank.

## Headless batch mode

```bash
dehydration_hydration /SNS/VENUS/IPTS-XXXX/.../Run_YYYY \
    --run --output /path/to/output \
    --input-type transmission --materials auto --spectra mle [--dose 50] [--device cuda:1]
```

Same load → correct → export pipeline without a window (the CLI's log is
relayed on stderr, indented; the created folder is printed on stdout).
`--bin N` runs spatially binned, `--run-number N` can replace the INPUT path,
`--detector` forces the orientation. The mask: `--mask FILE`,
`--mask-include x0,y0,x1,y1` / `--mask-exclude x0,y0,x1,y1` (repeatable,
half-open pixel bounds in the oriented frame), their ellipse and polygon
forms `--mask-include-ellipse cx,cy,rx,ry`, `--mask-include-polygon
x0,y0,x1,y1,x2,y2,…` (and `--mask-exclude-…`), and `--mask-range LO:HI` on
the integrated image. For scripts written against the
former NMF tool, `--materials N` works unchanged (`--rank` is the hsnt
synonym), `--dataset-type` means
`--input-type`, `--max-iter` means `--max-steps`, and `--beta-loss` /
`--safety-factor` are accepted and ignored with a warning — so the Workflow
Runner can be pointed at this binary with `WORKFLOW_DEHY_BIN`.

## Build & run

```bash
cargo build --release
# binary: target/release/dehydration_hydration

# or, rebuild-if-needed and run (needs a graphical session, e.g. ThinLinc):
./launch_dehydration_hydration.sh [folder-or-files...]
```

```bash
cargo test                 # IO / parsing unit tests, no Python needed
cargo test -- --ignored    # + one real run through the mbirtorch environment
```

The portal entries are **"Dehydration / Hydration Correction"** (production)
and **"Dehydration / Hydration Correction (beta)"** in the Beta category of
the unified launcher.

## Log

Every session appends what the user did (files opened, runs located, mask
and correction parameters, the `mbirtorch.hsnt` command launched and how it
ended, exports, errors) to the shared VENUS log
`/SNS/VENUS/shared/log/dehydration_hydration_<user>.log` (the beta build:
`dehydration_hydration_beta_<user>.log`, its own file). One
timestamped line per event; the file is readable in the Log Browser
(`rust_log_browser`), like the logs of the other VENUS tools.

## Implementation notes

- `src/hsnt_cli.rs`: parameters ↔ CLI arguments, exchange HDF5 writer /
  reader (streamed by blocks of rows; a masked stack goes as a (pixels ×
  bins) table and comes back into the raw frames), the subprocess with live
  log relay, stage detection from the log lines, cancel (kill), the report
  parser, the component maps read from the dehydrated file.
- `src/help.rs`: the illustrated tutorial window (drawings painted with
  egui), a second process of the binary (`--instructions`).
- `src/mask.rs`: the mask definition (shapes — rectangles, ellipses, polygons —, integrated-value range,
  mask file) and its evaluation, binning for previews, TIFF read/write with
  the detector orientation.
- `src/correction.rs`: the GUI-independent run (binning for previews,
  hand-off, read-back); `start_correction` streams `Progress`, `Log` and
  `Done` messages to the app.
- `src/export.rs`: provenance JSON (serde_json) + the artifacts;
  `src/config.rs`: format version 2 with version-1 fallback.
- The native NMF modules of the former tool (`hsnt.rs`, `nmf.rs`,
  `linalg.rs`, `examples/cross_check.rs`) are gone.
- TIFF frames are oriented on load per detector (shared
  `rust_detector_orientation` crate) and written back in the on-disk
  orientation on export; the orientation is irrelevant to the algorithm
  (per-pixel spectra), and the exchange file carries the oriented frames.
- Light/dark theme preference is shared with the other VENUS rust tools
  (`~/.config/venus_rust_tools/theme`).
