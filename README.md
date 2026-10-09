# Dehydration / Hydration Correction — beta (mbirtorch hsnt)

**This is the beta checkout** (`rust_dehydration_hydration_development`,
branch `mbirtorch_hsnt`) of the VENUS dehydration/hydration tool. It keeps the
production GUI (load a stack, compare corrected vs raw, profiles, export) but
replaces the correction engine: instead of the native Rust port of the
earlier least-squares NMF `hyper_denoise`, it runs the **new
dehydration/hydration of mbirtorch's `hsnt` package by Harel Dor** — the
maximum-likelihood factorization X = W·H of the attenuation under the Poisson
statistics of the counts, with the number of components estimated by
likelihood-ratio tests when it is not given. The production tool is
`../rust_dehydration_hydration` and is not affected.

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
cabouman/mbirtorch with the remote `harel` = harel55/mbirtorch, **branch
`hsnt`** checked out; `pixi install -e cuda` → torch 2.14 + CUDA 13, editable
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

To update the algorithm: `cd mbirtorch_hsnt && git fetch harel && git
checkout hsnt && git pull` (or check out another branch, e.g. upstream
`prerelease`, which carries the same package squashed in) — no rebuild of this
program is needed. The **ℹ mbirtorch** button shows the commit that is
checked out.

## Workflow (same as the notebook)

1. **Open Folder…** / **Open Files…** / **🕒 Recent** / **🔍 Run number…**
   (`-r/--run-number N`, NeXus lookup: detector, image folder and detector
   offset; a Timepix run asks for raw or autoreduce data, autoreduce being
   the loadable one). Folders and TIFF / `.npy` files can be dragged onto the
   window. Files load in parallel; NaN/Inf pixels are zeroed. When the file
   (or folder) names carry a run number (`…_Run_<N>_…`), the detector
   offset is read from that run's NeXus file, unless `-t/--offset` gave one.
2. **Raw data** view — slide through the images next to the integrated image.
3. **Correction parameters** (left panel) — the options of
   `mbirtorch-hsnt denoise`:
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
4. **Mask** (left panel, optional) — restrict the solve to part of the
   image: **✏ Draw rectangles** on the integrated image (right pane of the
   Raw data view; *include* = keep only the pixels inside any include
   rectangle, *exclude* = drop the pixels inside), keep the pixels whose
   **integrated value** lies in a range (e.g. leave out the open beam
   around the sample), and/or **📂 Load mask…** (TIFF or `.npy`, nonzero =
   selected, read with the stack's detector orientation — e.g. from the
   Hyperspectral Masker). The sources combine (file ∧ range ∧ rectangles);
   the excluded pixels are tinted dark red on the panes, the count of
   selected pixels is shown, **💾 Save mask…** writes the mask as an 8-bit
   TIFF. Only the selected pixels are written to the exchange file (as a
   pixels × bins table) and solved; the other pixels keep their raw values
   in the corrected stack. Note: on a masked run mbirtorch cannot pool
   neighbouring pixels in its rank estimate, so give the rank when the
   estimate looks low. The mask is saved with the settings (⚙ Config).
5. **▶ Perform correction** — runs the CLI on a background thread; the
   progress bar follows the CLI's log (loading → rank estimate → solve →
   outputs → read-back), **✖ Cancel** kills the Python process. **⚡ Preview**
   runs on 2×2-binned pixels. **📜 Log** (toolbar, or next to the progress
   bar) shows everything the CLI printed; it opens by itself when a run
   fails.
6. **Result** section — rank (and how it was obtained), solve mode / steps /
   time / loss, reduced chi-square with the CLI's verdict (or the relative
   residual when no dose was given), the CLI's warnings. **📈 Diagnostics**
   shows the component spectra and maps plots, the data checks and the
   memory plan.
7. **Corrected vs raw** and **Profiles** views — unchanged from production
   (shared contrast, difference pane, region + single-pixel profiles,
   TOF / wavelength axes from `*_Spectra.txt`, detector offset, CSV export).
   The profile series have fixed colors — uncorrected orange ✕, corrected
   blue ● — and the marker glyph is in the legend text.
8. **💾 Export corrected images…** — 32-bit float TIFFs (input names kept) in
   `dehydrated_hydrated_<input-folder>[_N]`, plus a copy of the input
   folder's `*_Spectra.txt` (so the TOF axis follows the corrected stack),
   `correction_config.json` (parameters, CLI arguments, mbirtorch commit, the
   fit summary) and the run's by-products: `hsnt_report.json`,
   `hsnt_dehydrated.h5` (`subspace_data` maps, `subspace_basis` spectra,
   `mean_pixel_spectrum`), `hsnt_spectra.png`, `hsnt_maps.png`,
   `hsnt_log.txt`, the component maps as images `hsnt_map_<i>.tif`
   (float32, 0 outside the mask) and, for a masked run, `mask.tif`.

**⚙ Config** saves / loads the settings as an HDF5 file (format version 2:
`/correction` holds `input_type`, `rank`, `max_rank`, `spectra`, `dose`,
`device`, `mode`, `max_steps`, `rel_tol`, `max_passes`, `compile`). A
version-1 file of the production tool still loads: its `dataset_type`
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
half-open pixel bounds in the oriented frame) and `--mask-range LO:HI` on
the integrated image. For scripts written against the
production tool, `--materials N` works unchanged (`--rank` is the hsnt
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

The portal entry is **"Dehydration / Hydration Correction (beta, mbirtorch
hsnt)"** in the Beta category of the unified launcher.

## Implementation notes

- `src/hsnt_cli.rs`: parameters ↔ CLI arguments, exchange HDF5 writer /
  reader (streamed by blocks of rows; a masked stack goes as a (pixels ×
  bins) table and comes back into the raw frames), the subprocess with live
  log relay, stage detection from the log lines, cancel (kill), the report
  parser, the component maps read from the dehydrated file.
- `src/mask.rs`: the mask definition (rectangles, integrated-value range,
  mask file) and its evaluation, binning for previews, TIFF read/write with
  the detector orientation.
- `src/correction.rs`: the GUI-independent run (binning for previews,
  hand-off, read-back); `start_correction` streams `Progress`, `Log` and
  `Done` messages to the app.
- `src/export.rs`: provenance JSON (serde_json) + the artifacts;
  `src/config.rs`: format version 2 with version-1 fallback.
- The native NMF modules of production (`hsnt.rs`, `nmf.rs`, `linalg.rs`,
  `examples/cross_check.rs`) are gone from this branch.
- TIFF frames are oriented on load per detector (shared
  `rust_detector_orientation` crate) and written back in the on-disk
  orientation on export; the orientation is irrelevant to the algorithm
  (per-pixel spectra), and the exchange file carries the oriented frames.
- Light/dark theme preference is shared with the other VENUS rust tools
  (`~/.config/venus_rust_tools/theme`).
