# Gait Analysis

This module turns the released 3D trajectories and a Keypoint-MoSeq syllable
bundle into a per-stride table and the six canonical release figures:

```
fig_main_canonical.png           Sheppard-style 3-panel speed-dependent gait
fig_main_traces.png              lateral postural curves + example bout traces
fig_individual_diffs.png         gait metrics vs spinal arc length per mouse
fig_syll_name_usage.png          per-session per-syllable-name frame usage
fig_syll_name_fingerprint.png    z-scored gait metrics per syllable name
fig_syll_name_speed_violins.png  stride-speed distribution per syllable name
```

All commands are scheduler-agnostic and read from `$QDMOUSE4M_RELEASE_ROOT`
or paths supplied on the command line.

## Data Source

```text
$QDMOUSE4M_RELEASE_ROOT/sessions/session_<session_id>/labels/keypoint_annotations.npz
```

The gait pipeline reads `keypoints_3d`, `valid_3d`,
`invalid_jump_timepoint` (if present), `keypoint_names`, and
`frame_index` (if present). Invalid 3D cells are masked before stride
detection, so downstream metrics use only trustworthy 3D positions.

The figure step additionally reads a Keypoint-MoSeq syllable bundle
(directory of `session_<id>_kpms_syllables.npz` files plus
`syllable_annotations.json`) produced by `qdmouse4m kpms export`.

## Recommended Release Config

```text
configs/gait.yaml
```

Defaults reproduce the public-release gait pipeline. The code masks
`valid_3d=False` cells before stride detection by default; downstream users
should trust the release validity flag rather than treating finite coordinates
as automatically trustworthy.

```text
animal_speed_min_mm_s: 50
stance_vel_hi_mm_s:    35
swing_vel_lo_mm_s:     120
min_stride_dur_s:      0.10
max_stride_dur_s:      1.20
max_nan_fraction:      0.10
mask_invalid_3d:       true
drop_first_last_per_bout: true
straight_gait omega:   <= 40 deg/s
```

Override any value in your own YAML.

## One-Step Wrapper

```bash
export QDMOUSE4M_RELEASE_ROOT=/path/to/qdmouse4m_release_neurips_v1

bash scripts/run_gait.sh \
  outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/syllable_bundle \
  outputs/gait \
  configs/gait_mouse_map.csv   # optional
```

Arguments:

```text
$1  syllable bundle directory  (qdmouse4m kpms export output)
$2  gait output directory   (default: outputs/gait)
$3  mouse-map CSV              (optional; enables fig_individual_diffs colours)
```

If the syllable bundle directory contains a `syllable_annotations.json`, it is
used. Otherwise the wrapper falls back to the syllable annotations defined in
`$KPMS_CONFIG` (default `configs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1.yaml`).

## Manual Steps

Build a per-stride CSV from the release:

```bash
qdmouse4m gait strides \
  --config configs/gait.yaml \
  --release-root "$QDMOUSE4M_RELEASE_ROOT" \
  --output-dir outputs/gait
```

Output: `outputs/gait/per_stride.csv` — one row per accepted stride with
the paper-faithful columns `stride_speed_cm_s`, `limb_duty_factor`,
`stride_length_mm`, `step_length_mm`, `step_width_mm`, `step_height_mean_mm`,
`stride_freq_hz`, `temporal_symmetry`, `body_length_mm`, `spine_angle_deg`,
`lat_disp_amp_ttip`, `lat_disp_phase_ttip`, plus the `straight_gait` flag
(|omega| <= straight-gait threshold).

Render the six figures:

```bash
qdmouse4m gait figures \
  --config configs/gait.yaml \
  --release-root "$QDMOUSE4M_RELEASE_ROOT" \
  --per-stride-csv outputs/gait/per_stride.csv \
  --syllable-bundle-dir outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/syllable_bundle \
  --annotations-json outputs/kpms/nst50_ds2_ld15_k1e06_seed1_seed1/syllable_bundle/syllable_annotations.json \
  --mouse-map configs/gait_mouse_map.csv \
  --output-dir outputs/gait/figures \
  --csv-dir outputs/gait
```

If `--annotations-json` is omitted, the figures step looks for
`syllable_annotations.json` inside `--syllable-bundle-dir`, then falls back to
the `syllable_annotations:` block of the `--config` YAML.

## Mouse Map Format

`fig_individual_diffs` groups sessions by mouse. Provide an optional CSV with
columns `session_id,mouse,trial`:

```csv
session_id,mouse,trial
20251030111548-808469,QDB03.1,1
20251030115501-797142,QDB03.2,1
...
```

When `--mouse-map` is omitted, each session is treated as a unique animal.

## Outputs

```text
outputs/gait/
  per_stride.csv                          one row per accepted stride
  per_stride_with_syllable_name.csv       strides + dominant KPMS syllable + name
  per_syllable_name_usage.csv             per-session per-syllable-name frame usage
  per_session_individual_diffs.csv        per-session medians used by fig_individual_diffs
  figures/
    fig_main_canonical.png
    fig_main_traces.png
    fig_individual_diffs.png
    fig_syll_name_usage.png
    fig_syll_name_fingerprint.png
    fig_syll_name_speed_violins.png
```
