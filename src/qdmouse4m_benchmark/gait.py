"""Stride-level gait analysis from QDMouse4M release 3D trajectories.

Two subcommands:
  strides   build a per-stride CSV from release 3D coordinates
  figures   render the canonical gait figures from the per-stride CSV
            and a Keypoint-MoSeq syllable bundle

The pipeline assumes the public QDMouse4M release layout (sessions/<id>/labels/
keypoint_annotations.npz). Invalid 3D cells are masked before stride detection
and figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from .constants import KEYPOINTS  # noqa: E402

FPS_DEFAULT = 40.724

# Stride detection thresholds. Defaults reproduce the public release gait
# pipeline. Override any value via the YAML config (configs/gait.yaml).
DEFAULT_STRIDE_PARAMS: dict[str, Any] = {
    "animal_speed_min_mm_s": 50.0,
    "stance_vel_hi_mm_s": 35.0,
    "swing_vel_lo_mm_s": 120.0,
    "min_stance_frames": 3,
    "min_swing_frames": 2,
    "min_stride_dur_s": 0.10,
    "max_stride_dur_s": 1.20,
    "max_nan_fraction": 0.10,
    "drop_first_last_per_bout": True,
    "savgol_window": 7,
    "savgol_polyorder": 2,
}

# Straight-gait gate (Sheppard et al. 2022). Strides whose median |omega| is
# at or below this threshold are flagged `straight_gait=True` in the CSV.
DEFAULT_STRAIGHT_GAIT_OMEGA_DEG_S = 40.0

DEFAULT_SPEED_BINS_CM_S: list[tuple[float, float]] = [
    (5.0, 7.5), (7.5, 10.0), (10.0, 12.5), (12.5, 15.0),
    (15.0, 17.5), (17.5, 20.0), (20.0, 22.5), (22.5, 25.0),
    (25.0, 30.0), (30.0, 40.0),
]

# Sheppard et al. 2022 reference ranges for the supplementary distributions
# panel. Empty here; figures we ship don't reference it.
SHEPPARD_REF: dict[str, tuple[float, float]] = {}

DEFAULT_FIGURE_PARAMS: dict[str, Any] = {
    "min_strides_per_syllable_name": 20,
    "max_kp_jump_mm": 80.0,
    "example_window_seconds": 6.0,
    "example_session": None,
}

GAIT_METRICS: list[tuple[str, str]] = [
    ("stride_speed_cm_s", "stride speed (cm/s)"),
    ("limb_duty_factor", "duty factor"),
    ("stride_length_mm", "stride length (mm)"),
    ("step_length_mm", "step length (mm)"),
    ("step_width_mm", "step width (mm)"),
    ("step_height_mean_mm", "step height (mm)"),
    ("stride_freq_hz", "stride freq (Hz)"),
    ("temporal_symmetry", "temp. symmetry"),
    ("body_length_mm", "body length (mm)"),
    ("spine_angle_deg", "spine flex (°)"),
    ("lat_disp_amp_ttip", "tail-tip lat amp"),
    ("lat_disp_phase_ttip", "tail-tip lat phase"),
]

DEFAULT_RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def release_root_from_arg(value: Path | None) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    env_value = os.environ.get("QDMOUSE4M_RELEASE_ROOT")
    if not env_value:
        raise SystemExit("Set QDMOUSE4M_RELEASE_ROOT or pass --release-root.")
    return Path(env_value).expanduser().resolve()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Install PyYAML to load gait YAML configs."
        ) from exc
    with path.expanduser().open() as f:
        return yaml.safe_load(f) or {}


def merged_stride_params(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_STRIDE_PARAMS)
    out.update(dict(config.get("stride_detection") or {}))
    return out


def merged_figure_params(config: dict[str, Any]) -> dict[str, Any]:
    out = dict(DEFAULT_FIGURE_PARAMS)
    out.update(dict(config.get("figures") or {}))
    return out


def merged_speed_bins(config: dict[str, Any]) -> list[tuple[float, float]]:
    raw = config.get("speed_bins_cm_s")
    if not raw:
        return list(DEFAULT_SPEED_BINS_CM_S)
    return [(float(lo), float(hi)) for lo, hi in raw]


def merged_omega_threshold(config: dict[str, Any]) -> float:
    return float((config.get("straight_gait") or {}).get(
        "omega_threshold_deg_s", DEFAULT_STRAIGHT_GAIT_OMEGA_DEG_S
    ))


# ---------------------------------------------------------------------------
# Release I/O
# ---------------------------------------------------------------------------


def keypoint_annotations_path(release_root: Path, session_id: str) -> Path:
    return (
        release_root
        / "sessions"
        / f"session_{session_id}"
        / "labels"
        / "keypoint_annotations.npz"
    )


def load_session_order(release_root: Path, session_ids_file: Path | None) -> list[str]:
    if session_ids_file:
        ids = []
        for raw in session_ids_file.read_text().splitlines():
            line = raw.split("#", 1)[0].strip()
            if line:
                ids.append(line.replace("session_", ""))
        return ids
    sessions_csv = release_root / "sessions.csv"
    if sessions_csv.is_file():
        with sessions_csv.open(newline="") as f:
            return [row["session_id"] for row in csv.DictReader(f)]
    return sorted(
        p.name.replace("session_", "")
        for p in (release_root / "sessions").glob("session_*")
    )


def invalid_jump_mask(z: Any, shape_TK: tuple[int, int]) -> np.ndarray:
    if "invalid_jump_timepoint" not in z.files:
        return np.zeros(shape_TK, dtype=bool)
    raw = np.asarray(z["invalid_jump_timepoint"]).astype(bool)
    if raw.shape == shape_TK:
        return raw
    if raw.shape == (shape_TK[0],):
        return np.repeat(raw[:, None], shape_TK[1], axis=1)
    raise RuntimeError(
        f"Unexpected invalid_jump_timepoint shape {raw.shape}; expected {shape_TK} or ({shape_TK[0]},)."
    )


def load_session_3d(
    release_root: Path,
    session_id: str,
    *,
    mask_invalid: bool = True,
    mask_invalid_jump: bool = True,
) -> np.ndarray:
    """Load (T, 12, 3) coordinates for a session, masking invalid cells with NaN."""
    npz_path = keypoint_annotations_path(release_root, session_id)
    z = np.load(npz_path, allow_pickle=False)
    keypoints = [str(x) for x in z["keypoint_names"].astype(str).tolist()]
    if keypoints != KEYPOINTS:
        raise RuntimeError(f"Unexpected keypoint order in {npz_path}: {keypoints}")
    coords = z["keypoints_3d"].astype(np.float32, copy=True)
    if "valid_3d" in z.files:
        valid = z["valid_3d"].astype(bool, copy=False)
    else:
        valid = np.isfinite(coords).all(axis=2)
    if mask_invalid_jump:
        valid = valid & ~invalid_jump_mask(z, valid.shape)
    if mask_invalid:
        coords[~valid] = np.nan
    return coords


# ---------------------------------------------------------------------------
# Stride detection
# ---------------------------------------------------------------------------


KP_NECK = "back_top"
KP_SPINE = "back_middle"
KP_TAIL_BASE = "tail_base"
KP_TAIL_TIP = "tail_tip"
KP_NOSE_PROXY = "back_top"
HIND_L = "hindpaw_L"
HIND_R = "hindpaw_R"
PAWS_ALL = ("forepaw_L", "forepaw_R", HIND_L, HIND_R)


def smooth_axis0(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    if len(x) < win + 2:
        return x.copy()
    nm = np.isnan(x)
    if nm.any():
        flat = x.copy().reshape(len(x), -1)
        nflat = nm.reshape(len(x), -1)
        idx = np.arange(len(x))
        for c in range(flat.shape[1]):
            m = nflat[:, c]
            if m.any() and (~m).any():
                flat[m, c] = np.interp(idx[m], idx[~m], flat[~m, c])
        x_filled = flat.reshape(x.shape)
    else:
        x_filled = x
    out = savgol_filter(x_filled, win, poly, axis=0, mode="interp")
    out[nm] = np.nan
    return out


def velocity(x: np.ndarray, fs: float) -> np.ndarray:
    return np.gradient(x, axis=0) * fs


def _enforce_min_runs(arr: np.ndarray, value: int, min_len: int) -> np.ndarray:
    out = arr.copy()
    T = len(out)
    i = 0
    while i < T:
        if out[i] != value:
            i += 1
            continue
        j = i
        while j < T and out[j] == value:
            j += 1
        if j - i < min_len:
            out[i:j] = 1 - value
        i = j
    return out


def find_runs(arr: np.ndarray, value: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    T = len(arr)
    i = 0
    while i < T:
        if arr[i] != value:
            i += 1
            continue
        j = i
        while j < T and arr[j] == value:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def stance_swing_mask(
    paw_speed_abs: np.ndarray,
    stance_hi: float,
    swing_lo: float,
    min_stance: int,
    min_swing: int,
) -> np.ndarray:
    """Two-threshold hysteresis on absolute paw speed (mm/s). 1=stance, 0=swing."""
    T = len(paw_speed_abs)
    state = np.zeros(T, dtype=np.int8)
    cur = 0
    for t in range(T):
        v = paw_speed_abs[t]
        if np.isnan(v):
            state[t] = cur
            continue
        if cur == 0 and v < stance_hi:
            cur = 1
        elif cur == 1 and v > swing_lo:
            cur = 0
        state[t] = cur
    state = _enforce_min_runs(state, value=1, min_len=min_stance)
    state = _enforce_min_runs(state, value=0, min_len=min_swing)
    return state


def angular_velocity_deg_s(
    neck_xy: np.ndarray,
    tail_base_xy: np.ndarray,
    fs: float,
    smooth_win: int = 21,
    smooth_poly: int = 3,
) -> np.ndarray:
    """Body-axis angular velocity (deg/s), smoothed first-derivative."""
    v = neck_xy - tail_base_xy
    ang = np.arctan2(v[:, 1], v[:, 0])
    if np.isnan(ang).any():
        idx = np.arange(len(ang))
        m = ~np.isnan(ang)
        if m.any():
            ang = np.interp(idx, idx[m], ang[m])
    ang_u = np.unwrap(ang)
    win = min(smooth_win, len(ang_u) // 2 * 2 - 1)
    if win < smooth_poly + 2:
        return np.gradient(ang_u) * fs * (180.0 / np.pi)
    omega_rad_s = savgol_filter(
        ang_u, win, smooth_poly, deriv=1, delta=1.0 / fs, mode="interp"
    )
    return omega_rad_s * (180.0 / np.pi)


def project_local_axes(start_xy: np.ndarray, end_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = end_xy - start_xy
    n = np.linalg.norm(d)
    if not np.isfinite(n) or n < 1e-6:
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])
    f = d / n
    lat = np.array([-f[1], f[0]])
    return f, lat


def stride_max_displacement_phase(displacement: np.ndarray) -> tuple[float, float]:
    n = len(displacement)
    if n < 4 or not np.isfinite(displacement).all():
        return float("nan"), float("nan")
    t = np.linspace(0, 1, n)
    cs = CubicSpline(t, displacement, bc_type="natural")
    tt = np.linspace(0, 1, 200)
    yy = cs(tt)
    i = int(np.argmax(np.abs(yy)))
    return float(tt[i]), float(yy[i])


def analyze_session(
    sess_key: str,
    X: np.ndarray,
    fs: float,
    params: dict[str, Any],
    omega_threshold_deg_s: float,
) -> list[dict[str, Any]]:
    """Detect strides anchored on left-hindpaw foot-strikes; compute paper-faithful
    stride-level metrics. Returns a list of per-stride dicts.
    """
    name_to_i = {n: i for i, n in enumerate(KEYPOINTS)}
    Xs = smooth_axis0(X, params["savgol_window"], params["savgol_polyorder"])
    spine = Xs[:, name_to_i[KP_SPINE], :]
    neck = Xs[:, name_to_i[KP_NECK], :]
    tbase = Xs[:, name_to_i[KP_TAIL_BASE], :]
    ttip = Xs[:, name_to_i[KP_TAIL_TIP], :]
    nose = Xs[:, name_to_i[KP_NOSE_PROXY], :]

    animal_speed = np.linalg.norm(velocity(tbase[:, :2], fs), axis=1)
    omega = angular_velocity_deg_s(neck[:, :2], tbase[:, :2], fs)
    body_len = np.linalg.norm(neck - tbase, axis=1)

    paw_pos: dict[str, np.ndarray] = {}
    paw_speed: dict[str, np.ndarray] = {}
    for paw in PAWS_ALL:
        if paw not in name_to_i:
            continue
        p = Xs[:, name_to_i[paw], :]
        paw_pos[paw] = p
        paw_speed[paw] = np.linalg.norm(velocity(p, fs), axis=1)

    state_per_paw = {
        paw: stance_swing_mask(
            paw_speed[paw],
            params["stance_vel_hi_mm_s"],
            params["swing_vel_lo_mm_s"],
            params["min_stance_frames"],
            params["min_swing_frames"],
        )
        for paw in PAWS_ALL
        if paw in paw_speed
    }
    stance_runs_per_paw = {
        paw: find_runs(state_per_paw[paw], 1) for paw in state_per_paw
    }
    foot_strikes_per_paw = {
        paw: np.array([s for s, _ in r], dtype=int)
        for paw, r in stance_runs_per_paw.items()
    }

    runs_L_all = stance_runs_per_paw[HIND_L]
    lh_pairs: list[tuple[int, int, int, int]] = []
    for k in range(len(runs_L_all) - 1):
        s0, e0 = runs_L_all[k]
        s1, e1 = runs_L_all[k + 1]
        lh_pairs.append((s0, e0, s1, e1))

    accepted_idx: list[int] = []
    candidate_meta: list[dict[str, Any]] = []
    for k, (s0, e0, s1, e1) in enumerate(lh_pairs):
        stride_a = s0
        stride_b = s1
        T_seg = stride_b - stride_a
        if T_seg < 2:
            candidate_meta.append({"ok": False})
            continue
        dur = T_seg / fs
        ok = params["min_stride_dur_s"] <= dur <= params["max_stride_dur_s"]
        seg_nan = float("nan")
        mean_speed = float("nan")
        mean_omega = float("nan")
        if ok:
            seg_nan = float(np.isnan(X[stride_a:stride_b + 1, :, :]).any(axis=(1, 2)).mean())
            ok &= seg_nan <= params["max_nan_fraction"]
        if ok:
            mean_speed = float(np.nanmean(animal_speed[stride_a:stride_b + 1]))
            ok &= mean_speed >= params["animal_speed_min_mm_s"]
        if ok:
            mean_omega = float(np.nanmedian(omega[stride_a:stride_b + 1]))
        candidate_meta.append({
            "ok": ok, "stride_a": stride_a, "stride_b": stride_b,
            "s0": s0, "e0": e0, "s1": s1, "e1": e1,
            "dur": dur, "mean_speed": mean_speed, "mean_omega": mean_omega,
        })
        if ok:
            accepted_idx.append(k)

    if params["drop_first_last_per_bout"] and len(accepted_idx) >= 3:
        sequences: list[list[int]] = []
        cur = [accepted_idx[0]]
        for i in accepted_idx[1:]:
            if i == cur[-1] + 1:
                cur.append(i)
            else:
                sequences.append(cur)
                cur = [i]
        sequences.append(cur)
        kept: list[int] = []
        for seq in sequences:
            if len(seq) >= 3:
                kept.extend(seq[1:-1])
            elif len(seq) == 1:
                kept.extend(seq)
        accepted_idx = kept

    rows: list[dict[str, Any]] = []
    for k in accepted_idx:
        m = candidate_meta[k]
        stride_a = m["stride_a"]
        stride_b = m["stride_b"]
        s0 = m["s0"]
        T_seg = stride_b - stride_a
        dur = m["dur"]
        mean_speed = m["mean_speed"]
        mean_omega = m["mean_omega"]

        fwd, lat = project_local_axes(spine[stride_a, :2], spine[stride_b, :2])

        lh_xy0 = paw_pos[HIND_L][stride_a, :2]
        lh_xy1 = paw_pos[HIND_L][stride_b, :2]
        stride_length_mm = float(np.dot(lh_xy1 - lh_xy0, fwd))

        body_disp_mm = float(np.dot(spine[stride_b, :2] - spine[stride_a, :2], fwd))

        R_strikes = foot_strikes_per_paw[HIND_R]
        R_in = R_strikes[(R_strikes > s0) & (R_strikes <= stride_b)]
        if len(R_in):
            rh_strike_t = int(R_in[0])
            lh_at_s0 = paw_pos[HIND_L][s0, :2]
            rh_at_strike = paw_pos[HIND_R][rh_strike_t, :2]
            step_length_mm = float(np.dot(rh_at_strike - lh_at_s0, fwd))
            step_width_mm = float(np.dot(rh_at_strike - lh_at_s0, lat))
        else:
            rh_strike_t = -1
            step_length_mm = float("nan")
            step_width_mm = float("nan")

        T_use = max(T_seg, 1)
        duty_LH = float((state_per_paw[HIND_L][stride_a:stride_b] == 1).sum() / T_use)
        duty_RH = float((state_per_paw[HIND_R][stride_a:stride_b] == 1).sum() / T_use)
        limb_duty_factor = 0.5 * (duty_LH + duty_RH)
        denom = duty_LH + duty_RH
        temporal_symmetry = float((duty_LH - duty_RH) / denom) if denom > 1e-6 else float("nan")

        step_h_LH = float(
            np.nanmax(paw_pos[HIND_L][stride_a:stride_b + 1, 2])
            - np.nanmin(paw_pos[HIND_L][stride_a:stride_b + 1, 2])
        )
        step_h_RH = float(
            np.nanmax(paw_pos[HIND_R][stride_a:stride_b + 1, 2])
            - np.nanmin(paw_pos[HIND_R][stride_a:stride_b + 1, 2])
        )

        seg = slice(stride_a, stride_b + 1)
        body_len_seg = float(np.nanmean(body_len[seg]))
        if not np.isfinite(body_len_seg) or body_len_seg < 1e-3:
            continue

        rel_nose = nose[seg, :2] - spine[stride_a, :2]
        rel_ttip = ttip[seg, :2] - spine[stride_a, :2]
        rel_tbase = tbase[seg, :2] - spine[stride_a, :2]
        lat_nose = rel_nose @ lat
        lat_ttip = rel_ttip @ lat
        lat_tbase = rel_tbase @ lat

        phase_n, amp_n = stride_max_displacement_phase(lat_nose / body_len_seg)
        phase_t, amp_t = stride_max_displacement_phase(lat_ttip / body_len_seg)
        phase_tb, amp_tb = stride_max_displacement_phase(lat_tbase / body_len_seg)

        a_vec = spine[seg, :] - tbase[seg, :]
        b_vec = neck[seg, :] - spine[seg, :]
        cos_a = (a_vec * b_vec).sum(axis=1) / (
            np.linalg.norm(a_vec, axis=1) * np.linalg.norm(b_vec, axis=1) + 1e-9
        )
        spine_angle = float(np.nanmean(np.degrees(np.arccos(np.clip(cos_a, -1, 1)))))

        rows.append({
            "session": sess_key,
            "lh_pair_idx": int(k),
            "stride_a": int(stride_a),
            "stride_b": int(stride_b),
            "stride_dur_s": dur,
            "stride_freq_hz": 1.0 / dur,
            "stride_speed_mm_s": mean_speed,
            "stride_speed_cm_s": mean_speed / 10.0,
            "stride_omega_deg_s": mean_omega,
            "body_disp_mm": body_disp_mm,
            "stride_length_mm": stride_length_mm,
            "step_length_mm": step_length_mm,
            "step_width_mm": step_width_mm,
            "duty_LH": duty_LH,
            "duty_RH": duty_RH,
            "limb_duty_factor": limb_duty_factor,
            "temporal_symmetry": temporal_symmetry,
            "step_height_LH_mm": step_h_LH,
            "step_height_RH_mm": step_h_RH,
            "step_height_mean_mm": 0.5 * (step_h_LH + step_h_RH),
            "lat_disp_phase_nose": phase_n,
            "lat_disp_amp_nose": amp_n,
            "lat_disp_phase_ttip": phase_t,
            "lat_disp_amp_ttip": amp_t,
            "lat_disp_phase_tbase": phase_tb,
            "lat_disp_amp_tbase": amp_tb,
            "body_length_mm": body_len_seg,
            "spine_angle_deg": spine_angle,
            "rh_strike_t": int(rh_strike_t),
            "straight_gait": bool(abs(mean_omega) <= omega_threshold_deg_s),
        })
    return rows


# ---------------------------------------------------------------------------
# Mouse map (optional, for fig_individual_diffs)
# ---------------------------------------------------------------------------


def load_mouse_map(path: Path | None) -> dict[str, tuple[str, int]]:
    """Read a CSV mapping session_id -> (mouse, trial). Trial is optional."""
    if path is None:
        return {}
    out: dict[str, tuple[str, int]] = {}
    with path.expanduser().open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get("session_id") or row.get("session")
            mouse = row.get("mouse") or row.get("mouse_id")
            if not sid or not mouse:
                continue
            sid = sid.strip()
            mouse = mouse.strip()
            try:
                trial = int(row.get("trial") or 0)
            except ValueError:
                trial = 0
            key = sid if sid.startswith("session_") else f"session_{sid}"
            out[key] = (mouse, trial)
    return out


# ---------------------------------------------------------------------------
# strides subcommand
# ---------------------------------------------------------------------------


def cmd_strides(args: argparse.Namespace, config: dict[str, Any]) -> None:
    release_root = release_root_from_arg(args.release_root)
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = float(args.fps if args.fps is not None else (
        config.get("dataset", {}).get("fps", FPS_DEFAULT)
    ))
    params = merged_stride_params(config)
    omega_thr = merged_omega_threshold(config)
    mask_invalid, mask_invalid_jump = _resolve_mask_flags(args, config)

    session_ids = load_session_order(release_root, args.session_ids_file)

    all_rows: list[dict[str, Any]] = []
    for sid in session_ids:
        sess_key = f"session_{sid}"
        coords = load_session_3d(
            release_root, sid,
            mask_invalid=mask_invalid,
            mask_invalid_jump=mask_invalid_jump,
        )
        rows = analyze_session(sess_key, coords, fps, params, omega_thr)
        n_straight = sum(1 for r in rows if r["straight_gait"])
        print(f"[strides] {sess_key}  T={coords.shape[0]}  strides={len(rows)}  straight={n_straight}", flush=True)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_csv = out_dir / "per_stride.csv"
    df.to_csv(out_csv, index=False)
    print(json.dumps({
        "per_stride_csv": str(out_csv),
        "n_strides": int(len(df)),
        "n_straight_strides": int(df["straight_gait"].sum()) if "straight_gait" in df.columns else 0,
        "n_sessions": int(df["session"].nunique()) if len(df) else 0,
        "fps": fps,
        "stride_detection": params,
        "straight_gait_omega_deg_s": omega_thr,
    }, indent=2))


# ---------------------------------------------------------------------------
# Syllable bundle helpers
# ---------------------------------------------------------------------------


def _load_syllable_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    if "syllable" in z.files:
        syllable = z["syllable"].astype(np.int32)
    elif "syllable_id" in z.files:
        syllable = z["syllable_id"].astype(np.int32)
    else:
        raise KeyError(f"{path} has no syllable or syllable_id array")
    if "known" in z.files:
        known = z["known"].astype(bool)
    else:
        known = np.isfinite(syllable) & (syllable >= 0)
    return syllable, known


def assign_dominant_syllable(rows: pd.DataFrame, bundle_dir: Path) -> np.ndarray:
    out = np.full(len(rows), -1, dtype=np.int32)
    for sess, grp in rows.groupby("session"):
        path = bundle_dir / f"{sess}_kpms_syllables.npz"
        if not path.exists():
            print(f"[figures] WARN: missing syllable bundle for {sess}", flush=True)
            continue
        s, known = _load_syllable_series(path)
        for i, r in grp.iterrows():
            a, b = int(r["stride_a"]), int(r["stride_b"])
            if a < 0 or b > len(s) or a >= b:
                continue
            window = s[a:b]
            mask = known[a:b]
            window = window[mask & (window >= 0)]
            if not len(window):
                continue
            vals, counts = np.unique(window, return_counts=True)
            out[grp.index.get_loc(i)] = int(vals[counts.argmax()])
    return out


def per_session_syllable_usage(
    bundle_dir: Path, ann: dict[int, str]
) -> pd.DataFrame:
    """Per-session, per-syllable-name fraction of valid frames."""
    rows: list[dict[str, Any]] = []
    for path in sorted(bundle_dir.glob("*_kpms_syllables.npz")):
        z = np.load(path, allow_pickle=False)
        sess_key = str(z["session_key"]) if "session_key" in z.files else path.stem.replace(
            "_kpms_syllables", ""
        )
        s, known = _load_syllable_series(path)
        valid = known & (s >= 0)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        for syl_id in np.unique(s[valid]):
            label = ann.get(int(syl_id))
            if not label:
                continue
            n = int(((s == syl_id) & valid).sum())
            rows.append({
                "session": sess_key, "syllable_name": label,
                "syllable": int(syl_id), "n_frames": n, "frac": n / n_valid,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["session", "syllable_name"]).agg(
        n_frames=("n_frames", "sum"),
        frac=("frac", "sum"),
        n_syllables=("syllable", "count"),
    ).reset_index()


# ---------------------------------------------------------------------------
# Generic figure helpers
# ---------------------------------------------------------------------------


def speed_bins_mm_s(bins_cm_s: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(lo * 10.0, hi * 10.0) for lo, hi in bins_cm_s]


def binned_mean_sem(
    x: np.ndarray, y: np.ndarray, bins: list[tuple[float, float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cs: list[float] = []
    ms: list[float] = []
    ss: list[float] = []
    ns: list[int] = []
    for lo, hi in bins:
        mask = (x >= lo) & (x < hi) & np.isfinite(y)
        if mask.sum() < 5:
            continue
        cs.append((lo + hi) / 2.0)
        ms.append(float(np.mean(y[mask])))
        ss.append(float(np.std(y[mask]) / math.sqrt(mask.sum())))
        ns.append(int(mask.sum()))
    return np.array(cs), np.array(ms), np.array(ss), np.array(ns)


def label_panel(ax: plt.Axes, label: str, dx: float = -0.12, dy: float = 1.04) -> None:
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")


# ---------------------------------------------------------------------------
# Syllable-name figures
# ---------------------------------------------------------------------------


def fig_syll_name_usage(usage: pd.DataFrame, out: Path) -> None:
    if usage.empty:
        return
    pivot = usage.pivot_table(
        index="session", columns="syllable_name", values="frac",
        aggfunc="sum", fill_value=0.0,
    )
    name_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[name_order]
    fig, ax = plt.subplots(
        figsize=(max(8, 0.4 * len(name_order) + 4), max(5, 0.25 * len(pivot) + 2))
    )
    im = ax.imshow(pivot.values * 100, aspect="auto", cmap="magma_r")
    ax.set_xticks(np.arange(len(name_order)))
    ax.set_xticklabels(name_order, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Per-session syllable-name usage (% of valid frames)")
    plt.colorbar(im, ax=ax, label="% frames")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_syll_name_fingerprint(df: pd.DataFrame, out: Path, min_strides: int) -> None:
    counts = df.groupby("syllable_name").size()
    keep = counts[counts >= min_strides].sort_values(ascending=False).index.tolist()
    if not keep:
        return
    medians = (
        df[df["syllable_name"].isin(keep)]
        .groupby("syllable_name")[[m for m, _ in GAIT_METRICS]]
        .median()
        .reindex(keep)
    )
    z = (medians - medians.mean(axis=0)) / (medians.std(axis=0) + 1e-9)
    fig, ax = plt.subplots(
        figsize=(max(8, 0.5 * len(GAIT_METRICS) + 2), max(4, 0.35 * len(keep) + 2))
    )
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.0, vmax=2.0)
    ax.set_xticks(np.arange(len(GAIT_METRICS)))
    ax.set_xticklabels([lbl for _, lbl in GAIT_METRICS], rotation=60, ha="right")
    ax.set_yticks(np.arange(len(keep)))
    ax.set_yticklabels([f"{b} (n={int(counts[b])})" for b in keep])
    ax.set_title(f"Per-syllable-name gait fingerprint (z-scored medians; ≥{min_strides} strides)")
    plt.colorbar(im, ax=ax, label="z-score")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_syll_name_speed_violins(df: pd.DataFrame, out: Path, min_strides: int) -> None:
    counts = df.groupby("syllable_name").size()
    keep = counts[counts >= min_strides].sort_values(ascending=False).index.tolist()
    if not keep:
        return
    data = [
        df[df["syllable_name"] == b]["stride_speed_cm_s"].dropna().values for b in keep
    ]
    fig, ax = plt.subplots(figsize=(max(7, 0.4 * len(keep) + 3), 5))
    parts = ax.violinplot(data, showmedians=True, widths=0.8)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    ax.set_xticks(np.arange(1, len(keep) + 1))
    ax.set_xticklabels(
        [f"{b}\n(n={int(counts[b])})" for b in keep], rotation=60, ha="right"
    )
    ax.set_ylabel("stride speed (cm/s)")
    ax.set_title(f"Stride-speed distribution per syllable name (≥{min_strides} strides)")
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main paper figures
# ---------------------------------------------------------------------------


def fig_main_canonical(
    df: pd.DataFrame,
    speed_bins_cm: list[tuple[float, float]],
    out: Path,
) -> None:
    sub = df[df["straight_gait"]].copy()
    bins = speed_bins_mm_s(speed_bins_cm)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    ax = axes[0]
    ax.scatter(sub["stride_speed_cm_s"], sub["limb_duty_factor"],
               s=4, alpha=0.18, color="grey", zorder=0)
    c, m, s, n = binned_mean_sem(
        sub["stride_speed_mm_s"].values, sub["limb_duty_factor"].values, bins
    )
    ax.errorbar(c / 10, m, yerr=s, fmt="o-", color="tab:blue", lw=1.8, capsize=3, markersize=7)
    for ci, mi, ni in zip(c, m, n):
        ax.text(ci / 10, mi + 0.04, f"{ni}", ha="center", fontsize=7, color="tab:blue")
    ax.set_xlabel("stride speed (cm/s)")
    ax.set_ylabel("limb duty factor")
    ax.set_xlim(2, 25)
    ax.set_ylim(0.1, 1.0)
    ax.set_title("Duty factor ↓ with speed")
    label_panel(ax, "a")

    ax = axes[1]
    for col, lab, color in [("duty_LH", "LH", "tab:red"), ("duty_RH", "RH", "tab:orange")]:
        c, m, s, n = binned_mean_sem(
            sub["stride_speed_mm_s"].values, (sub[col] * 100).values, bins
        )
        ax.errorbar(c / 10, m, yerr=s, fmt="o-", color=color, lw=1.8, capsize=3,
                    markersize=6, label=f"{lab} stance %")
    ax.set_xlabel("stride speed (cm/s)")
    ax.set_ylabel("stance % of stride")
    ax.set_xlim(2, 25)
    ax.set_ylim(20, 90)
    ax.legend()
    ax.set_title("Stance % ↓ with speed")
    label_panel(ax, "b")

    ax = axes[2]
    ax.scatter(sub["stride_speed_cm_s"], sub["stride_length_mm"],
               s=4, alpha=0.18, color="grey", zorder=0)
    c, m, s, n = binned_mean_sem(
        sub["stride_speed_mm_s"].values, sub["stride_length_mm"].values, bins
    )
    ax.errorbar(c / 10, m, yerr=s, fmt="o-", color="tab:green", lw=1.8, capsize=3, markersize=7)
    ax.set_xlabel("stride speed (cm/s)")
    ax.set_ylabel("stride length (mm) [LH]")
    ax.set_xlim(2, 25)
    ax.set_ylim(0, 80)
    ax.set_title("Stride length ↑ with speed")
    label_panel(ax, "c")

    plt.suptitle(
        f"Canonical speed-dependent gait — straight-gait strides (n={len(sub):,})",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _smoothed_session_3d(coords: np.ndarray, win: int, poly: int) -> np.ndarray:
    Xs = coords.copy()
    nm = np.isnan(Xs)
    if nm.any():
        flat = Xs.reshape(len(Xs), -1)
        mflat = nm.reshape(len(Xs), -1)
        idx = np.arange(len(Xs))
        for c in range(flat.shape[1]):
            mask = mflat[:, c]
            if mask.any() and (~mask).any():
                flat[mask, c] = np.interp(idx[mask], idx[~mask], flat[~mask, c])
        Xs = flat.reshape(coords.shape)
    return savgol_filter(Xs, win, poly, axis=0, mode="interp")


def _pick_example_session(
    sessions_3d: dict[str, np.ndarray],
    fps: float,
    requested: str | None,
) -> str:
    if requested and requested in sessions_3d:
        return requested
    if requested and f"session_{requested}" in sessions_3d:
        return f"session_{requested}"
    ix = {n: i for i, n in enumerate(KEYPOINTS)}
    best_key = None
    best_score = -np.inf
    for key, X in sessions_3d.items():
        if X.shape[0] < int(6 * fps):
            continue
        tbase_xy = X[:, ix[KP_TAIL_BASE], :2]
        speed = np.linalg.norm(np.gradient(np.nan_to_num(tbase_xy), axis=0) * fps, axis=1)
        if not np.isfinite(speed).any():
            continue
        score = float(np.nanpercentile(speed, 95))
        if score > best_score:
            best_score = score
            best_key = key
    if best_key is None:
        best_key = next(iter(sessions_3d))
    return best_key


def fig_main_traces(
    df: pd.DataFrame,
    sessions_3d: dict[str, np.ndarray],
    fps: float,
    out: Path,
    *,
    example_session: str | None,
    example_window_seconds: float,
    max_kp_jump_mm: float,
    smooth_win: int = 7,
    smooth_poly: int = 2,
) -> None:
    ix = {n: i for i, n in enumerate(KEYPOINTS)}
    sub = df[df["straight_gait"]].copy()

    nose_traces: list[tuple[np.ndarray, np.ndarray]] = []
    ttip_traces: list[tuple[np.ndarray, np.ndarray]] = []
    for sess, grp in sub.groupby("session"):
        if sess not in sessions_3d:
            continue
        Xs = _smoothed_session_3d(sessions_3d[sess], smooth_win, smooth_poly)
        spine = Xs[:, ix[KP_SPINE], :]
        ttip = Xs[:, ix[KP_TAIL_TIP], :]
        nose = Xs[:, ix[KP_NOSE_PROXY], :]
        for _, r in grp.iterrows():
            a, b = int(r["stride_a"]), int(r["stride_b"])
            if (b - a) < 6:
                continue
            d_xy = spine[b, :2] - spine[a, :2]
            n_disp = np.linalg.norm(d_xy)
            if n_disp < 1e-6:
                continue
            fwd = d_xy / n_disp
            lat = np.array([-fwd[1], fwd[0]])
            bl = float(r["body_length_mm"])
            if not (np.isfinite(bl) and bl > 1e-3):
                continue
            t_norm = np.linspace(0, 1, b - a + 1)
            for src, store in ((nose, nose_traces), (ttip, ttip_traces)):
                rel = src[a:b + 1, :2] - spine[a, :2]
                y = (rel @ lat) / bl
                if np.isfinite(y).all():
                    store.append((t_norm, y))

    example_key = _pick_example_session(sessions_3d, fps, example_session)
    X = sessions_3d[example_key].copy()
    n_dropped = 0
    if len(X) > 1:
        kp_step = np.linalg.norm(np.diff(X, axis=0), axis=2)
        bad_kp = kp_step > max_kp_jump_mm
        bad_frame = np.zeros(len(X), dtype=bool)
        bad_frame[1:] |= bad_kp.any(axis=1)
        bad_frame[:-1] |= bad_kp.any(axis=1)
        X[bad_frame] = np.nan
        n_dropped = int(bad_frame.sum())
    Xs = savgol_filter(np.nan_to_num(X), smooth_win, smooth_poly, axis=0, mode="interp")
    Xs[np.isnan(X)] = np.nan
    spine = Xs[:, ix[KP_SPINE], :]
    tbase = Xs[:, ix[KP_TAIL_BASE], :]
    neck = Xs[:, ix[KP_NECK], :]
    animal_speed = np.linalg.norm(
        np.gradient(np.nan_to_num(tbase[:, :2]), axis=0) * fps, axis=1
    )
    animal_speed[np.isnan(tbase[:, 0])] = np.nan

    v = neck[:, :2] - tbase[:, :2]
    ang = np.arctan2(v[:, 1], v[:, 0])
    if np.isnan(ang).any():
        idx_t = np.arange(len(ang))
        m = ~np.isnan(ang)
        ang = np.interp(idx_t, idx_t[m], ang[m])
    ang_u = np.unwrap(ang)
    win = min(21, len(ang_u) // 2 * 2 - 1)
    if win < 5:
        omega = np.gradient(ang_u) * fps * 180 / np.pi
    else:
        omega = (
            savgol_filter(ang_u, win, 3, deriv=1, delta=1.0 / fps, mode="interp")
            * 180
            / np.pi
        )

    paw_speed: dict[str, np.ndarray] = {}
    for paw in ("forepaw_L", "forepaw_R", "hindpaw_L", "hindpaw_R"):
        p = Xs[:, ix[paw], :]
        ps = np.linalg.norm(np.gradient(np.nan_to_num(p), axis=0), axis=1) * fps
        ps[np.isnan(p[:, 0])] = np.nan
        paw_speed[paw] = ps

    n_frames = int(example_window_seconds * fps)
    speed_clean = np.where(np.isfinite(animal_speed), animal_speed, 0.0)
    in_walk = ((speed_clean > 100) & (speed_clean < 300)).astype(float)
    too_fast = (speed_clean > 600).astype(float)
    if len(speed_clean) <= n_frames:
        start = 0
    else:
        walk_score = np.convolve(in_walk, np.ones(n_frames), mode="valid")
        burst_pen = np.convolve(too_fast, np.ones(n_frames) * 50.0, mode="valid")
        start = int(np.argmax(walk_score - burst_pen))
    t = np.arange(n_frames) / fps

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.4, 1.4],
                          width_ratios=[1, 1, 1], hspace=0.45, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    grid = np.linspace(0, 1, 100)
    for ax, traces, color, label in [
        (ax_a, nose_traces, "tab:blue", "nose-proxy"),
        (ax_b, ttip_traces, "tab:red", "tail tip"),
    ]:
        if not traces:
            ax.set_title(f"{label} (no data)")
            continue
        S: list[np.ndarray] = []
        for t_norm, y in traces:
            try:
                cs = CubicSpline(t_norm, y, bc_type="natural")
                S.append(cs(grid))
            except Exception:
                continue
        S_arr = np.array(S)
        peak_sign = np.sign(np.nanmax(S_arr, axis=1) + np.nanmin(S_arr, axis=1))
        peak_sign = np.where(peak_sign == 0, 1, peak_sign)
        S_arr = S_arr * peak_sign[:, None]
        m = np.nanmean(S_arr, axis=0)
        ci = 1.96 * np.nanstd(S_arr, axis=0) / math.sqrt(max(1, len(S_arr)))
        ax.plot(grid * 100, m, color=color, lw=2.0)
        ax.fill_between(grid * 100, m - ci, m + ci, color=color, alpha=0.25)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("stride phase (%)")
        ax.set_ylabel(f"{label} lateral / body length")
        ax.set_xlim(0, 100)
    label_panel(ax_a, "a")
    label_panel(ax_b, "b")
    ax_a.set_title("Nose-proxy lateral displacement")
    ax_b.set_title("Tail-tip lateral displacement")

    ax_c = fig.add_subplot(gs[0, 2])
    ss = df[df["straight_gait"]]
    ax_c.scatter(ss["lat_disp_phase_ttip"], ss["lat_disp_amp_ttip"],
                 s=4, color="tab:red", alpha=0.5)
    ax_c.axhline(0, color="grey", lw=0.5)
    ax_c.set_xlim(0, 1)
    ax_c.set_xlabel("tail-tip phase (stride frac)")
    ax_c.set_ylabel("tail-tip amp / body len")
    ax_c.set_title("Phase × amplitude (tail tip)")
    label_panel(ax_c, "c")

    speed_window = animal_speed[start:start + n_frames] / 10
    speed_ymax = float(min(80.0, max(40.0, np.nanpercentile(speed_window, 99.5) * 1.2)))
    ax_speed = fig.add_subplot(gs[1, :])
    ax_speed.plot(t, speed_window, color="black", lw=1.0)
    ax_speed.axhline(5, color="grey", ls="--", lw=0.6, label="5 cm/s")
    ax_speed.axhline(10, color="grey", ls=":", lw=0.6, label="10 cm/s")
    ax_speed.set_ylim(0, speed_ymax)
    ax_speed.set_ylabel("animal speed (cm/s)")
    ax_speed.set_xticklabels([])
    ax_speed.legend(loc="upper right", fontsize=7)
    label_panel(ax_speed, "d")

    paw_colors = {
        "forepaw_L": "tab:blue", "forepaw_R": "tab:cyan",
        "hindpaw_L": "tab:red", "hindpaw_R": "tab:orange",
    }
    paw_window_max = max(
        np.nanpercentile(v[start:start + n_frames] / 10, 99.5)
        for v in paw_speed.values()
    )
    paw_ymax = float(min(150.0, max(60.0, paw_window_max * 1.15)))
    ax_paws = fig.add_subplot(gs[2, :], sharex=ax_speed)
    for paw, vps in paw_speed.items():
        ax_paws.plot(t, vps[start:start + n_frames] / 10,
                     color=paw_colors[paw], lw=1.0, label=paw)
    ax_paws.axhline(35 / 10, color="grey", lw=0.5, ls=":", alpha=0.7)
    ax_paws.axhline(120 / 10, color="grey", lw=0.5, ls=":", alpha=0.7)
    ax_paws.set_ylim(0, paw_ymax)
    ax_paws.set_xlabel("time (s)")
    ax_paws.set_ylabel("paw speed (cm/s)")
    ax_paws.legend(loc="upper right", fontsize=7, ncol=2)
    label_panel(ax_paws, "e")

    plt.suptitle(
        f"Lateral postural oscillations + example locomotor bout"
        f"  (lat n={len(ss):,};  bout: {example_key}, {example_window_seconds:.0f}s @ frame {start};"
        f" jump-filter dropped {n_dropped} frames)",
        fontsize=11, y=0.99,
    )
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _spinal_arc_per_session(sessions_3d: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    ix = {n: i for i, n in enumerate(KEYPOINTS)}
    out: dict[str, dict[str, float]] = {}
    for sess, X in sessions_3d.items():
        tb = X[:, ix["tail_base"], :]
        bb = X[:, ix["back_bottom"], :]
        bm = X[:, ix["back_middle"], :]
        bt = X[:, ix["back_top"], :]
        s1 = np.linalg.norm(bb - tb, axis=1)
        s2 = np.linalg.norm(bm - bb, axis=1)
        s3 = np.linalg.norm(bt - bm, axis=1)
        arc = s1 + s2 + s3
        out[sess] = {
            "spinal_arc_mm": float(np.nanmedian(arc)),
            "stretched_len_mm": float(np.nanpercentile(np.linalg.norm(bt - tb, axis=1), 95)),
        }
    return out


def fig_individual_diffs(
    df: pd.DataFrame,
    sessions_3d: dict[str, np.ndarray],
    mouse_map: dict[str, tuple[str, int]],
    out: Path,
    sess_summary_csv: Path | None = None,
) -> None:
    sub = df[df["straight_gait"]].copy()
    sub["mouse"] = sub["session"].map(
        lambda s: mouse_map.get(s, (s.replace("session_", ""), 0))[0]
    )
    sub["trial"] = sub["session"].map(
        lambda s: mouse_map.get(s, (s.replace("session_", ""), 0))[1]
    )

    arc_data = _spinal_arc_per_session(sessions_3d)
    arc_per_sess = {k: v["spinal_arc_mm"] for k, v in arc_data.items()}
    stretched_per_sess = {k: v["stretched_len_mm"] for k, v in arc_data.items()}
    sub["spinal_arc_mm"] = sub["session"].map(arc_per_sess)
    sub["stretched_len_mm"] = sub["session"].map(stretched_per_sess)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    mice = sorted(sub["mouse"].unique())
    palette = plt.get_cmap("tab10")
    mouse_color = {m: palette(i % 10) for i, m in enumerate(mice)}

    ax = axes[0, 0]
    rng = np.random.default_rng(0)
    for i, m in enumerate(mice):
        sess_arcs = sorted({
            arc_per_sess[s] for s in sub[sub["mouse"] == m]["session"].unique()
            if s in arc_per_sess
        })
        for v in sess_arcs:
            ax.plot(i + rng.uniform(-0.12, 0.12), v, "o",
                    color=mouse_color[m], markeredgecolor="black",
                    markeredgewidth=0.7, markersize=10)
        if sess_arcs:
            ax.hlines(np.median(sess_arcs), i - 0.25, i + 0.25, color="black", lw=1.6)
    ax.set_xticks(range(len(mice)))
    ax.set_xticklabels([
        f"{m}\n(n={int(sub[sub['mouse'] == m]['session'].nunique())} sess)" for m in mice
    ])
    ax.set_ylabel("spinal arc length (mm)\n[tb→bb + bb→bm + bm→bt]")
    ax.set_title("Body size per mouse — posture-invariant")
    label_panel(ax, "a")

    sess_summary = (
        sub.groupby(["mouse", "session"])
        .agg(
            spinal_arc_mm=("spinal_arc_mm", "first"),
            stretched_len_mm=("stretched_len_mm", "first"),
            body_length_per_stride_mm=("body_length_mm", "median"),
            stride_length_mm=("stride_length_mm", "median"),
            step_length_mm=("step_length_mm", "median"),
            step_width_mm=("step_width_mm", "median"),
            limb_duty_factor=("limb_duty_factor", "median"),
            stride_freq_hz=("stride_freq_hz", "median"),
            stride_speed_cm_s=("stride_speed_cm_s", "median"),
            n=("session", "size"),
        )
        .reset_index()
    )

    def _scatter_panel(ax: plt.Axes, ycol: str, ylabel: str, title: str, panel_lbl: str) -> None:
        for m in mice:
            mm = sub[sub["mouse"] == m]
            ax.scatter(mm["spinal_arc_mm"], mm[ycol], s=4, alpha=0.18, color=mouse_color[m])
        for _, r in sess_summary.iterrows():
            ax.scatter(r["spinal_arc_mm"], r[ycol], s=85,
                       color=mouse_color[r["mouse"]], edgecolor="black",
                       linewidth=1.0, zorder=5)
        valid = sess_summary[["spinal_arc_mm", ycol]].dropna()
        if len(valid) >= 3:
            r = float(np.corrcoef(valid["spinal_arc_mm"], valid[ycol])[0, 1])
            ax.text(0.05, 0.97, f"r (per-session) = {r:+.2f}\nn={len(valid)} sessions",
                    transform=ax.transAxes, va="top", fontsize=8)
        ax.set_xlabel("spinal arc length (mm) — body size proxy")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        label_panel(ax, panel_lbl)

    _scatter_panel(axes[0, 1], "stride_length_mm", "stride length (mm)",
                   "Stride length vs body size", "b")
    _scatter_panel(axes[0, 2], "step_length_mm", "step length (mm)",
                   "Step length vs body size", "c")
    _scatter_panel(axes[1, 0], "step_width_mm", "step width (mm)",
                   "Step width vs body size", "d")
    _scatter_panel(axes[1, 1], "limb_duty_factor", "limb duty factor",
                   "Duty factor vs body size", "e")
    _scatter_panel(axes[1, 2], "stride_freq_hz", "stride frequency (Hz)",
                   "Stride frequency vs body size", "f")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=mouse_color[m], markeredgecolor="black",
                   markersize=10, label=m) for m in mice
    ]
    fig.legend(handles=handles, loc="lower center", ncol=max(1, len(mice)),
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.suptitle(
        "Individual differences — gait metrics vs SPINAL ARC LENGTH (posture-invariant)"
        f"  (small dots = strides, n={len(sub):,};  large dots = per-session medians)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    if sess_summary_csv is not None:
        sess_summary.to_csv(sess_summary_csv, index=False)


# ---------------------------------------------------------------------------
# figures subcommand
# ---------------------------------------------------------------------------


def _load_annotations(path: Path) -> dict[int, str]:
    if path is None:
        return {}
    raw = json.loads(Path(path).expanduser().read_text())
    if isinstance(raw, dict) and isinstance(raw.get("id_to_name"), dict):
        raw = raw["id_to_name"]
    out: dict[int, str] = {}
    for k, v in raw.items():
        try:
            sid = int(k)
        except (TypeError, ValueError):
            continue
        label = str(v).strip()
        if label:
            out[sid] = label
    return out


def _load_sessions_3d_for_figures(
    sessions: Iterable[str],
    release_root: Path,
    mask_invalid: bool,
    mask_invalid_jump: bool,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for sess_key in sessions:
        sid = sess_key.replace("session_", "")
        try:
            coords = load_session_3d(
                release_root, sid,
                mask_invalid=mask_invalid,
                mask_invalid_jump=mask_invalid_jump,
            )
        except FileNotFoundError:
            continue
        out[sess_key] = coords
    return out


def cmd_figures(args: argparse.Namespace, config: dict[str, Any]) -> None:
    plt.rcParams.update(DEFAULT_RC)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = (args.csv_dir.expanduser().resolve()
               if args.csv_dir is not None else out_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    fig_params = merged_figure_params(config)
    speed_bins_cm = merged_speed_bins(config)

    df = pd.read_csv(args.per_stride_csv.expanduser())
    if "straight_gait" in df.columns:
        df["straight_gait"] = df["straight_gait"].astype(bool)
    print(f"[figures] loaded {len(df):,} strides "
          f"({int(df['straight_gait'].sum()) if 'straight_gait' in df.columns else 0:,} straight)",
          flush=True)

    bundle_dir = args.syllable_bundle_dir.expanduser().resolve()
    ann_path = (
        args.annotations_json.expanduser()
        if args.annotations_json is not None
        else bundle_dir / "syllable_annotations.json"
    )
    if not ann_path.exists():
        ann_inline = config.get("syllable_annotations") or {}
        if not ann_inline:
            raise SystemExit(
                f"Annotations not found at {ann_path} and not provided in --config."
            )
        ann = {int(k): str(v).strip() for k, v in ann_inline.items()}
    else:
        ann = _load_annotations(ann_path)

    df["syllable_dominant"] = assign_dominant_syllable(df, bundle_dir)
    df["syllable_name"] = df["syllable_dominant"].map(ann).where(
        df["syllable_dominant"].map(ann).notna(),
        other=pd.NA,
    )
    n_named = int(df["syllable_name"].notna().sum())
    print(f"[figures] {n_named:,}/{len(df):,} strides have a syllable-name label", flush=True)
    df.to_csv(csv_dir / "per_stride_with_syllable_name.csv", index=False)

    usage = per_session_syllable_usage(bundle_dir, ann)
    if not usage.empty:
        usage.to_csv(csv_dir / "per_syllable_name_usage.csv", index=False)

    syll_df = df.dropna(subset=["syllable_name"]).copy()
    min_strides = int(fig_params["min_strides_per_syllable_name"])

    fig_syll_name_usage(usage, out_dir / "fig_syll_name_usage.png")
    fig_syll_name_fingerprint(syll_df, out_dir / "fig_syll_name_fingerprint.png", min_strides)
    fig_syll_name_speed_violins(syll_df, out_dir / "fig_syll_name_speed_violins.png", min_strides)

    fig_main_canonical(df, speed_bins_cm, out_dir / "fig_main_canonical.png")

    release_root = release_root_from_arg(args.release_root)
    fps = float(args.fps if args.fps is not None else (
        config.get("dataset", {}).get("fps", FPS_DEFAULT)
    ))
    sessions_needed = sorted(set(df["session"].astype(str).tolist()))
    mask_invalid, mask_invalid_jump = _resolve_mask_flags(args, config)
    sessions_3d = _load_sessions_3d_for_figures(
        sessions_needed, release_root,
        mask_invalid=mask_invalid,
        mask_invalid_jump=mask_invalid_jump,
    )
    if not sessions_3d:
        raise SystemExit("No release session 3D coordinates available; cannot render fig_main_traces.")

    fig_main_traces(
        df, sessions_3d, fps,
        out_dir / "fig_main_traces.png",
        example_session=fig_params.get("example_session"),
        example_window_seconds=float(fig_params["example_window_seconds"]),
        max_kp_jump_mm=float(fig_params["max_kp_jump_mm"]),
    )

    mouse_map = load_mouse_map(args.mouse_map)
    fig_individual_diffs(
        df, sessions_3d, mouse_map,
        out_dir / "fig_individual_diffs.png",
        sess_summary_csv=csv_dir / "per_session_individual_diffs.csv",
    )

    print(json.dumps({
        "output_dir": str(out_dir),
        "figures": [
            "fig_syll_name_usage.png",
            "fig_syll_name_fingerprint.png",
            "fig_syll_name_speed_violins.png",
            "fig_main_canonical.png",
            "fig_main_traces.png",
            "fig_individual_diffs.png",
        ],
        "n_strides": int(len(df)),
        "n_named_strides": n_named,
        "n_sessions_with_3d": len(sessions_3d),
        "min_strides_per_syllable_name": min_strides,
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None,
                        help="Optional gait YAML config (configs/gait.yaml).")
    parser.add_argument("--release-root", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--mask-invalid", dest="mask_invalid",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Mask cells with valid_3d=False before stride detection. "
                             "Default: dataset.mask_invalid_3d in --config (true).")
    parser.add_argument("--mask-invalid-jump-timepoint", dest="mask_invalid_jump_timepoint",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="Mask cells with invalid_jump_timepoint=True before stride detection. "
                             "Default: dataset.mask_invalid_jump_timepoint in --config (true).")


def _resolve_mask_flags(args: argparse.Namespace, config: dict[str, Any]) -> tuple[bool, bool]:
    dcfg = config.get("dataset") or {}
    mask_invalid = (
        args.mask_invalid if args.mask_invalid is not None
        else bool(dcfg.get("mask_invalid_3d", True))
    )
    mask_invalid_jump = (
        args.mask_invalid_jump_timepoint if args.mask_invalid_jump_timepoint is not None
        else bool(dcfg.get("mask_invalid_jump_timepoint", True))
    )
    return mask_invalid, mask_invalid_jump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    strides = subparsers.add_parser(
        "strides",
        help="Detect strides and write per_stride.csv from release 3D trajectories.",
    )
    _add_common_dataset_args(strides)
    strides.add_argument("--output-dir", type=Path, required=True)
    strides.add_argument("--session-ids-file", type=Path, default=None)

    figures = subparsers.add_parser(
        "figures",
        help="Render the canonical gait figures from per_stride.csv.",
    )
    _add_common_dataset_args(figures)
    figures.add_argument("--per-stride-csv", type=Path, required=True)
    figures.add_argument("--syllable-bundle-dir", type=Path, required=True,
                         help="Directory containing session_<id>_kpms_syllables.npz files.")
    figures.add_argument("--annotations-json", type=Path, default=None,
                         help="syllable_id -> name JSON. Defaults to "
                              "<syllable-bundle-dir>/syllable_annotations.json.")
    figures.add_argument("--mouse-map", type=Path, default=None,
                         help="Optional CSV mapping session_id,mouse,trial for "
                              "fig_individual_diffs.")
    figures.add_argument("--output-dir", type=Path, required=True,
                         help="Output directory for the six figures.")
    figures.add_argument("--csv-dir", type=Path, default=None,
                         help="Output directory for derived CSVs (per-stride + "
                              "per-syllable-name usage). Defaults to --output-dir.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.subcommand == "strides":
        cmd_strides(args, config)
    elif args.subcommand == "figures":
        cmd_figures(args, config)
    else:
        raise SystemExit(f"Unknown gait subcommand: {args.subcommand}")


if __name__ == "__main__":
    main()
