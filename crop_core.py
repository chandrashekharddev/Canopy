
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --------------------------------------------------------------------------
# 1. Crop-specific trough-detection parameters (from the notebook)
# --------------------------------------------------------------------------
CROP_PARAMS = {
    "Tomato":    {"height": -0.30, "prominence": 0.15, "min_len": 5, "min_peak": 0.45},
    "Cotton":    {"height": -0.30, "prominence": 0.20, "min_len": 5, "min_peak": 0.50},
    "Paddy":     {"height": -0.30, "prominence": 0.25, "min_len": 5, "min_peak": 0.55},
    "Onion":     {"height": -0.30, "prominence": 0.15, "min_len": 5, "min_peak": 0.45},
    "Sugarcane": {"height": -0.30, "prominence": 0.20, "min_len": 5, "min_peak": 0.50},
}
DEFAULT_PARAMS = {"height": -0.30, "prominence": 0.20, "min_len": 5, "min_peak": 0.50}

SUGARCANE_MIN_DAYS = 390
SUGARCANE_MAX_DAYS = 430

CROP_DURATION_BOUNDS = {
    "Tomato": (90, 180),
    "Cotton": (140, 210),
    "Paddy": (100, 160),
    "Onion": (90, 150),
    "Sugarcane": (SUGARCANE_MIN_DAYS, SUGARCANE_MAX_DAYS),
}
DEFAULT_DURATION_BOUNDS = (90, 200)


def get_duration_bounds(crop: str | None) -> tuple[int, int]:
    return CROP_DURATION_BOUNDS.get(crop, DEFAULT_DURATION_BOUNDS)


def is_duration_plausible(crop: str | None, duration_days: int) -> bool:
    min_days, max_days = get_duration_bounds(crop)
    return min_days <= duration_days <= max_days

FEATURE_COLS = [
    "duration_days", "n_obs", "obs_density", "peak_ndvi", "start_ndvi", "end_ndvi",
    "min_ndvi", "max_ndvi", "mean_ndvi", "std_ndvi", "amplitude", "growth_rate",
    "decay_rate", "overall_slope", "days_to_peak", "days_from_peak", "peak_position_ratio",
    "auc_norm", "pct_negative", "pct_low_ndvi", "internal_sign_changes",
    "q1_mean", "q2_mean", "q3_mean", "q4_mean", "peak_month", "start_month", "end_month",
    "peak_month_sin", "peak_month_cos", "min_negative_flag",
]

CROPS = ["Tomato", "Cotton", "Paddy", "Onion", "Sugarcane"]


SMOOTHING_WINDOW = 5


def add_smoothed_ndvi(series_df: pd.DataFrame, window: int = SMOOTHING_WINDOW) -> pd.DataFrame:
    """
    Returns a copy of series_df with an added 'ndvi_smooth' column: a centered
    rolling-average of the raw NDVI values (min_periods=1 so the ends of the
    series aren't dropped/NaN, they just average over fewer neighbours).
    """
    out = series_df.copy()
    out["ndvi_smooth"] = (
        out["ndvi"].rolling(window=window, center=True, min_periods=1).mean().round(4)
    )
    return out


# --------------------------------------------------------------------------
# 2. Lifecycle extraction
# --------------------------------------------------------------------------
def _raw_lifecycles(farm_df, height, prominence, min_len, min_peak):
    ndvi = farm_df["ndvi"].values
    smoothed = pd.Series(ndvi).rolling(window=3, center=True, min_periods=1).mean().values
    trough_idx, _ = find_peaks(-smoothed, height=height, prominence=prominence)
    boundaries = [0] + list(trough_idx) + [len(smoothed) - 1]

    lifecycles = []
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        segment = farm_df.iloc[start: end + 1]
        if len(segment) < min_len:
            continue
        seg_smoothed = pd.Series(segment["ndvi"].values).rolling(window=3, center=True, min_periods=1).mean().values

        if seg_smoothed.max() < min_peak:
            continue
        peak_pos = int(np.argmax(seg_smoothed))
        lifecycles.append({
            "start_date": segment.iloc[0]["date"], "end_date": segment.iloc[-1]["date"],
            "duration_days": (segment.iloc[-1]["date"] - segment.iloc[0]["date"]).days,
            "peak_ndvi": round(float(seg_smoothed[peak_pos]), 4),
            "peak_date": segment.iloc[peak_pos]["date"],
        })
    return lifecycles


def _select_lifecycle_by_ground_truth(lifecycles, gt_date):
    if not lifecycles:
        return None
    if gt_date is None or pd.isna(gt_date):
        return max(lifecycles, key=lambda lc: lc["duration_days"])


    containing = [lc for lc in lifecycles if lc["start_date"] <= gt_date <= lc["end_date"]]
    if len(containing) == 1:
        return containing[0]
    if len(containing) > 1:

        return min(containing, key=lambda lc: abs((lc["peak_date"] - gt_date).days))

    return min(lifecycles, key=lambda lc: abs((lc["peak_date"] - gt_date).days))


def _clip_segment_to_duration(farm_df, peak_date, target_days, full_start, full_end):

    half = target_days / 2
    start_date = max(full_start, peak_date - pd.Timedelta(days=half))
    end_date = min(full_end, start_date + pd.Timedelta(days=target_days))
    if (end_date - start_date).days < target_days and start_date > full_start:
        start_date = max(full_start, end_date - pd.Timedelta(days=target_days))

    mask = (farm_df["date"] >= start_date) & (farm_df["date"] <= end_date)
    segment = farm_df.loc[mask]
    if segment.empty:
        return None
    seg_smoothed = pd.Series(segment["ndvi"].values).rolling(window=3, center=True, min_periods=1).mean().values
    peak_pos = int(np.argmax(seg_smoothed))
    return {
        "start_date": segment.iloc[0]["date"], "end_date": segment.iloc[-1]["date"],
        "duration_days": (segment.iloc[-1]["date"] - segment.iloc[0]["date"]).days,
        "peak_ndvi": round(float(seg_smoothed[peak_pos]), 4),
        "peak_date": segment.iloc[peak_pos]["date"],
    }


SUGARCANE_TROUGH_PARAMS = {"height": -0.30, "prominence": 0.30, "min_len": 20, "min_peak": 0.45}
SUGARCANE_PLAUSIBLE_MIN_DAYS = 250  

def _sugarcane_lifecycle(farm_df, gt_date):
    full_start, full_end = farm_df["date"].iloc[0], farm_df["date"].iloc[-1]


    candidates = _raw_lifecycles(farm_df, **SUGARCANE_TROUGH_PARAMS)
    candidates = [c for c in candidates if c["duration_days"] >= SUGARCANE_PLAUSIBLE_MIN_DAYS]
    lc = _select_lifecycle_by_ground_truth(candidates, gt_date)
    if lc is not None:
        if lc["duration_days"] > SUGARCANE_MAX_DAYS:
            target = (SUGARCANE_MIN_DAYS + SUGARCANE_MAX_DAYS) / 2
            clipped = _clip_segment_to_duration(farm_df, lc["peak_date"], target, full_start, full_end)
            if clipped is not None:
                lc = clipped
        return lc

    ndvi = farm_df["ndvi"].values
    smoothed = pd.Series(ndvi).rolling(window=3, center=True, min_periods=1).mean().values
    peak_idx = int(np.argmax(smoothed))
    peak_date = farm_df.loc[peak_idx, "date"]

    total_days = (full_end - full_start).days
    if total_days <= SUGARCANE_MAX_DAYS:
        segment = farm_df
        seg_smoothed = pd.Series(segment["ndvi"].values).rolling(window=3, center=True, min_periods=1).mean().values
        peak_pos = int(np.argmax(seg_smoothed))
        return {
            "start_date": segment.iloc[0]["date"], "end_date": segment.iloc[-1]["date"],
            "duration_days": (segment.iloc[-1]["date"] - segment.iloc[0]["date"]).days,
            "peak_ndvi": round(float(seg_smoothed[peak_pos]), 4),
            "peak_date": segment.iloc[peak_pos]["date"],
        }

    target = (SUGARCANE_MIN_DAYS + SUGARCANE_MAX_DAYS) / 2
    return _clip_segment_to_duration(farm_df, peak_date, target, full_start, full_end)


def _segment_to_lifecycle(segment: pd.DataFrame, seg_smoothed) -> dict:
    peak_pos = int(np.argmax(seg_smoothed))
    return {
        "start_date": segment.iloc[0]["date"], "end_date": segment.iloc[-1]["date"],
        "duration_days": (segment.iloc[-1]["date"] - segment.iloc[0]["date"]).days,
        "peak_ndvi": round(float(seg_smoothed[peak_pos]), 4),
        "peak_date": segment.iloc[peak_pos]["date"],
    }


def _find_deepest_internal_trough(seg_smoothed):

    if len(seg_smoothed) < 9:
        return None
    trough_idx, props = find_peaks(-seg_smoothed, prominence=0.05)

    interior = (trough_idx > 1) & (trough_idx < len(seg_smoothed) - 2)
    trough_idx = trough_idx[interior]
    if len(trough_idx) == 0:
        return None
    prominences = props["prominences"][interior]
    return int(trough_idx[np.argmax(prominences)])


def _resplit_if_merged(farm_df, lc, gt_date, min_peak, min_len, min_relative_drop, max_iterations=3):

    for _ in range(max_iterations):
        mask = (farm_df["date"] >= lc["start_date"]) & (farm_df["date"] <= lc["end_date"])
        segment = farm_df.loc[mask].reset_index(drop=True)
        if len(segment) < 2 * min_len:
            break

        seg_smoothed = pd.Series(segment["ndvi"].values).rolling(window=3, center=True, min_periods=1).mean().values
        trough_pos = _find_deepest_internal_trough(seg_smoothed)
        if trough_pos is None:
            break

        left_smoothed, right_smoothed = seg_smoothed[:trough_pos + 1], seg_smoothed[trough_pos:]
        if len(left_smoothed) < min_len or len(right_smoothed) < min_len:
            break

        left_peak, right_peak = float(left_smoothed.max()), float(right_smoothed.max())
        trough_val = float(seg_smoothed[trough_pos])
        lower_peak = min(left_peak, right_peak)
        if lower_peak <= 0:
            break
        relative_drop = (lower_peak - trough_val) / lower_peak

        if relative_drop < min_relative_drop or left_peak < min_peak or right_peak < min_peak:
            break  # not a genuine split - a shallow/partial dip is normal within one real cycle

        left_lc = _segment_to_lifecycle(segment.iloc[:trough_pos + 1], left_smoothed)
        right_lc = _segment_to_lifecycle(segment.iloc[trough_pos:], right_smoothed)
        chosen = _select_lifecycle_by_ground_truth([left_lc, right_lc], gt_date)
        if chosen is None:
            break
        lc = chosen
    return lc


def extract_ground_truth_lifecycle(farm_df: pd.DataFrame, crop: str | None, gt_date):

    if crop == "Sugarcane":
        lc = _sugarcane_lifecycle(farm_df, gt_date)
        min_relative_drop = 0.55  # sugarcane tolerates a deeper natural mid-season dip before we call it "two crops"
        params = SUGARCANE_TROUGH_PARAMS
    else:
        params = CROP_PARAMS.get(crop, DEFAULT_PARAMS)
        candidates = _raw_lifecycles(farm_df, **params)
        lc = _select_lifecycle_by_ground_truth(candidates, gt_date)
        if lc is None:
            return None

        min_days, max_days = get_duration_bounds(crop)
        if lc["duration_days"] > max_days:
            full_start, full_end = farm_df["date"].iloc[0], farm_df["date"].iloc[-1]
            clipped = _clip_segment_to_duration(farm_df, lc["peak_date"], max_days, full_start, full_end)
            if clipped is not None:
                lc = clipped
        min_relative_drop = 0.35

    if lc is None:
        return None

    lc = _resplit_if_merged(
        farm_df, lc, gt_date,
        min_peak=params["min_peak"], min_len=params["min_len"], min_relative_drop=min_relative_drop,
    )
    return lc


def extract_best_effort_lifecycle(farm_df: pd.DataFrame, gt_date):

    out = {}
    for crop in CROPS:
        try:
            out[crop] = extract_ground_truth_lifecycle(farm_df, crop, gt_date)
        except Exception:
            out[crop] = None
    return out


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def _quartile_means(ndvi):
    n = len(ndvi)
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    edges = np.linspace(0, n, 5).astype(int)
    out = []
    for i in range(4):
        seg = ndvi[edges[i]: max(edges[i + 1], edges[i] + 1)]
        out.append(float(np.mean(seg)) if len(seg) else 0.0)
    return out


def extract_features(series_df: pd.DataFrame, lifecycle: dict) -> dict:
    ndvi = series_df["ndvi"].values.astype(float)
    day_offsets = (series_df["date"] - series_df["date"].iloc[0]).dt.days.values.astype(float)

    duration_days = lifecycle["duration_days"]
    peak_ndvi = lifecycle["peak_ndvi"]
    peak_date = lifecycle["peak_date"]
    start_ndvi, end_ndvi = float(ndvi[0]), float(ndvi[-1])
    min_ndvi, max_ndvi = float(np.min(ndvi)), float(np.max(ndvi))
    mean_ndvi, std_ndvi = float(np.mean(ndvi)), float(np.std(ndvi))

    days_to_peak = (peak_date - series_df["date"].iloc[0]).days
    days_from_peak = (series_df["date"].iloc[-1] - peak_date).days

    growth_rate = (peak_ndvi - start_ndvi) / days_to_peak if days_to_peak > 0 else 0.0
    decay_rate = (peak_ndvi - end_ndvi) / days_from_peak if days_from_peak > 0 else 0.0
    amplitude = peak_ndvi - min_ndvi
    peak_position_ratio = days_to_peak / duration_days if duration_days > 0 else 0.0

    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapz_fn(ndvi, day_offsets)) if len(ndvi) > 1 else 0.0
    auc_norm = auc / duration_days if duration_days > 0 else 0.0

    pct_negative = float(np.mean(ndvi < 0))
    pct_low = float(np.mean(ndvi < 0.20))

    smoothed = pd.Series(ndvi).rolling(3, center=True, min_periods=1).mean().values
    diffs = np.diff(smoothed)
    sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0)) if len(diffs) > 1 else 0

    q1, q2, q3, q4 = _quartile_means(ndvi)
    n_obs = len(ndvi)
    obs_density = n_obs / duration_days if duration_days > 0 else 0.0

    peak_month = peak_date.month
    start_month = series_df["date"].iloc[0].month
    end_month = series_df["date"].iloc[-1].month
    peak_month_sin = np.sin(2 * np.pi * peak_month / 12)
    peak_month_cos = np.cos(2 * np.pi * peak_month / 12)

    overall_slope = float(np.polyfit(day_offsets, ndvi, 1)[0]) if len(set(day_offsets)) > 1 else 0.0

    return {
        "duration_days": duration_days, "n_obs": n_obs, "obs_density": round(obs_density, 5),
        "peak_ndvi": peak_ndvi, "start_ndvi": round(start_ndvi, 4), "end_ndvi": round(end_ndvi, 4),
        "min_ndvi": round(min_ndvi, 4), "max_ndvi": round(max_ndvi, 4),
        "mean_ndvi": round(mean_ndvi, 4), "std_ndvi": round(std_ndvi, 4),
        "amplitude": round(amplitude, 4), "growth_rate": round(growth_rate, 6),
        "decay_rate": round(decay_rate, 6), "overall_slope": round(overall_slope, 6),
        "days_to_peak": days_to_peak, "days_from_peak": days_from_peak,
        "peak_position_ratio": round(peak_position_ratio, 4), "auc_norm": round(auc_norm, 4),
        "pct_negative": round(pct_negative, 4), "pct_low_ndvi": round(pct_low, 4),
        "internal_sign_changes": sign_changes,
        "q1_mean": round(q1, 4), "q2_mean": round(q2, 4), "q3_mean": round(q3, 4), "q4_mean": round(q4, 4),
        "peak_month": peak_month, "start_month": start_month, "end_month": end_month,
        "peak_month_sin": round(float(peak_month_sin), 4), "peak_month_cos": round(float(peak_month_cos), 4),
        "min_negative_flag": int(min_ndvi < 0),
    }


# --------------------------------------------------------------------------
# 4. NDVI -> plant-health classification (rule-based, not ML)
# --------------------------------------------------------------------------
def classify_ndvi_health(ndvi_value: float) -> dict:
    """
    Standard, widely-used NDVI health bands. Returns a label, a color
    (for the UI) and a short explanation.
    """
    if ndvi_value is None or np.isnan(ndvi_value):
        return {"label": "Unknown", "color": "#9CA3AF", "description": "No NDVI value available."}
    if ndvi_value < 0.0:
        return {"label": "Water / Cloud / No Vegetation", "color": "#60A5FA",
                "description": "Negative NDVI usually indicates water, cloud, snow, or built-up surface."}
    if ndvi_value < 0.2:
        return {"label": "Bare Soil / Very Poor", "color": "#B45309",
                "description": "Little to no live vegetation - bare soil, early growth, or severe stress."}
    if ndvi_value < 0.4:
        return {"label": "Sparse / Stressed Vegetation", "color": "#F59E0B",
                "description": "Sparse canopy cover or early growth stage; possible water/nutrient stress."}
    if ndvi_value < 0.6:
        return {"label": "Moderate Health", "color": "#EAB308",
                "description": "Developing canopy with moderate photosynthetic activity."}
    if ndvi_value < 0.8:
        return {"label": "Healthy / Vigorous Growth", "color": "#84CC16",
                "description": "Dense, healthy canopy - typical of a crop near its growth peak."}
    return {"label": "Very Healthy / Peak Vigor", "color": "#16A34A",
            "description": "Very dense, highly vigorous vegetation - at or near peak biomass."}