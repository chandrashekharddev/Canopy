from __future__ import annotations

import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from crop_core import (
    CROPS,
    add_smoothed_ndvi,
    classify_ndvi_health,
    extract_features,
    extract_ground_truth_lifecycle,
    get_duration_bounds,
    is_duration_plausible,
)
from gee_service import extract_ndvi_series

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model", "crop_classifier.joblib"))

UNKNOWN_CONFIDENCE_THRESHOLD = float(os.environ.get("CROP_CONFIDENCE_THRESHOLD", 0.55))


class ModelNotTrainedError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise ModelNotTrainedError(
            f"No trained model found at {MODEL_PATH}. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


def _predict_for_lifecycle(series: pd.DataFrame, lifecycle: dict) -> dict:
    bundle = _load_model()
    pipe, label_enc, feat_cols = bundle["pipeline"], bundle["label_encoder"], bundle["feature_cols"]

    feats = extract_features(series, lifecycle)
    X = pd.DataFrame([feats])[feat_cols].values

    pred_idx = pipe.predict(X)[0]
    pred_crop = label_enc.inverse_transform([pred_idx])[0]

    proba = None
    top_prob = None
    if hasattr(pipe, "predict_proba"):
        proba = dict(zip(label_enc.classes_, [round(float(p), 4) for p in pipe.predict_proba(X)[0]]))
        top_prob = proba[pred_crop]

    return {"predicted_crop": pred_crop, "probabilities": proba, "confidence": top_prob, "features": feats}


def _nearest_ndvi(series: pd.DataFrame, target_date: pd.Timestamp) -> dict | None:
    if series.empty:
        return None
    idx = (series["date"] - target_date).abs().idxmin()
    row = series.loc[idx]
    return {"date": row["date"].strftime("%Y-%m-%d"), "ndvi": float(row["ndvi"])}


def _serialize_series(series: pd.DataFrame) -> list[dict]:
    """Raw + running-average-smoothed NDVI series, ready for JSON/the UI chart."""
    smoothed = add_smoothed_ndvi(series)
    smoothed = smoothed.assign(date=smoothed["date"].dt.strftime("%Y-%m-%d"))
    return smoothed[["date", "ndvi", "ndvi_smooth"]].to_dict("records")


def predict_for_point(
    latitude: float,
    longitude: float,
    target_date: str,
    crop_hint: str | None = None,
    farm_id: str | None = None,
) -> dict:
    """
    Full pipeline for ONE point: GEE extraction -> lifecycle -> features -> prediction -> health.

    If crop_hint is given (e.g. "Sugarcane"), it is used both to pick the correct
    GEE extraction window (+/-12 months for sugarcane, +/-6 for everything else)
    AND the correct trough-detection parameters for lifecycle extraction.

    If crop_hint is None, this is the true "unknown crop" production case: we try
    every crop's lifecycle-extraction parameters (with the standard +/-6 month
    window), generate features for each, run the classifier on each candidate,
    and keep whichever candidate the classifier is most confident about.
    """
    gt_date = pd.to_datetime(target_date)

    if crop_hint:
        crop_hint = crop_hint.strip().title()

    raw_series = extract_ndvi_series(latitude, longitude, gt_date, crop_hint=crop_hint)
    if raw_series.empty:
        return {
            "farm_id": farm_id,
            "latitude": latitude,
            "longitude": longitude,
            "target_date": gt_date.strftime("%Y-%m-%d"),
            "status": "failed",
            "error": "No cloud-free Sentinel-2 imagery found for this location/date window "
                     "(try a different date or check the coordinates).",
        }

    result_candidates = []

    if crop_hint and crop_hint in CROPS:
        lc = extract_ground_truth_lifecycle(raw_series, crop_hint, gt_date)
        if lc is not None:
            mask = (raw_series["date"] >= lc["start_date"]) & (raw_series["date"] <= lc["end_date"])
            series = raw_series.loc[mask].reset_index(drop=True)
            pred = _predict_for_lifecycle(series, lc)
            result_candidates.append({"lifecycle": lc, "series": series, **pred})
    else:
        for crop in CROPS:
            try:
                lc = extract_ground_truth_lifecycle(raw_series, crop, gt_date)
            except Exception:
                lc = None
            if lc is None:
                continue
            mask = (raw_series["date"] >= lc["start_date"]) & (raw_series["date"] <= lc["end_date"])
            series = raw_series.loc[mask].reset_index(drop=True)
            if len(series) < 3:
                continue
            try:
                pred = _predict_for_lifecycle(series, lc)
            except Exception:
                continue
            result_candidates.append({"lifecycle": lc, "series": series, **pred})

    if not result_candidates:
        return {
            "farm_id": farm_id,
            "latitude": latitude,
            "longitude": longitude,
            "target_date": gt_date.strftime("%Y-%m-%d"),
            "status": "failed",
            "error": "NDVI signal too weak/noisy to extract a valid crop lifecycle for this point.",
            "ndvi_series": _serialize_series(raw_series),
        }

    best = max(result_candidates, key=lambda r: (r["confidence"] or 0.0))
    lc = best["lifecycle"]

    nearest = _nearest_ndvi(raw_series, gt_date)
    peak_health = classify_ndvi_health(lc["peak_ndvi"])
    current_health = classify_ndvi_health(nearest["ndvi"]) if nearest else None

    min_d, max_d = get_duration_bounds(best["predicted_crop"])
    duration_plausible = is_duration_plausible(best["predicted_crop"], lc["duration_days"])

    lifecycle_payload = {
        "start_date": lc["start_date"].strftime("%Y-%m-%d"),
        "end_date": lc["end_date"].strftime("%Y-%m-%d"),
        "duration_days": lc["duration_days"],
        "peak_date": lc["peak_date"].strftime("%Y-%m-%d"),
        "peak_ndvi": lc["peak_ndvi"],
        "expected_duration_days": {"min": min_d, "max": max_d},
        "duration_plausible": duration_plausible,
    }
    health_payload = {
        "at_peak": peak_health,
        "nearest_to_target_date": {**current_health, "date": nearest["date"], "ndvi": nearest["ndvi"]}
        if current_health else None,
    }

    # Low confidence -> don't force a label onto one of the 5 supported crops.
    if (best["confidence"] or 0.0) < UNKNOWN_CONFIDENCE_THRESHOLD:
        return {
            "farm_id": farm_id,
            "latitude": latitude,
            "longitude": longitude,
            "target_date": gt_date.strftime("%Y-%m-%d"),
            "status": "unknown",
            "predicted_crop": None,
            "closest_match": {"crop": best["predicted_crop"], "confidence": best["confidence"]},
            "message": (
                f"This location/date's NDVI pattern doesn't clearly match any of the 5 supported "
                f"crops (Tomato, Cotton, Paddy, Sugarcane, Onion). Closest match was "
                f"{best['predicted_crop']} at only "
                f"{(best['confidence'] or 0) * 100:.1f}% confidence - too low to report as a reliable "
                f"prediction."
            ),
            "confidence": best["confidence"],
            "probabilities": best["probabilities"],
            "crop_hint_used": crop_hint,
            "lifecycle": lifecycle_payload,
            "health": health_payload,
            "ndvi_series": _serialize_series(raw_series),
            "n_images_used": int(len(raw_series)),
        }

    return {
        "farm_id": farm_id,
        "latitude": latitude,
        "longitude": longitude,
        "target_date": gt_date.strftime("%Y-%m-%d"),
        "status": "ok",
        "predicted_crop": best["predicted_crop"],
        "confidence": best["confidence"],
        "probabilities": best["probabilities"],
        "crop_hint_used": crop_hint,
        "lifecycle": lifecycle_payload,
        "health": health_payload,
        "ndvi_series": _serialize_series(raw_series),
        "n_images_used": int(len(raw_series)),
    }