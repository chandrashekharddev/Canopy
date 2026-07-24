from __future__ import annotations

import io

import pandas as pd

from inference_service import predict_for_point

COLUMN_ALIASES = {
    "farm_id": ["farm_id", "farm", "id", "farmid", "name"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "long", "lng"],
    "target_date": [
        "groundtruth_date", "ground_truth_date", "ground_truth_month", "groundtruth_month",
        "date", "target_date", "gt_date", "gt",
    ],
    "crop_hint": ["crop", "crop_hint", "crop_label", "expected_crop"],
}


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower().strip().replace(" ", "_"): c for c in columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def parse_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Reads a CSV or Excel upload and normalizes it to columns:
    farm_id, latitude, longitude, target_date, crop_hint (crop_hint optional/NaN)."""
    lower_name = filename.lower()
    if lower_name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))

    columns = df.columns.tolist()
    resolved = {}
    for key, aliases in COLUMN_ALIASES.items():
        col = _find_column(columns, aliases)
        resolved[key] = col

    if resolved["latitude"] is None or resolved["longitude"] is None or resolved["target_date"] is None:
        raise ValueError(
            "Could not find required columns. Your file must include Latitude, Longitude, "
            "and a ground-truth/target Date column (Farm_ID and Crop are optional)."
        )

    out = pd.DataFrame()
    out["farm_id"] = df[resolved["farm_id"]] if resolved["farm_id"] else [f"Point_{i+1}" for i in range(len(df))]
    out["latitude"] = pd.to_numeric(df[resolved["latitude"]], errors="coerce")
    out["longitude"] = pd.to_numeric(df[resolved["longitude"]], errors="coerce")
    out["target_date"] = pd.to_datetime(df[resolved["target_date"]], dayfirst=True, errors="coerce")
    out["crop_hint"] = df[resolved["crop_hint"]] if resolved["crop_hint"] else None

    out = out.dropna(subset=["latitude", "longitude", "target_date"])
    out = out[out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)]
    return out.reset_index(drop=True)


def run_batch(df: pd.DataFrame) -> list[dict]:
    results = []
    for _, row in df.iterrows():
        crop_hint = row["crop_hint"] if pd.notna(row.get("crop_hint")) else None
        try:
            res = predict_for_point(
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                target_date=row["target_date"].strftime("%Y-%m-%d"),
                crop_hint=crop_hint,
                farm_id=str(row["farm_id"]),
            )
        except Exception as e:
            res = {
                "farm_id": str(row["farm_id"]),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "target_date": row["target_date"].strftime("%Y-%m-%d"),
                "status": "failed",
                "error": str(e),
            }
        results.append(res)
    return results


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        closest = r.get("closest_match") or {}
        rows.append({
            "Farm_ID": r.get("farm_id"),
            "Latitude": r.get("latitude"),
            "Longitude": r.get("longitude"),
            "Target_Date": r.get("target_date"),
            "Status": r.get("status"),
            "Predicted_Crop": r.get("predicted_crop"),
            "Confidence": r.get("confidence"),
            "Closest_Match_Crop": closest.get("crop"),
            "Closest_Match_Confidence": closest.get("confidence"),
            "Lifecycle_Start": (r.get("lifecycle") or {}).get("start_date"),
            "Lifecycle_End": (r.get("lifecycle") or {}).get("end_date"),
            "Duration_Days": (r.get("lifecycle") or {}).get("duration_days"),
            "Duration_Plausible": (r.get("lifecycle") or {}).get("duration_plausible"),
            "Peak_NDVI": (r.get("lifecycle") or {}).get("peak_ndvi"),
            "Peak_Health": ((r.get("health") or {}).get("at_peak") or {}).get("label"),
            "Nearest_NDVI": ((r.get("health") or {}).get("nearest_to_target_date") or {}).get("ndvi"),
            "Nearest_Health": ((r.get("health") or {}).get("nearest_to_target_date") or {}).get("label"),
            "Message": r.get("message"),
            "Error": r.get("error"),
        })
    return pd.DataFrame(rows)