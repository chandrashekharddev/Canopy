from __future__ import annotations

import os
import threading
from datetime import date, datetime

import ee
import pandas as pd

CLOUD_LIMIT = 40         # max scene-level cloud %
BUFFER_SIZE = 50          # metres - filtering only, never sampled
PIXEL_SCALE = 10          # Sentinel-2 native resolution

WINDOW_MONTHS = 12


def get_window_months(crop_hint: str | None) -> int:
    return WINDOW_MONTHS


_init_lock = threading.Lock()
_initialized = False


def ee_initialize():
    """Idempotent, thread-safe Earth Engine initialization using a service account."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return
        email = os.environ.get("GEE_SERVICE_ACCOUNT_EMAIL")
        key_file = os.environ.get("GEE_PRIVATE_KEY_FILE")
        project = os.environ.get("GEE_PROJECT")

        if email and key_file:
            credentials = ee.ServiceAccountCredentials(email, key_file)
            ee.Initialize(credentials, project=project)
        else:
          
            ee.Initialize(project=project)
        _initialized = True


def _add_ndvi(image: "ee.Image") -> "ee.Image":
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)

_SCL_EXCLUDE = [0, 1, 3, 8, 9, 10, 11]


def _mask_clouds_scl(image: "ee.Image") -> "ee.Image":
    scl = image.select("SCL")
    good_pixel = scl.remap(_SCL_EXCLUDE, [0] * len(_SCL_EXCLUDE), 1)
    return image.updateMask(good_pixel)


def _tag_date(image: "ee.Image") -> "ee.Image":
    return image.set("Date", image.date().format("YYYY-MM-dd"))


def extract_ndvi_series(
    latitude: float,
    longitude: float,
    ground_truth_date: "date | datetime | str",
    crop_hint: str | None = None,
) -> pd.DataFrame:
   
    ee_initialize()

    if isinstance(ground_truth_date, str):
        gt_date = pd.to_datetime(ground_truth_date)
    else:
        gt_date = pd.to_datetime(ground_truth_date)

    window = get_window_months(crop_hint)

    point = ee.Geometry.Point([longitude, latitude])
    buffer_geom = point.buffer(BUFFER_SIZE)

    gt_date_str = gt_date.strftime("%Y-%m-%d")
    ee_gt_date = ee.Date(gt_date_str)
    start_date = ee_gt_date.advance(-window, "month")
    end_date = ee_gt_date.advance(window, "month")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(buffer_geom)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUD_LIMIT))
        .map(_tag_date)
    )


    collection = ee.ImageCollection(collection.sort("CLOUDY_PIXEL_PERCENTAGE").distinct("Date"))
    collection = collection.sort("system:time_start").map(_mask_clouds_scl).map(_add_ndvi)

    def _sample(image):
        pixel = image.select("NDVI").sample(
            region=point, scale=PIXEL_SCALE, numPixels=1, geometries=False, dropNulls=True
        )
        return pixel.map(lambda ft: ft.set({
            "Image_Date": image.date().format("YYYY-MM-dd"),
            "Timestamp": image.get("system:time_start"),
            "Cloud_Percentage": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "Image_ID": image.get("system:index"),
            "Satellite": image.get("SPACECRAFT_NAME"),
            "Tile": image.get("MGRS_TILE"),
        }))

    samples = ee.FeatureCollection(collection.map(_sample)).flatten()

    feats = samples.getInfo()["features"]

    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({
            "date": p["Image_Date"],
            "ndvi": p.get("NDVI"),
            "cloud_percentage": p.get("Cloud_Percentage"),
            "image_id": p.get("Image_ID"),
            "satellite": p.get("Satellite"),
            "tile": p.get("Tile"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["ndvi"]).sort_values("date").reset_index(drop=True)
    return df