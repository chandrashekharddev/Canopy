
from __future__ import annotations

import io
import uuid

from dotenv import load_dotenv

load_dotenv()  # reads a .env file in the working directory, if present, into os.environ
from datetime import date
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from batch_service import parse_upload, results_to_dataframe, run_batch
from crop_core import CROPS
from inference_service import ModelNotTrainedError, predict_for_point

app = FastAPI(
    title="NDVI Crop Classifier",
    description="Predicts crop type and vegetation health from satellite NDVI time series (Sentinel-2 / Google Earth Engine).",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory store for the last few batch runs (job_id -> DataFrame).
# Fine for a single-instance demo/small-team deployment; swap for
# Redis/DB storage if you deploy this behind multiple workers.
_BATCH_RESULTS: dict[str, pd.DataFrame] = {}
_MAX_BATCH_ROWS = 200


# --------------------------------------------------------------------------
# Request / response models
# --------------------------------------------------------------------------
class SinglePredictionRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    target_date: date
    crop_hint: Optional[str] = Field(
        default=None, description="Optional. One of Tomato, Cotton, Paddy, Sugarcane, Onion."
    )
    farm_id: Optional[str] = None


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"crops": CROPS})


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    model_ready = True
    model_error = None
    try:
        from inference_service import _load_model
        _load_model()
    except ModelNotTrainedError as e:
        model_ready = False
        model_error = str(e)

    gee_ready = True
    gee_error = None
    try:
        from gee_service import ee_initialize
        ee_initialize()
    except Exception as e:  # noqa: BLE001 - surfaced to the user as a status flag
        gee_ready = False
        gee_error = str(e)

    return {
        "status": "ok",
        "model_ready": model_ready,
        "model_error": model_error,
        "gee_ready": gee_ready,
        "gee_error": gee_error,
    }


@app.get("/api/crops")
def list_crops():
    return {"crops": CROPS}


@app.post("/api/predict/single")
def predict_single(payload: SinglePredictionRequest):
    try:
        result = predict_for_point(
            latitude=payload.latitude,
            longitude=payload.longitude,
            target_date=payload.target_date.isoformat(),
            crop_hint=payload.crop_hint,
            farm_id=payload.farm_id or "Point_1",
        )
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Earth Engine / prediction error: {e}") from e

    if result.get("status") == "failed":
        # Still 200 - it's a valid, well-formed "no data" answer, not a server error.
        return result
    return result


@app.post("/api/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    file_bytes = await file.read()
    try:
        df = parse_upload(file_bytes, file.filename)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e)) from e

    if df.empty:
        raise HTTPException(status_code=400, detail="No valid rows found (check Latitude/Longitude/Date columns).")
    if len(df) > _MAX_BATCH_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"File has {len(df)} rows; batch is capped at {_MAX_BATCH_ROWS} per upload to keep GEE calls responsive.",
        )

    try:
        results = run_batch(df)
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    result_df = results_to_dataframe(results)
    job_id = uuid.uuid4().hex[:12]
    _BATCH_RESULTS[job_id] = result_df

    n_ok = int((result_df["Status"] == "ok").sum())
    return {
        "job_id": job_id,
        "n_total": len(results),
        "n_ok": n_ok,
        "n_failed": len(results) - n_ok,
        "results": results,
    }


@app.get("/api/predict/batch/{job_id}/csv")
def download_batch_csv(job_id: str):
    if job_id not in _BATCH_RESULTS:
        raise HTTPException(status_code=404, detail="Unknown or expired job_id.")
    df = _BATCH_RESULTS[job_id]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=crop_predictions_{job_id}.csv"},
    )
