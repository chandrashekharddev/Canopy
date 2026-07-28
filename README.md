# Canopy — NDVI Crop Classifier

A FastAPI application that predicts **crop type** (Tomato / Cotton / Paddy / Sugarcane / Onion)
and **vegetation health** from a location + date, using live Sentinel-2 NDVI data pulled through
the **Google Earth Engine (GEE) Python API**, and a trained crop-lifecycle classifier.

It reproduces your GEE JavaScript extraction script and your notebook's lifecycle-extraction /
feature-engineering / classification pipeline exactly — just wired into a web app instead of the
Code Editor + a static CSV.

```
Latitude, Longitude, Date  ──►  GEE: Sentinel-2 NDVI time series  ──►  lifecycle extraction
                                                                             │
                        health readout  ◄──  trained classifier  ◄──  feature engineering
```

---


## 1. Project structure

```
app/
├── main.py                 FastAPI app + all HTTP endpoints
├── crop_core.py             Shared lifecycle-extraction + feature-engineering + NDVI health rules
├── gee_service.py           Google Earth Engine NDVI extraction (Python port of your JS script)
├── inference_service.py     Glue: GEE → lifecycle → features → model → health
├── batch_service.py         CSV/Excel upload parsing + batch runner
├── train_model.py           Offline training script (reproduces the notebook)
├── model/
│   └── crop_classifier.joblib   Trained model (already built for you — see §4)
├── templates/
│   └── index.html           Web UI
├── static/
│   ├── style.css
│   └── app.js
├── requirements.txt
└── .env.example
```

---

## 3. Google Earth Engine setup (do this once)

You need a Google Cloud project with the Earth Engine API enabled, and a **service account** so
the backend can authenticate without any interactive login.

1. **Create/choose a Google Cloud project** and enable the Earth Engine API:
   `https://console.cloud.google.com/apis/library/earthengine.googleapis.com`

2. **Create a service account**: IAM & Admin → Service Accounts → Create Service Account.
   Grant it the **"Earth Engine Resource Viewer"** role (or "Editor" if you also export assets).

3. **Create a JSON key** for that service account and download it
   (Service Accounts → your account → Keys → Add Key → JSON).

4. **Register the service account for Earth Engine access** (one-time, per Google account/project):
   `https://signup.earthengine.google.com/#!/service_accounts`

5. Save the JSON key somewhere safe, e.g. `secrets/gee-service-account.json`, and set environment
   variables (copy `.env.example` → `.env` and fill in, or export them directly):

   `gee_service.py` reads these automatically and initializes Earth Engine on first use — nothing
   else to wire up. If you don't set them, it falls back to whatever `earthengine authenticate`
   has cached locally (fine for local dev, not for a deployed server).

---

## 4. The trained model

`model/crop_classifier.joblib` has already been trained for you on `all_5_crops_combined.csv`
using **exactly** the pipeline from your notebook (ground-truth-aware lifecycle extraction →
30 engineered features → 5-fold CV across 7-8 classifier families → best model refit on all data).

- **Best model selected:** XGBoost
- **5-fold CV accuracy:** ~96.5%
- **Held-out test accuracy:** ~96%

Retrain any time you have new/expanded ground-truth data:

```bash
python train_model.py --data all_5_crops_combined.csv --out model/crop_classifier.joblib
```

---

## 5. Running the app

```bash
python -m venv venv && source venv/bin/activate      # optional but recommended
pip install -r requirements.txt

# GEE credentials (see §3)
export GEE_SERVICE_ACCOUNT_EMAIL="..."
export GEE_PRIVATE_KEY_FILE="secrets/gee-service-account.json"
export GEE_PROJECT="..."

uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`. The top bar shows two live status pills — **model** and
**earth engine** — so you immediately know if either isn't configured correctly.

---

## 6. Using the app

### Single point
Enter latitude, longitude, and the date you care about (the "target"/ground-truth date). Optionally
pick a **crop hint** — if you already suspect the crop (e.g. validating a known Sugarcane field),
giving the hint (a) selects the correct ±12 vs ±6 month GEE window and (b) uses that crop's exact
trough-detection parameters, which improves both speed and accuracy. Leave it on "Let the model
decide" for the real unknown-crop case — the app then tries every crop's lifecycle extraction, runs
the classifier on each valid candidate, and returns the one the model is most confident about.

You get back:
- **Predicted crop** + confidence + full probability breakdown
- The **extracted lifecycle window** (start/end/peak dates, duration, peak NDVI)
- **Health at peak growth** and **health nearest to your target date**, using standard NDVI bands
- An inline NDVI time-series chart with the lifecycle window shaded, peak marked, and your target
  date marked

### Batch upload
Upload a CSV or Excel file (up to 200 rows). The parser recognizes flexible column names:

| Needed              | Accepted header names (case-insensitive)                          |
|----------------------|--------------------------------------------------------------------|
| Farm ID *(optional)* | `Farm_ID`, `Farm`, `ID`, `Name`                                     |
| Latitude *(required)*| `Latitude`, `Lat`                                                   |
| Longitude *(required)*| `Longitude`, `Lon`, `Long`, `Lng`                                  |
| Date *(required)*    | `GroundTruth_Date`, `Ground Truth Month`, `Date`, `Target_Date`, `GT` |
| Crop *(optional)*    | `Crop`, `Crop_Hint`, `Crop Label`                                   |

Results show inline as a table with a status per row (`ok` / `failed` + reason), and a
**Download CSV** button for the full result set.

---

## 7. API reference

| Method | Path                              | Description                                   |
|--------|------------------------------------|------------------------------------------------|
| GET    | `/api/health`                      | Model / Earth Engine readiness check           |
| GET    | `/api/crops`                       | List of recognized crops                       |
| POST   | `/api/predict/single`              | `{latitude, longitude, target_date, crop_hint?}` → prediction JSON |
| POST   | `/api/predict/batch`               | multipart file upload → `{job_id, results: [...]}` |
| GET    | `/api/predict/batch/{job_id}/csv`  | Download batch results as CSV                  |

Interactive API docs (Swagger UI) are auto-generated at `/docs`.

---

## 8. Notes on accuracy & production hardening

- **Cloud-free imagery availability** ultimately limits accuracy — a location/date with very few
  usable Sentinel-2 scenes (heavy monsoon cloud cover, etc.) yields a noisier NDVI series and a
  less confident lifecycle/classification. The response always tells you `n_images_used`.
- **Batch is synchronous** and capped at 200 rows per upload to keep response times reasonable
  (each row makes a live GEE call). For larger batches, wire this into a background task queue
  (Celery/RQ) and poll a job-status endpoint instead — the core `run_batch()` function in
  `batch_service.py` doesn't need to change, just how it's invoked.
- **In-memory batch result storage**: `_BATCH_RESULTS` in `main.py` is a simple dict, fine for a
  single-process deployment. Swap for Redis/a database if you run multiple workers behind a load
  balancer, since each worker would otherwise only see its own jobs.
