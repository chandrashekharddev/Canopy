import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from crop_core import FEATURE_COLS, extract_features, extract_ground_truth_lifecycle

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# --------------------------------------------------------------------------
def load_data(path: str):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {}
    for c in df.columns:
        if c in ("farm_id", "farm"):
            rename_map[c] = "farm"
        elif c in ("crop label", "crop"):
            rename_map[c] = "crop"
        elif c in ("ground truth month", "ground_truth_month", "gt"):
            rename_map[c] = "gt"
    df = df.rename(columns=rename_map)

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["gt"] = pd.to_datetime(df["gt"], dayfirst=True)

    farms = {}
    for crop, crop_df in df.groupby("crop"):
        farms[crop] = {}
        for farm_name, farm_df in crop_df.groupby("farm"):
            farms[crop][farm_name] = farm_df.sort_values("date").reset_index(drop=True)
    return farms


def build_all_lifecycles(farms: dict):
    results = {}
    for crop, farm_dict in farms.items():
        results[crop] = {}
        for farm_name, farm_df in farm_dict.items():
            gt_date = farm_df["gt"].iloc[0] if "gt" in farm_df.columns else None
            lc = extract_ground_truth_lifecycle(farm_df, crop, gt_date)
            if lc is None:
                continue
            mask = (farm_df["date"] >= lc["start_date"]) & (farm_df["date"] <= lc["end_date"])
            series = farm_df.loc[mask].reset_index(drop=True)
            results[crop][farm_name] = {"lifecycle": lc, "series": series, "gt": gt_date}
    return results


def build_feature_dataframe(lifecycle_results: dict) -> pd.DataFrame:
    records = []
    for crop, farm_dict in lifecycle_results.items():
        for farm_name, d in farm_dict.items():
            feats = extract_features(d["series"], d["lifecycle"])
            feats["crop"] = crop
            feats["farm"] = farm_name
            records.append(feats)
    return pd.DataFrame(records)[["crop", "farm"] + FEATURE_COLS]


def get_candidate_models(random_state=RANDOM_STATE):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=random_state),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "NaiveBayes": GaussianNB(),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=300, eval_metric="mlogloss", random_state=random_state)
    return models


# --------------------------------------------------------------------------
def main(data_path: str, out_path: str):
    print(f"Loading {data_path} ...")
    farms = load_data(data_path)
    print("Crops found:", list(farms.keys()))
    for crop, fd in farms.items():
        print(f"  {crop:<10s}: {len(fd)} farms")

    print("\nExtracting ground-truth-validated lifecycles ...")
    lifecycle_results = build_all_lifecycles(farms)
    n_extracted = sum(len(v) for v in lifecycle_results.values())
    n_total = sum(len(v) for v in farms.values())
    print(f"Extracted {n_extracted}/{n_total} lifecycles.")

    print("\nEngineering features ...")
    feature_df = build_feature_dataframe(lifecycle_results)
    print("Feature matrix:", feature_df.shape)

    X_all = feature_df[FEATURE_COLS].values
    y_all = feature_df["crop"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    print("\nComparing candidate models (5-fold stratified CV) ...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in get_candidate_models().items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        acc = cross_val_score(pipe, X_all, y_enc, cv=skf, scoring="accuracy")
        f1 = cross_val_score(pipe, X_all, y_enc, cv=skf, scoring="f1_macro")
        rows.append({"model": name, "cv_accuracy_mean": acc.mean(), "cv_f1_macro_mean": f1.mean()})
        print(f"  {name:<20s} acc={acc.mean():.4f}  f1_macro={f1.mean():.4f}")

    comparison_df = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    cv_winner = comparison_df.iloc[0]["model"]
    print(f"\nBest model (by CV accuracy): {cv_winner}")


    selected_name = "RandomForest"
    if cv_winner != selected_name:
        print(f"Note: CV winner was '{cv_winner}', but pinning production model to '{selected_name}' as requested.")
    selected_row = comparison_df.loc[comparison_df["model"] == selected_name].iloc[0]

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_enc, test_size=0.25, stratify=y_enc, random_state=RANDOM_STATE
    )
    holdout_pipe = Pipeline([("scaler", StandardScaler()), ("clf", get_candidate_models()[selected_name])])
    holdout_pipe.fit(X_train, y_train)
    y_pred = holdout_pipe.predict(X_test)
    print(f"Held-out test accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Held-out test macro-F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print(f"Refitting {selected_name} on ALL data ...")
    production_pipeline = Pipeline([("scaler", StandardScaler()), ("clf", get_candidate_models()[selected_name])])
    production_pipeline.fit(X_all, y_enc)

    joblib.dump({
        "pipeline": production_pipeline,
        "label_encoder": le,
        "feature_cols": FEATURE_COLS,
        "model_name": selected_name,
        "cv_accuracy": float(selected_row["cv_accuracy_mean"]),
        "holdout_accuracy": float(accuracy_score(y_test, y_pred)),
    }, out_path)
    print(f"\nSaved production model ({selected_name}) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="all_5_crops_combined.csv")
    parser.add_argument("--out", default="model/crop_classifier.joblib")
    args = parser.parse_args()
    main(args.data, args.out)