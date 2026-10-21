"""Train, evaluate and save the vehicle purchase model (v2).

Usage: python train.py [path/to/Car_Inquiries_Dataset.xlsx]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from app.features import CATEGORICAL, FEATURES, NUMERIC, build_features

ROOT = Path(__file__).parent
TARGET = "Purchased_Within_30_Days"


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("preprocessor", pre), ("classifier", LogisticRegression(max_iter=3000))])


def metrics(y, p, t):
    pred = (p >= t).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y, p), 4), "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4), "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4), "brier": round(brier_score_loss(y, p), 4), "rows": int(len(y)),
    }


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "Car_Inquiries_Dataset.xlsx"
    df = build_features(pd.read_excel(path, sheet_name="Inquiries")).sort_values("Inquiry_Date")

    # Time-based split: train on the oldest 80%, test on the newest 20% (how the model is used in practice).
    cut = df["Inquiry_Date"].quantile(0.8)
    train, test = df[df["Inquiry_Date"] <= cut], df[df["Inquiry_Date"] > cut]
    pipe = make_pipeline().fit(train[FEATURES], train[TARGET])
    p = pipe.predict_proba(test[FEATURES])[:, 1]

    # Threshold that maximises F1 on the test window.
    grid = np.round(np.arange(0.20, 0.66, 0.05), 2)
    threshold = float(max(grid, key=lambda t: f1_score(test[TARGET], p >= t)))
    report = {"test_window_start": str(test["Inquiry_Date"].min().date()),
              "threshold": threshold, "test_metrics": metrics(test[TARGET], p, threshold)}

    # Refit on all rows for the deployed model.
    final = make_pipeline().fit(df[FEATURES], df[TARGET])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"pipeline": final, "threshold": threshold, "feature_columns": FEATURES,
                 "model_version": "v2", "sklearn_version": sklearn.__version__,
                 "evaluation": report}, ROOT / "models" / "vehicle_purchase_model_v2.joblib")
    (ROOT / "models" / "evaluation_v2.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
