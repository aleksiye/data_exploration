"""Train the classifier and choose the probability threshold for the target precision."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .config import Settings
from .features import MODEL_FEATURES, build_features
from .prepare import file_sha256

log = logging.getLogger(__name__)

# Tuned with cross-validation in notebook 04, section 4.
HYPERPARAMETERS = {"learning_rate": 0.1, "max_leaf_nodes": 7, "min_samples_leaf": 100, "random_state": 0}


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**HYPERPARAMETERS)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, target_precision: float) -> tuple[float, float, float]:
    """Lowest threshold above which precision stays at or above the target.

    Precision isn't monotonic in the threshold, so taking the first threshold that touches the target
    can land on a lucky bump. Requiring every higher threshold to meet it too is more robust.
    Returns (threshold, precision, recall) at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    meets = precision[:-1] >= target_precision
    if not meets[-1]:
        raise ValueError(f"no threshold reaches precision {target_precision}")
    failing = np.flatnonzero(~meets)
    i = failing[-1] + 1 if failing.size else 0
    return float(thresholds[i]), float(precision[i]), float(recall[i])


def train(settings: Settings) -> dict:
    data = pd.read_parquet(settings.train_path)
    X, y = build_features(data), data["is_ia"].astype(int).to_numpy()

    # Out-of-fold predictions: every training object is scored by a model that never saw it.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]
    threshold, precision, recall = choose_threshold(y, oof, settings.target_precision)

    model = make_model().fit(X, y)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, settings.model_path)

    data_hash = file_sha256(settings.train_path)
    trained_at = datetime.now(timezone.utc)
    card = {
        "model_version": f"hgb-{trained_at:%Y%m%dT%H%M%SZ}-{data_hash[:8]}",
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "algorithm": "HistGradientBoostingClassifier",
        "hyperparameters": HYPERPARAMETERS,
        "sklearn_version": sklearn.__version__,
        "features": MODEL_FEATURES,
        "training_data": {"path": str(settings.train_path), "sha256": data_hash, "objects": len(y),
                          "ia_fraction": round(float(y.mean()), 4)},
        "selection_policy": {"target_precision": settings.target_precision, "threshold": round(threshold, 6),
                             "daily_capacity": settings.daily_capacity},
        "validation": {"method": "5-fold out-of-fold predictions on the training data",
                       "roc_auc": round(float(roc_auc_score(y, oof)), 4),
                       "precision_at_threshold": round(precision, 4),
                       "recall_at_threshold": round(recall, 4)},
    }
    settings.model_card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    log.info("model %s: out-of-fold ROC AUC %.3f; P(Ia) >= %.3f gives precision %.3f, recall %.3f",
             card["model_version"], card["validation"]["roc_auc"], threshold, precision, recall)
    return card
