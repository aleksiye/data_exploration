"""Loading the trained model together with its model card."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import Settings
from .features import MODEL_FEATURES, build_features

log = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    model: object
    card: dict

    @property
    def version(self) -> str:
        return self.card["model_version"]

    @property
    def threshold(self) -> float:
        return self.card["selection_policy"]["threshold"]

    def score(self, alerts: pd.DataFrame) -> np.ndarray:
        """P(Ia) for each row of alert fields."""
        return self.model.predict_proba(build_features(alerts))[:, 1]

    @classmethod
    def load(cls, settings: Settings) -> ModelBundle:
        card = json.loads(settings.model_card_path.read_text(encoding="utf-8"))
        if card["features"] != MODEL_FEATURES:
            raise RuntimeError("model was trained on different features than this code builds; retrain it")
        if card["sklearn_version"] != sklearn.__version__:
            log.warning("model trained with scikit-learn %s, running %s", card["sklearn_version"], sklearn.__version__)
        bundle = cls(model=joblib.load(settings.model_path), card=card)
        log.info("loaded model %s (threshold P(Ia) >= %.3f)", bundle.version, bundle.threshold)
        return bundle
