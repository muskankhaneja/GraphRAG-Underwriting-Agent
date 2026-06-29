"""
inference.py
-------------
Stateless, deterministic inference utility for the XGBoost credit risk model.

Usage
-----
from src.phase1_quantitative.inference import CreditRiskPredictor

predictor = CreditRiskPredictor()
result = predictor.predict({
    "currentRatio": 1.5,
    "debtEquityRatio": 0.8,
    ...  # all 25+ feature keys
})
# {
#   "probability_of_speculative": 0.23,
#   "predicted_class": 0,
#   "risk_tier": "Investment Grade",
#   "confidence": "High",
#   "model_version": "xgb_credit_model_v1"
# }
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Default artifact paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODEL_PATH = _ROOT / "models" / "xgb_credit_model.json"
_DEFAULT_SCALER_PATH = _ROOT / "models" / "scaler.pkl"
_DEFAULT_METRICS_PATH = _ROOT / "models" / "metrics.json"


class CreditRiskPredictor:
    """
    Load trained XGBoost credit model + StandardScaler and expose a
    single `predict` method that returns an interpretable risk assessment.

    Parameters
    ----------
    model_path  : Path to the XGBoost JSON artifact.
    scaler_path : Path to the pickled scaler bundle.
    threshold   : Decision threshold for the binary prediction.
                  Defaults to the CV-optimal threshold stored in metrics.json.
    """

    def __init__(
        self,
        model_path: str | Path = _DEFAULT_MODEL_PATH,
        scaler_path: str | Path = _DEFAULT_SCALER_PATH,
        threshold: float | None = None,
    ) -> None:
        model_path = Path(model_path)
        scaler_path = Path(scaler_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {model_path}. "
                "Run train_credit_model.py first."
            )
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found: {scaler_path}. "
                "Run data_engineering.py first."
            )

        # Load model
        self._model = XGBClassifier()
        self._model.load_model(model_path)

        # Load scaler bundle: {"scaler": StandardScaler, "feature_columns": [...], "train_medians": {...}}
        with open(scaler_path, "rb") as f:
            bundle = pickle.load(f)
        self._scaler = bundle["scaler"]
        self._feature_columns: list[str] = bundle["feature_columns"]
        self._train_medians: dict[str, float] = bundle["train_medians"]

        # Resolve threshold
        if threshold is not None:
            self._threshold = float(threshold)
        else:
            metrics_path = _DEFAULT_METRICS_PATH
            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                self._threshold = float(metrics.get("optimal_threshold", 0.5))
            else:
                self._threshold = 0.5

        # Model version
        metrics_path = _DEFAULT_METRICS_PATH
        if metrics_path.exists():
            with open(metrics_path) as f:
                self._model_version: str = json.load(f).get("model_version", "unknown")
        else:
            self._model_version = "unknown"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def predict(self, financial_ratios: dict[str, Any]) -> dict[str, Any]:
        """
        Score a single company's financial ratios.

        Parameters
        ----------
        financial_ratios : dict mapping feature name → numeric value.
                           All columns in self.feature_columns must be present.

        Returns
        -------
        dict with keys:
          probability_of_speculative : float  — P(class=1 | features)
          predicted_class            : int    — 0 or 1
          risk_tier                  : str    — "Investment Grade" / "Speculative Grade"
          confidence                 : str    — "High" / "Moderate" / "Low"
          model_version              : str
        """
        missing = [col for col in self._feature_columns if col not in financial_ratios]
        if missing:
            raise ValueError(
                f"Missing required feature(s): {missing}. "
                f"Expected all of: {self._feature_columns}"
            )

        # Build row in the exact training column order
        row: dict[str, float] = {}
        for col in self._feature_columns:
            val = financial_ratios[col]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = self._train_medians.get(col, 0.0)
            row[col] = float(val)

        X = pd.DataFrame([row], columns=self._feature_columns)
        X_scaled = self._scaler.transform(X)

        prob_speculative = float(self._model.predict_proba(X_scaled)[0, 1])
        predicted_class = int(prob_speculative >= self._threshold)
        risk_tier = "Speculative Grade" if predicted_class == 1 else "Investment Grade"
        confidence = self._confidence_label(prob_speculative)

        return {
            "probability_of_speculative": round(prob_speculative, 6),
            "predicted_class": predicted_class,
            "risk_tier": risk_tier,
            "confidence": confidence,
            "model_version": self._model_version,
        }

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _confidence_label(self, prob: float) -> str:
        """
        Confidence is determined by distance from the decision threshold.
          >= 0.25 away → High
          >= 0.10 away → Moderate
          < 0.10 away  → Low
        """
        distance = abs(prob - self._threshold)
        if distance >= 0.25:
            return "High"
        if distance >= 0.10:
            return "Moderate"
        return "Low"
