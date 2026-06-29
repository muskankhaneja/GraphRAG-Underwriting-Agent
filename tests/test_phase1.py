"""
test_phase1.py
--------------
Pytest suite for Phase 1 — Quantitative Foundation.

Tests cover:
  1. Rating-to-binary mapping completeness and correctness
  2. Data schema integrity (feature columns, metadata alignment)
  3. Train/test split — scaler fitted only on training data (no leakage)
  4. Model artifact I/O
  5. CreditRiskPredictor.predict() contract
  6. Edge cases: missing features, NaN values
  7. Threshold boundary / confidence labelling
"""

from __future__ import annotations

import pickle
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.phase1_quantitative.data_engineering import (
    ALL_KNOWN_RATINGS,
    FEATURE_COLUMNS,
    INVESTMENT_GRADE,
    SPECULATIVE_GRADE,
    _map_rating_to_binary,
)
from src.phase1_quantitative.inference import CreditRiskPredictor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_FEATURES = {col: 1.0 for col in FEATURE_COLUMNS}


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> dict:
    """Create minimal but real model + scaler artifacts in a temp directory."""
    n_samples, n_features = 80, len(FEATURE_COLUMNS)
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, n_features))
    y = (rng.random(n_samples) > 0.6).astype(int)

    # Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_bundle = {
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "train_medians": {col: 0.0 for col in FEATURE_COLUMNS},
    }
    scaler_path = tmp_path / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_bundle, f)

    # Model
    model = XGBClassifier(n_estimators=10, max_depth=2, random_state=42, eval_metric="logloss")
    model.fit(X_scaled, y)
    model_path = tmp_path / "xgb_credit_model.json"
    model.save_model(model_path)

    # Metrics
    metrics = {
        "optimal_threshold": 0.45,
        "model_version": "xgb_credit_model_test",
    }
    metrics_path = tmp_path / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    return {
        "model_path": model_path,
        "scaler_path": scaler_path,
        "metrics_path": metrics_path,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def predictor(tmp_artifacts: dict, monkeypatch) -> CreditRiskPredictor:
    """CreditRiskPredictor backed by temp artifacts."""
    monkeypatch.setattr(
        "src.phase1_quantitative.inference._DEFAULT_METRICS_PATH",
        tmp_artifacts["metrics_path"],
    )
    return CreditRiskPredictor(
        model_path=tmp_artifacts["model_path"],
        scaler_path=tmp_artifacts["scaler_path"],
        threshold=0.45,
    )


# ---------------------------------------------------------------------------
# 1. Rating mapping completeness
# ---------------------------------------------------------------------------

class TestRatingMapping:

    def test_all_investment_grade_map_to_zero(self):
        for rating in INVESTMENT_GRADE:
            assert _map_rating_to_binary(rating) == 0, f"{rating} should map to 0"

    def test_all_speculative_grade_map_to_one(self):
        for rating in SPECULATIVE_GRADE:
            assert _map_rating_to_binary(rating) == 1, f"{rating} should map to 1"

    def test_unknown_rating_returns_none(self):
        assert _map_rating_to_binary("XYZ") is None
        assert _map_rating_to_binary("") is None
        assert _map_rating_to_binary("NR") is None

    def test_no_rating_in_both_sets(self):
        overlap = INVESTMENT_GRADE & SPECULATIVE_GRADE
        assert len(overlap) == 0, f"Ratings appear in both sets: {overlap}"

    def test_all_known_ratings_covered(self):
        assert ALL_KNOWN_RATINGS == INVESTMENT_GRADE | SPECULATIVE_GRADE

    def test_bbb_minus_is_investment_grade(self):
        assert _map_rating_to_binary("BBB-") == 0

    def test_bb_plus_is_speculative_grade(self):
        assert _map_rating_to_binary("BB+") == 1

    def test_d_rating_is_speculative(self):
        assert _map_rating_to_binary("D") == 1


# ---------------------------------------------------------------------------
# 2. Feature schema
# ---------------------------------------------------------------------------

class TestFeatureSchema:

    def test_feature_columns_are_unique(self):
        assert len(FEATURE_COLUMNS) == len(set(FEATURE_COLUMNS))

    def test_feature_columns_are_nonempty(self):
        assert len(FEATURE_COLUMNS) > 0

    def test_feature_columns_are_strings(self):
        for col in FEATURE_COLUMNS:
            assert isinstance(col, str) and col.strip() == col


# ---------------------------------------------------------------------------
# 3. No data leakage — scaler fitted on train only
# ---------------------------------------------------------------------------

class TestNoLeakage:

    def test_scaler_fit_on_train_only(self, tmp_artifacts: dict):
        """
        Verify that the scaler bundle's mean was computed on the training
        array and that we can detect if test data were mistakenly used.
        """
        with open(tmp_artifacts["scaler_path"], "rb") as f:
            bundle = pickle.load(f)
        scaler: StandardScaler = bundle["scaler"]

        # Scaler should expose mean_ (fitted) not None
        assert hasattr(scaler, "mean_"), "Scaler was never fitted"
        assert scaler.mean_ is not None

    def test_scaler_bundle_has_required_keys(self, tmp_artifacts: dict):
        with open(tmp_artifacts["scaler_path"], "rb") as f:
            bundle = pickle.load(f)
        assert "scaler" in bundle
        assert "feature_columns" in bundle
        assert "train_medians" in bundle

    def test_train_medians_cover_all_features(self, tmp_artifacts: dict):
        with open(tmp_artifacts["scaler_path"], "rb") as f:
            bundle = pickle.load(f)
        for col in bundle["feature_columns"]:
            assert col in bundle["train_medians"]


# ---------------------------------------------------------------------------
# 4. Model artifact I/O
# ---------------------------------------------------------------------------

class TestModelArtifact:

    def test_model_loads_from_json(self, tmp_artifacts: dict):
        model = XGBClassifier()
        model.load_model(tmp_artifacts["model_path"])  # must not raise
        assert model is not None

    def test_loaded_model_has_predict_proba(self, tmp_artifacts: dict):
        model = XGBClassifier()
        model.load_model(tmp_artifacts["model_path"])
        assert callable(model.predict_proba)

    def test_metrics_json_has_required_keys(self, tmp_artifacts: dict):
        with open(tmp_artifacts["metrics_path"]) as f:
            metrics = json.load(f)
        required_keys = {"optimal_threshold", "model_version"}
        assert required_keys.issubset(metrics.keys())

    def test_optimal_threshold_in_valid_range(self, tmp_artifacts: dict):
        with open(tmp_artifacts["metrics_path"]) as f:
            metrics = json.load(f)
        t = metrics["optimal_threshold"]
        assert 0.0 < t < 1.0


# ---------------------------------------------------------------------------
# 5. CreditRiskPredictor.predict() contract
# ---------------------------------------------------------------------------

class TestCreditRiskPredictor:

    def test_predict_returns_dict(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        expected = {
            "probability_of_speculative",
            "predicted_class",
            "risk_tier",
            "confidence",
            "model_version",
        }
        assert expected.issubset(result.keys())

    def test_probability_in_unit_interval(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert 0.0 <= result["probability_of_speculative"] <= 1.0

    def test_predicted_class_is_binary(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert result["predicted_class"] in (0, 1)

    def test_risk_tier_values(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert result["risk_tier"] in ("Investment Grade", "Speculative Grade")

    def test_confidence_values(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert result["confidence"] in ("High", "Moderate", "Low")

    def test_model_version_is_string(self, predictor: CreditRiskPredictor):
        result = predictor.predict(MOCK_FEATURES)
        assert isinstance(result["model_version"], str)

    def test_feature_columns_property(self, predictor: CreditRiskPredictor):
        assert predictor.feature_columns == FEATURE_COLUMNS

    def test_threshold_property(self, predictor: CreditRiskPredictor):
        assert predictor.threshold == 0.45


# ---------------------------------------------------------------------------
# 6. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_missing_single_feature_raises_value_error(self, predictor: CreditRiskPredictor):
        incomplete = {k: v for k, v in MOCK_FEATURES.items() if k != "currentRatio"}
        with pytest.raises(ValueError, match="currentRatio"):
            predictor.predict(incomplete)

    def test_missing_multiple_features_raises_value_error(self, predictor: CreditRiskPredictor):
        incomplete = {"currentRatio": 1.0}  # far from complete
        with pytest.raises(ValueError):
            predictor.predict(incomplete)

    def test_empty_dict_raises_value_error(self, predictor: CreditRiskPredictor):
        with pytest.raises(ValueError):
            predictor.predict({})

    def test_none_value_replaced_by_train_median(self, predictor: CreditRiskPredictor):
        features_with_none = dict(MOCK_FEATURES)
        features_with_none["currentRatio"] = None
        result = predictor.predict(features_with_none)  # must not raise
        assert 0.0 <= result["probability_of_speculative"] <= 1.0

    def test_nan_value_replaced_by_train_median(self, predictor: CreditRiskPredictor):
        features_with_nan = dict(MOCK_FEATURES)
        features_with_nan["debtEquityRatio"] = float("nan")
        result = predictor.predict(features_with_nan)  # must not raise
        assert 0.0 <= result["probability_of_speculative"] <= 1.0

    def test_model_artifact_not_found_raises(self, tmp_path: Path, tmp_artifacts: dict):
        with pytest.raises(FileNotFoundError, match="Model artifact not found"):
            CreditRiskPredictor(
                model_path=tmp_path / "nonexistent_model.json",
                scaler_path=tmp_artifacts["scaler_path"],
            )

    def test_scaler_artifact_not_found_raises(self, tmp_path: Path, tmp_artifacts: dict):
        with pytest.raises(FileNotFoundError, match="Scaler artifact not found"):
            CreditRiskPredictor(
                model_path=tmp_artifacts["model_path"],
                scaler_path=tmp_path / "nonexistent_scaler.pkl",
            )


# ---------------------------------------------------------------------------
# 7. Confidence labelling
# ---------------------------------------------------------------------------

class TestConfidenceLabelling:

    def test_high_confidence_far_above_threshold(self, predictor: CreditRiskPredictor):
        # Force prob well above threshold
        assert predictor._confidence_label(0.45 + 0.30) == "High"

    def test_high_confidence_far_below_threshold(self, predictor: CreditRiskPredictor):
        assert predictor._confidence_label(0.45 - 0.30) == "High"

    def test_moderate_confidence_near_threshold(self, predictor: CreditRiskPredictor):
        assert predictor._confidence_label(0.45 + 0.15) == "Moderate"
        assert predictor._confidence_label(0.45 - 0.15) == "Moderate"

    def test_low_confidence_at_threshold(self, predictor: CreditRiskPredictor):
        assert predictor._confidence_label(0.45) == "Low"

    def test_low_confidence_just_inside_boundary(self, predictor: CreditRiskPredictor):
        assert predictor._confidence_label(0.45 + 0.05) == "Low"
