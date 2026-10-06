from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ztf_pipeline.cleaning import clean, load_raw
from ztf_pipeline.features import ALERT_FIELDS, LEAKY_COLUMNS, MODEL_FEATURES, build_features
from ztf_pipeline.train import choose_threshold

NOTEBOOK_CLEAN = Path(__file__).resolve().parents[2] / "data" / "bts_clean.parquet"


def test_no_leaky_column_is_allow_listed():
    assert not set(ALERT_FIELDS) & set(LEAKY_COLUMNS)
    assert not set(ALERT_FIELDS.values()) & set(LEAKY_COLUMNS)


def test_stream_contains_only_alert_fields(prepared):
    stream = pd.read_csv(prepared.stream_path)
    assert list(stream.columns) == list(ALERT_FIELDS.values())


def test_stream_and_training_data_do_not_overlap(prepared):
    stream = pd.read_csv(prepared.stream_path)
    train = pd.read_parquet(prepared.train_path)
    assert not set(stream["ztf_id"]) & set(train["ztf_id"])
    assert pd.to_datetime(train["discovered_at"]).max() < pd.Timestamp(prepared.stream_start)
    assert pd.to_datetime(stream["discovered_at"], format="mixed").min() >= pd.Timestamp(prepared.stream_start)


def test_build_features_columns_and_missing_values():
    alerts = pd.DataFrame({
        "peak_mag": [18.0, 17.0], "rise_days": [9.0, 0.0], "fade_days": [14.0, 3.0], "duration_days": [23.0, 3.0],
        "rise_is_limit": [True, False], "fade_is_limit": [False, False], "duration_is_limit": [True, False],
        "peak_filter": ["g", "r"], "gal_b": [-30.0, 12.0], "a_v": [None, 0.2],
    })
    features = build_features(alerts)
    assert list(features.columns) == MODEL_FEATURES
    assert np.isnan(features.loc[0, "a_v"])
    assert features["peak_in_g"].tolist() == [1, 0]
    assert features["abs_gal_b"].tolist() == [30.0, 12.0]
    assert features.loc[1, "log_rise"] == 0.0


@pytest.mark.skipif(not NOTEBOOK_CLEAN.exists(), reason="notebook 01 output not available")
def test_cleaning_matches_notebook_01(settings):
    ours = clean(load_raw(settings.raw_csv)).set_index("ZTFID").sort_index()
    notebook = pd.read_parquet(NOTEBOOK_CLEAN).set_index("ZTFID").sort_index()
    assert ours.index.equals(notebook.index)
    pd.testing.assert_series_equal(ours["is_ia"], notebook["is_ia"], check_names=False)
    for col in ["rise", "fade", "duration", "b", "A_V", "peakmag"]:
        np.testing.assert_allclose(ours[col], notebook[col], equal_nan=True)


def test_threshold_ignores_a_lucky_bump():
    # Ranked best first: 10 Ia, 2 non-Ia, 30 Ia, 20 non-Ia. Precision is 100% for the top 10, dips to
    # 83% at 12, then climbs back to 95% between 40 and 42 by luck. Taking the lowest threshold that
    # touches 95% would accept the dip; the robust rule stops at the top 10.
    y = np.array([1] * 10 + [0] * 2 + [1] * 30 + [0] * 20)
    scores = np.linspace(1.0, 0.0, len(y))
    threshold, precision, _ = choose_threshold(y, scores, 0.95)
    above = scores >= threshold
    assert above.sum() == 10 and precision == 1.0
    for t in scores[above]:                       # every higher threshold also meets the target
        assert y[scores >= t].mean() >= 0.95
