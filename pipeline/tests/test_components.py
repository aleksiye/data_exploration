from __future__ import annotations

import math

import pandas as pd
import pytest

from tests.conftest import make_alert
from ztf_pipeline.messages import Alert, InvalidMessage
from ztf_pipeline.replay import assign_nights
from ztf_pipeline.store import CandidateStore
from ztf_pipeline.transport import encode


# --- messages ------------------------------------------------------------------------------------

def test_alert_round_trip():
    alert = Alert.from_dict(make_alert("ZTF1", "2024-01-01"))
    assert Alert.from_dict(alert.to_dict()) == alert


def test_alert_missing_number_becomes_null():
    alert = Alert.from_dict(make_alert("ZTF1", "2024-01-01", a_v=float("nan")))
    assert alert.a_v is None
    encode(alert.to_dict())     # would raise on NaN


@pytest.mark.parametrize("bad", [
    {"schema": "something.else"},
    {"peak_mag": "bright"},
    {"ztf_id": None},
    {"peak_mag": None},
])
def test_invalid_alerts_are_rejected(bad):
    with pytest.raises(InvalidMessage):
        Alert.from_dict({**make_alert("ZTF1", "2024-01-01"), **bad})


def test_missing_field_is_rejected():
    data = make_alert("ZTF1", "2024-01-01")
    del data["fade_days"]
    with pytest.raises(InvalidMessage):
        Alert.from_dict(data)


def test_encode_refuses_nan():
    with pytest.raises(ValueError):
        encode({"x": math.nan})


# --- store ---------------------------------------------------------------------------------------

def test_store_ranks_filters_and_limits():
    store = CandidateStore()
    store.upsert([("a", "n1", 0.90, "t3", {}), ("b", "n1", 0.99, "t1", {"ra_deg": 1.5}), ("c", "n1", 0.95, "t2", {}),
                  ("d", "n1", 0.95, "t1", {}), ("e", "n1", 0.50, "t1", {}), ("f", "n2", 0.999, "t1", {})])
    assert [(z, p) for z, p, _ in store.top("n1", 0.9, 10)] == [("b", 0.99), ("d", 0.95), ("c", 0.95), ("a", 0.90)]
    assert [z for z, _, _ in store.top("n1", 0.9, 2)] == ["b", "d"]                      # tie -> earlier alert
    assert store.top("n1", 0.99, 1)[0][2] == {"ra_deg": 1.5}                             # details survive
    assert store.count("n1") == 5 and store.count("n1", 0.9) == 4


def test_store_upsert_replaces_and_evict_removes_a_night():
    store = CandidateStore()
    store.upsert([("a", "n1", 0.9, "t1", {})])
    store.upsert([("a", "n1", 0.2, "t2", {})])
    assert store.count("n1") == 1 and store.top("n1", 0.0, 5) == [("a", 0.2, {})]
    store.upsert([("b", "n2", 0.5, "t1", {})])
    assert store.evict("n1") == 1
    assert store.count("n1") == 0 and store.count("n2") == 1


# --- replay --------------------------------------------------------------------------------------

def test_assign_nights_packs_alerts_in_order():
    stream = pd.DataFrame({"ztf_id": list("edcba"),
                           "discovered_at": ["2024-01-05", "2024-01-04", "2024-01-03", "2024-01-02", "2024-01-01"]})
    out = assign_nights(stream, alerts_per_night=2)
    assert out["ztf_id"].tolist() == list("abcde")                     # real discovery order kept
    assert out["night"].tolist() == ["2024-01-01"] * 2 + ["2024-01-02"] * 2 + ["2024-01-03"]
    assert out["event_time"].is_monotonic_increasing


def test_assign_nights_can_use_real_dates():
    stream = pd.DataFrame({"ztf_id": ["a", "b"], "discovered_at": ["2024-03-01 05:00", "2024-03-07 23:00"]})
    assert assign_nights(stream, alerts_per_night=None)["night"].tolist() == ["2024-03-01", "2024-03-07"]
