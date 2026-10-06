"""What an alert contains, what the model may see, and how model inputs are built.

`build_features` is used both for training and inside the streaming service, so the model is never
trained on anything it won't receive in production (no training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Allow-list: cleaned-catalogue column -> alert field. Only these columns ever reach the stream.
ALERT_FIELDS = {
    "ZTFID": "ztf_id",
    "discdate": "discovered_at",
    "ra_deg": "ra_deg",
    "dec_deg": "dec_deg",
    "b": "gal_b",
    "A_V": "a_v",
    "peakfilt": "peak_filter",
    "peakmag": "peak_mag",
    "rise": "rise_days",
    "rise_is_limit": "rise_is_limit",
    "fade": "fade_days",
    "fade_is_limit": "fade_is_limit",
    "duration": "duration_days",
    "duration_is_limit": "duration_is_limit",
}

# Columns that would leak the answer, and why. Kept here so the reasons live next to the allow-list;
# tests check that none of them can reach the stream or the model.
LEAKY_COLUMNS = {
    "type": "the answer itself, from the classification spectrum",
    "subclass": "derived from type",
    "is_ia": "derived from type",
    "IAUID": "the SN/AT prefix is assigned after spectroscopic confirmation",
    "redshift": "measured from the classification spectrum",
    "peakabs": "computed from the redshift",
    "hostabs": "computed using the redshift",
    "hostcol": "computed using the redshift",
}

MODEL_FEATURES = [
    "peak_mag", "log_rise", "log_fade", "log_duration",
    "rise_is_limit", "fade_is_limit", "duration_is_limit",
    "peak_in_g", "abs_gal_b", "a_v",
]


def to_alerts(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Keep only allow-listed columns from a cleaned catalogue, renamed to alert field names."""
    return cleaned[list(ALERT_FIELDS)].rename(columns=ALERT_FIELDS)


def build_features(alerts: pd.DataFrame) -> pd.DataFrame:
    """Model inputs from alert fields. Missing a_v is left as NaN; the model handles it."""
    out = pd.DataFrame(index=alerts.index)
    out["peak_mag"] = alerts["peak_mag"].astype(float)
    for name in ("rise", "fade", "duration"):
        out[f"log_{name}"] = np.log10(alerts[f"{name}_days"].astype(float) + 1)
        out[f"{name}_is_limit"] = alerts[f"{name}_is_limit"].astype(bool).astype(int)
    out["peak_in_g"] = (alerts["peak_filter"] == "g").astype(int)
    out["abs_gal_b"] = alerts["gal_b"].astype(float).abs()
    out["a_v"] = alerts["a_v"].astype(float)
    return out[MODEL_FEATURES]
