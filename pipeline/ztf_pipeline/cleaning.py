"""Cleaning the raw BTS export: the steps worked out in notebook 01, as reusable functions."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

log = logging.getLogger(__name__)

LIMIT_COLUMNS = ("rise", "fade", "duration")
B_TOLERANCE_DEG = 0.01      # catalogue b must agree with b recomputed from RA/Dec to this precision


def load_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, na_values="-")


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row per object with parsed values and spectroscopic labels (is_ia is NA if unclassified)."""
    df = raw.copy()
    df = _parse_lower_limits(df)
    df["discdate"] = pd.to_datetime(df["discdate"], format="mixed")
    df = _fix_positions(df)
    df = _drop_duplicates(df)
    df = _add_labels(df)
    return df.reset_index(drop=True)


def _parse_lower_limits(df: pd.DataFrame) -> pd.DataFrame:
    # Timescales like ">5.6" are lower limits: keep the number and a flag.
    for col in LIMIT_COLUMNS:
        text = df[col].astype("string")
        df[f"{col}_is_limit"] = text.str.startswith(">").fillna(False).astype(bool)
        df[col] = pd.to_numeric(text.str.lstrip(">"))
    return df


def _fix_positions(df: pd.DataFrame) -> pd.DataFrame:
    # Some rows carry another object's galactic latitude and extinction. Recompute b from RA/Dec and
    # blank A_V wherever the catalogue's b disagrees, since that A_V belongs to the wrong position.
    coords = SkyCoord(df["RA"], df["Dec"], unit=(u.hourangle, u.deg))
    df["ra_deg"] = coords.ra.deg
    df["dec_deg"] = coords.dec.deg
    b_true = coords.galactic.b.deg
    corrupted = np.abs(df["b"] - b_true) > B_TOLERANCE_DEG
    df["b"] = b_true
    df.loc[corrupted, "A_V"] = np.nan
    log.info("recomputed galactic latitude; %d rows had a corrupted b/A_V", int(corrupted.sum()))
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # The same object can appear under two ZTF IDs with one IAU name. Keep the row that has a type.
    df = df.sort_values(["IAUID", "type"], na_position="last")
    repeat = df["IAUID"].notna() & df["IAUID"].duplicated(keep="first")
    log.info("dropped %d duplicate objects", int(repeat.sum()))
    return df[~repeat].sort_index()


def to_subclass(label: object) -> str:
    if pd.isna(label):
        return "Unclassified"
    t = str(label).rstrip("?")
    if t == "SN Iax" or "Ca-rich" in t:      # before the generic "SN Ia" test
        return "SN peculiar"
    if t.startswith("SN Ia"):
        return "SN Ia"
    if t.startswith("SN II"):
        return "SN II"
    if t.startswith("SN I"):
        return "SN Ib/c"
    if t.startswith("SLSN"):
        return "SLSN"
    if t.startswith("TDE"):
        return "TDE"
    if t in ("CV", "nova"):
        return "CV / nova"
    if t == "AGN":
        return "AGN"
    return "Other"


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["subclass"] = df["type"].map(to_subclass)
    classified = df["subclass"] != "Unclassified"
    df["is_ia"] = pd.Series(pd.NA, index=df.index, dtype="boolean")
    df.loc[classified, "is_ia"] = df.loc[classified, "subclass"] == "SN Ia"
    return df
