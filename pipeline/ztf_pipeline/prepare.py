"""Clean the raw export, split it by time, and write the training set and the stream CSV."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .cleaning import clean, load_raw
from .config import Settings
from .features import ALERT_FIELDS, LEAKY_COLUMNS, to_alerts

log = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(settings: Settings) -> dict:
    cleaned = clean(load_raw(settings.raw_csv))
    labelled = cleaned[cleaned["is_ia"].notna()]
    start, end = pd.Timestamp(settings.stream_start), pd.Timestamp(settings.stream_end)

    train = labelled[labelled["discdate"] < start]
    stream = labelled[(labelled["discdate"] >= start) & (labelled["discdate"] < end)].sort_values("discdate")

    overlap = set(train["ZTFID"]) & set(stream["ZTFID"])
    if overlap:
        raise RuntimeError(f"{len(overlap)} objects are in both the training set and the stream")

    leaked = set(ALERT_FIELDS) & set(LEAKY_COLUMNS)
    if leaked:
        raise RuntimeError(f"leaky columns are on the alert allow-list: {sorted(leaked)}")
    stream_alerts = to_alerts(stream)

    settings.data_dir.mkdir(parents=True, exist_ok=True)
    training = to_alerts(train).assign(is_ia=train["is_ia"].astype(bool).to_numpy())
    training.to_parquet(settings.train_path, index=False)
    stream_alerts.to_csv(settings.stream_path, index=False)    # alert fields only: no labels

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_csv": str(settings.raw_csv),
        "raw_sha256": file_sha256(settings.raw_csv),
        "objects_after_cleaning": len(cleaned),
        "labelled_objects": len(labelled),
        "train": {"objects": len(train), "discovered_before": settings.stream_start,
                  "ia_fraction": round(float(train["is_ia"].astype(float).mean()), 4),
                  "sha256": file_sha256(settings.train_path)},
        "stream": {"objects": len(stream), "discovered_from": settings.stream_start,
                   "discovered_before": settings.stream_end, "columns": list(stream_alerts.columns),
                   "sha256": file_sha256(settings.stream_path)},
        "excluded": {"unlabelled": int(cleaned["is_ia"].isna().sum()),
                     "labelled_after_stream_end": int((labelled["discdate"] >= end).sum())},
    }
    settings.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("training set: %d objects (%.1f%% Ia) -> %s", len(train), 100 * manifest["train"]["ia_fraction"],
             settings.train_path)
    log.info("stream: %d alerts from %s to %s -> %s", len(stream), settings.stream_start, settings.stream_end,
             settings.stream_path)
    return manifest
