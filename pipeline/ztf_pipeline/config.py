"""Runtime settings.

Every setting can be overridden with an environment variable named ZTF_<SETTING>, for example
ZTF_BOOTSTRAP_SERVERS=broker:9092 or ZTF_DAILY_CAPACITY=150.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = PIPELINE_DIR.parent


@dataclass(frozen=True)
class Settings:
    # --- files -------------------------------------------------------------------------------
    raw_csv: Path = PROJECT_DIR / "bts.csv"
    data_dir: Path = PIPELINE_DIR / "data"
    artifacts_dir: Path = PIPELINE_DIR / "artifacts"
    output_dir: Path = PIPELINE_DIR / "output"

    # --- train / stream split ----------------------------------------------------------------
    # Labelled objects discovered in [stream_start, stream_end) are held out and replayed as the
    # live stream; everything labelled before stream_start is training data. 2026 objects are
    # excluded because their types are missing from this export.
    stream_start: str = "2024-01-01"
    stream_end: str = "2026-01-01"

    # --- selection policy ----------------------------------------------------------------------
    target_precision: float = 0.95
    daily_capacity: int = 100

    # --- Kafka -----------------------------------------------------------------------------------
    bootstrap_servers: str = "localhost:9092"
    alerts_topic: str = "ztf.alerts"
    classifications_topic: str = "ztf.classifications"
    shortlist_topic: str = "ztf.shortlist"
    selections_topic: str = "ztf.selections"
    dead_letter_topic: str = "ztf.alerts.dlq"
    consumer_group: str = "ia-classifier"

    # --- classifier service ------------------------------------------------------------------
    rerank_interval_s: float = 2.0     # how often tonight's shortlist is re-ranked and published
    idle_close_s: float = 10.0         # close the night after this long without new alerts

    # --- derived paths -------------------------------------------------------------------------
    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def stream_path(self) -> Path:
        return self.data_dir / "stream.csv"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.joblib"

    @property
    def model_card_path(self) -> Path:
        return self.artifacts_dir / "model_card.json"

    @property
    def classifications_log(self) -> Path:
        return self.output_dir / "classifications.jsonl"

    @property
    def selections_log(self) -> Path:
        return self.output_dir / "selections.jsonl"

    @property
    def live_shortlist_path(self) -> Path:
        return self.output_dir / "shortlist_live.json"

    @property
    def worklist_path(self) -> Path:
        return self.output_dir / "worklist.db"

    @property
    def report_path(self) -> Path:
        return self.output_dir / "evaluation_report.md"

    @property
    def topics(self) -> list[str]:
        return [self.alerts_topic, self.classifications_topic, self.shortlist_topic,
                self.selections_topic, self.dead_letter_topic]

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> Settings:
        environ = os.environ if environ is None else environ
        overrides = {}
        for f in fields(cls):
            value = environ.get(f"ZTF_{f.name.upper()}")
            if value is not None:
                overrides[f.name] = type(f.default)(value)    # Path, str, int or float
        return replace(cls(), **overrides)
