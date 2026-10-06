"""Cross-reference what the service decided with the original catalogue, and check the targets.

The evaluator is the only component that reads labels for the streamed objects. It takes them from the
raw bts.csv (through the same cleaning), not from anything the pipeline produced.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .cleaning import clean, load_raw
from .config import Settings
from .model import ModelBundle

log = logging.getLogger(__name__)


@dataclass
class Check:
    name: str
    passed: bool
    detail: str

    def __post_init__(self):
        self.passed = bool(self.passed)       # numpy booleans aren't JSON serialisable


def markdown_table(df: pd.DataFrame) -> str:
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    rows = [[str(i)] + [str(v) for v in row] for i, row in zip(df.index, df.itertuples(index=False))]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: has the classifier service run?")
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; well behaved near 0 and 1, unlike the normal approximation."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def evaluate(settings: Settings) -> bool:
    truth = clean(load_raw(settings.raw_csv)).set_index("ZTFID")[["type", "subclass", "is_ia"]]
    stream = pd.read_csv(settings.stream_path)
    train_ids = set(pd.read_parquet(settings.train_path, columns=["ztf_id"])["ztf_id"])
    model = ModelBundle.load(settings)

    # Delivery is at-least-once: keep the latest message per object and per night.
    classified = read_jsonl(settings.classifications_log).drop_duplicates("ztf_id", keep="last")
    nights = read_jsonl(settings.selections_log).drop_duplicates("night", keep="last").sort_values("night")
    selected = pd.DataFrame([{"night": s["night"], **c} for s in nights.to_dict("records") for c in s["candidates"]],
                            columns=["night", "rank", "ztf_id", "p_ia"])

    classified = classified.join(truth, on="ztf_id")
    selected = selected.join(truth, on="ztf_id")
    checks: list[Check] = []

    missing = set(stream["ztf_id"]) - set(classified["ztf_id"])
    unexpected = set(classified["ztf_id"]) - set(stream["ztf_id"])
    checks.append(Check("every streamed alert was classified", not missing and not unexpected,
                        f"{len(classified):,} classified of {len(stream):,} streamed; "
                        f"{len(missing)} missing, {len(unexpected)} unexpected"))

    unknown = classified["is_ia"].isna().sum()
    checks.append(Check("every classified object is in the original catalogue with a type", unknown == 0,
                        f"{unknown} objects without a label in bts.csv"))

    in_training = set(classified["ztf_id"]) & train_ids
    checks.append(Check("no streamed object was in the training data", not in_training,
                        f"{len(in_training)} overlapping objects"))

    offline = pd.Series(model.score(stream), index=stream["ztf_id"])
    max_diff = float((classified.set_index("ztf_id")["p_ia"] - offline).abs().max())
    checks.append(Check("streamed scores match offline scoring (no training/serving skew)", max_diff < 1e-5,
                        f"largest difference in P(Ia): {max_diff:.2g}"))

    versions = classified["model_version"].unique()
    checks.append(Check("one model version served the whole stream", len(versions) == 1, ", ".join(versions)))

    per_night_sent = selected.groupby("night").size().reindex(nights["night"], fill_value=0)
    checks.append(Check(f"at most {settings.daily_capacity} candidates per night",
                        int(per_night_sent.max()) <= settings.daily_capacity,
                        f"largest night: {int(per_night_sent.max())}"))

    tp = int(selected["is_ia"].astype(bool).sum())
    n_sent = len(selected)
    precision = tp / n_sent if n_sent else float("nan")
    low, high = wilson_interval(tp, n_sent)
    checks.append(Check(f"precision of selected candidates >= {settings.target_precision:.0%}",
                        precision >= settings.target_precision,
                        f"{precision:.1%} ({tp} of {n_sent} were SN Ia; 95% interval {low:.1%}-{high:.1%})"))

    per_night = _per_night(classified, selected, model.threshold)
    false_positives = (selected[~selected["is_ia"].astype(bool)]
                       .groupby("subclass").size().sort_values(ascending=False).rename("false positives"))
    auc = roc_auc_score(classified["is_ia"].astype(int), classified["p_ia"])

    passed = all(c.passed for c in checks)
    _write_report(settings, model, checks, per_night, false_positives, precision, (low, high), auc, passed)
    for c in checks:
        log.info("[%s] %s: %s", "PASS" if c.passed else "FAIL", c.name, c.detail)
    log.info("overall: %s. Report: %s", "PASSED" if passed else "FAILED", settings.report_path)
    return passed


def _per_night(classified: pd.DataFrame, selected: pd.DataFrame, threshold: float) -> pd.DataFrame:
    alerts = classified.assign(ia=classified["is_ia"].astype(bool), eligible=classified["p_ia"] >= threshold)
    table = alerts.groupby("night").agg(alerts=("ztf_id", "size"), ia_alerts=("ia", "sum"),
                                        eligible=("eligible", "sum"))
    sent = selected.assign(ia=selected["is_ia"].astype(bool)).groupby("night").agg(
        selected=("ztf_id", "size"), true_ia=("ia", "sum"))
    table = table.join(sent).fillna(0).astype(int)
    table["false_positives"] = table["selected"] - table["true_ia"]
    table["precision"] = (table["true_ia"] / table["selected"].where(table["selected"] > 0)).round(3)
    table["share_of_ia_found"] = (table["true_ia"] / table["ia_alerts"].where(table["ia_alerts"] > 0)).round(3)
    return table


def _write_report(settings, model, checks, per_night, false_positives, precision, interval, auc, passed) -> None:
    lines = [
        "# Stream evaluation report", "",
        f"**Result: {'PASSED' if passed else 'FAILED'}**", "",
        f"* Model: `{model.version}`, selection threshold P(Ia) ≥ {model.threshold:.3f}, "
        f"up to {settings.daily_capacity} candidates per night",
        f"* Target precision: {settings.target_precision:.0%}. Achieved: **{precision:.1%}** "
        f"(95% interval {interval[0]:.1%}–{interval[1]:.1%})",
        f"* ROC AUC over all {per_night['alerts'].sum():,} streamed alerts: {auc:.3f}",
        "* Labels come from the original `bts.csv`; the stream itself never contained them.", "",
        "## Checks", "", "| | Check | Detail |", "|---|---|---|",
        *[f"| {'✅' if c.passed else '❌'} | {c.name} | {c.detail} |" for c in checks], "",
        "## Per night", "", markdown_table(per_night), "",
        "## What the false positives really were", "",
        markdown_table(false_positives.to_frame()) if len(false_positives) else "None.", "",
        "*Precision here is the share of selected candidates that are SN Ia. Its complement (the share that "
        "aren't) is the false discovery rate, not the false-positive rate, which is the share of all non-Ia "
        "that get selected.*",
    ]
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"passed": passed, "precision": precision, "precision_interval": interval, "roc_auc": auc,
               "model_version": model.version, "threshold": model.threshold,
               "checks": [c.__dict__ for c in checks], "per_night": per_night.reset_index().to_dict("records")}
    settings.report_path.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
