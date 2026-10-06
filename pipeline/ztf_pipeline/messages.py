"""Message schemas for every topic. Messages are JSON objects tagged with a schema name."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone


class InvalidMessage(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clean_float(value: object) -> float | None:
    # JSON has no NaN: missing numbers travel as null.
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


@dataclass(frozen=True)
class Alert:
    """One new transient, as the survey would announce it (topic: alerts)."""
    SCHEMA = "ztf.alert.v1"
    NULLABLE = frozenset({"a_v"})

    ztf_id: str
    night: str              # observing night the alert belongs to (YYYY-MM-DD)
    event_time: str         # when the alert was issued (simulated clock)
    discovered_at: str      # original discovery time in the catalogue
    ra_deg: float
    dec_deg: float
    gal_b: float
    a_v: float | None
    peak_filter: str
    peak_mag: float
    rise_days: float
    rise_is_limit: bool
    fade_days: float
    fade_is_limit: bool
    duration_days: float
    duration_is_limit: bool

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, **asdict(self)}

    @classmethod
    def from_dict(cls, data: dict) -> Alert:
        if data.get("schema") != cls.SCHEMA:
            raise InvalidMessage(f"expected schema {cls.SCHEMA}, got {data.get('schema')!r}")
        missing = [f.name for f in fields(cls) if f.name not in data]
        if missing:
            raise InvalidMessage(f"alert is missing fields: {missing}")
        null = [f.name for f in fields(cls) if data[f.name] is None and f.name not in cls.NULLABLE]
        if null:
            raise InvalidMessage(f"alert has null required fields: {null}")
        try:
            return cls(
                ztf_id=str(data["ztf_id"]), night=str(data["night"]), event_time=str(data["event_time"]),
                discovered_at=str(data["discovered_at"]),
                ra_deg=float(data["ra_deg"]), dec_deg=float(data["dec_deg"]), gal_b=float(data["gal_b"]),
                a_v=_clean_float(data["a_v"]), peak_filter=str(data["peak_filter"]), peak_mag=float(data["peak_mag"]),
                rise_days=float(data["rise_days"]), rise_is_limit=bool(data["rise_is_limit"]),
                fade_days=float(data["fade_days"]), fade_is_limit=bool(data["fade_is_limit"]),
                duration_days=float(data["duration_days"]), duration_is_limit=bool(data["duration_is_limit"]),
            )
        except (TypeError, ValueError) as exc:
            raise InvalidMessage(f"alert has a badly typed field: {exc}") from exc


# Alert fields copied onto every shortlisted candidate: what the team needs to plan an observation.
CANDIDATE_FIELDS = ("ra_deg", "dec_deg", "peak_mag", "peak_filter", "rise_days", "rise_is_limit",
                    "fade_days", "fade_is_limit", "discovered_at")


@dataclass(frozen=True)
class Classification:
    """The model's verdict on one alert (topic: classifications)."""
    SCHEMA = "ztf.classification.v1"

    ztf_id: str
    night: str
    p_ia: float
    eligible: bool          # P(Ia) clears the precision threshold
    model_version: str
    scored_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, **asdict(self)}


@dataclass(frozen=True)
class Shortlist:
    """Tonight's best candidates (topic: shortlist while the night is open, selections once it closes)."""
    SCHEMA = "ztf.shortlist.v1"

    night: str
    final: bool
    model_version: str
    threshold: float
    capacity: int
    alerts_tonight: int
    eligible_tonight: int
    candidates: list[dict]  # [{"rank": 1, "ztf_id": ..., "p_ia": ..., <CANDIDATE_FIELDS>}, ...]
    generated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {"schema": self.SCHEMA, **asdict(self)}
