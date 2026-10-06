"""Replay the held-out stream CSV into the alerts topic, as if the discoveries were happening now."""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .messages import Alert
from .transport import Publisher

log = logging.getLogger(__name__)


def assign_nights(stream: pd.DataFrame, alerts_per_night: int | None) -> pd.DataFrame:
    """Give every alert an observing night and an event time, keeping the real discovery order.

    alerts_per_night=None uses the real discovery dates (a few alerts a night in this catalogue).
    Otherwise alerts are packed into consecutive simulated nights of that many alerts, starting on the
    first discovery date. That models a busier survey: 333 a night is about 10,000 a month.
    """
    df = stream.sort_values("discovered_at", kind="stable").reset_index(drop=True)
    discovered = pd.to_datetime(df["discovered_at"], format="mixed")
    if alerts_per_night is None:
        nights, event_times = discovered.dt.normalize(), discovered
    else:
        position = np.arange(len(df))
        nights = discovered.iloc[0].normalize() + pd.to_timedelta(position // alerts_per_night, unit="D")
        event_times = nights + pd.to_timedelta((position % alerts_per_night) * 86400 / alerts_per_night, unit="s")
        nights, event_times = pd.Series(nights, index=df.index), pd.Series(event_times, index=df.index)
    return df.assign(night=nights.dt.strftime("%Y-%m-%d"), event_time=event_times.dt.strftime("%Y-%m-%dT%H:%M:%SZ"))


def replay(stream_path: Path, publisher: Publisher, topic: str, alerts_per_night: int | None = 333,
           seconds_per_night: float = 20.0, limit: int | None = None,
           stop: threading.Event | None = None) -> int:
    """Publish alerts night by night, spread evenly over seconds_per_night of wall time (0 = no waiting)."""
    stop = stop or threading.Event()
    alerts = assign_nights(pd.read_csv(stream_path), alerts_per_night)
    if limit is not None:
        alerts = alerts.head(limit)
    sent = 0
    for night, tonight in alerts.groupby("night", sort=True):
        log.info("night %s: replaying %d alerts over %.0f s", night, len(tonight), seconds_per_night)
        interval = seconds_per_night / len(tonight)
        started = time.monotonic()
        for i, row in enumerate(tonight.to_dict("records")):
            if stop.is_set():
                log.info("replay stopped after %d alerts", sent)
                publisher.flush()
                return sent
            alert = Alert.from_dict({"schema": Alert.SCHEMA, **row})     # validates before publishing
            publisher.publish(topic, key=alert.ztf_id, value=alert.to_dict())
            sent += 1
            wait = started + (i + 1) * interval - time.monotonic()
            if wait > 0:
                stop.wait(wait)
    publisher.flush()
    log.info("replay finished: %d alerts over %d nights", sent, alerts["night"].nunique())
    return sent
