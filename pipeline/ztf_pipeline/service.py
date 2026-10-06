"""The classifier service: score alerts as they arrive and keep tonight's shortlist up to date.

Delivery guarantees
-------------------
Input offsets are committed only when a night closes, after that night's outputs have been flushed.
If the service dies mid-night, a restart replays the night's alerts from Kafka and rebuilds the
in-memory store, so no candidate is lost. The price is that classifications (and, rarely, a night's
selection) can be published twice. Consumers keep the latest message per ztf_id or per night.

Night boundaries
----------------
Alerts arrive in order on a single partition. A night closes when the first alert of a later night
arrives, or when no alert has arrived for `idle_close_s` (the end of the stream, or a quiet night).
Alerts for a night that is already closed are still classified, but can no longer be selected.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from collections import Counter
from dataclasses import asdict
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .audit import AuditLog
from .config import Settings
from .messages import CANDIDATE_FIELDS, Alert, Classification, InvalidMessage, Shortlist, utc_now
from .store import CandidateStore
from .transport import Publisher, Record, Subscriber

log = logging.getLogger(__name__)


class Scorer(Protocol):
    version: str
    threshold: float

    def score(self, alerts: pd.DataFrame) -> np.ndarray: ...


class ClassifierService:
    def __init__(self, settings: Settings, model: Scorer, subscriber: Subscriber, publisher: Publisher,
                 audit: AuditLog, clock: Callable[[], float] = time.monotonic):
        self.settings, self.model = settings, model
        self.subscriber, self.publisher, self.audit = subscriber, publisher, audit
        self.clock = clock
        self.store = CandidateStore()
        self.night: str | None = None              # the open night
        self.last_closed: str | None = None
        self.stats: Counter = Counter()
        self._uncommitted: list[Record] = []       # records consumed since the last commit
        self._shortlist_ids: list[str] = []
        self._last_rerank = self._last_alert = clock()
        self.closed_on_idle = False

    # --- main loop -------------------------------------------------------------------------------

    def run(self, stop: threading.Event, exit_when_idle: bool = False) -> None:
        log.info("classifier service running: model %s, P(Ia) >= %.3f, up to %d per night",
                 self.model.version, self.model.threshold, self.settings.daily_capacity)
        try:
            while not stop.is_set():
                self.step()
                if exit_when_idle and self.closed_on_idle:
                    log.info("stream idle and last night closed; exiting")
                    break
        finally:
            self.publisher.flush()
            if self.night is not None:
                log.info("night %s left open: its %d alerts will be replayed on restart",
                         self.night, self.store.count(self.night))
            log.info("service stopped: %s", dict(self.stats))

    def step(self) -> None:
        records = self.subscriber.poll(max_records=500, timeout=0.5)
        if records:
            self.handle(records)
        now = self.clock()
        if self.night is not None and now - self._last_rerank >= self.settings.rerank_interval_s:
            self.rerank()
        if self.night is not None and now - self._last_alert >= self.settings.idle_close_s:
            self.close_night(reason="no alerts for %.0f s" % self.settings.idle_close_s)
            self.closed_on_idle = True

    # --- processing --------------------------------------------------------------------------------

    def handle(self, records: list[Record]) -> None:
        valid: list[tuple[Record, Alert]] = []
        for record in records:
            try:
                if record.value is None:
                    raise InvalidMessage("payload is not a JSON object")
                valid.append((record, Alert.from_dict(record.value)))
            except InvalidMessage as exc:
                self._dead_letter(record, exc)
        if not valid:
            return
        self._last_alert = self.clock()
        self.closed_on_idle = False
        # Consecutive alerts from the same night are scored as one batch.
        for night, group in itertools.groupby(valid, key=lambda pair: pair[1].night):
            batch = list(group)
            late = ((self.last_closed is not None and night <= self.last_closed)
                    or (self.night is not None and night < self.night))
            if not late:
                if self.night is not None and night > self.night:
                    self.close_night(reason=f"first alert of {night} arrived")
                if self.night is None:
                    self.night = night
                    log.info("night %s opened", night)
            self._score(batch, store=not late)
            self._uncommitted.extend(record for record, _ in batch)

    def _score(self, batch: list[tuple[Record, Alert]], store: bool) -> None:
        alerts = [alert for _, alert in batch]
        p_ia = self.model.score(pd.DataFrame([asdict(a) for a in alerts]))
        threshold = self.model.threshold
        for alert, p in zip(alerts, p_ia):
            c = Classification(ztf_id=alert.ztf_id, night=alert.night, p_ia=round(float(p), 6),
                               eligible=bool(p >= threshold), model_version=self.model.version)
            self.publisher.publish(self.settings.classifications_topic, key=alert.ztf_id, value=c.to_dict())
            self.audit.classification(c)
        if store:
            self.store.upsert((a.ztf_id, a.night, float(p), a.event_time, {f: getattr(a, f) for f in CANDIDATE_FIELDS})
                              for a, p in zip(alerts, p_ia))
        else:
            self.stats["late_alerts"] += len(alerts)
            log.warning("%d late alerts for closed night %s: classified but not selectable",
                        len(alerts), alerts[0].night)
        self.stats["alerts_scored"] += len(alerts)

    def _dead_letter(self, record: Record, error: Exception) -> None:
        self.stats["dead_lettered"] += 1
        log.warning("dead-lettering %s[%d]@%d: %s", record.topic, record.partition, record.offset, error)
        self.publisher.publish(self.settings.dead_letter_topic, key=record.key or "", value={
            "error": str(error), "topic": record.topic, "partition": record.partition, "offset": record.offset,
            "payload": record.raw.decode("utf-8", errors="replace"), "failed_at": utc_now()})
        self._uncommitted.append(record)

    # --- shortlist -----------------------------------------------------------------------------

    def _shortlist(self, final: bool) -> Shortlist:
        threshold, night = self.model.threshold, self.night
        top = self.store.top(night, threshold, self.settings.daily_capacity)
        return Shortlist(night=night, final=final, model_version=self.model.version, threshold=threshold,
                         capacity=self.settings.daily_capacity, alerts_tonight=self.store.count(night),
                         eligible_tonight=self.store.count(night, threshold),
                         candidates=[{"rank": i + 1, "ztf_id": z, "p_ia": round(p, 6), **details}
                                     for i, (z, p, details) in enumerate(top)])

    def rerank(self) -> Shortlist | None:
        """Re-rank tonight's candidates; publish the shortlist if its membership or order changed."""
        self._last_rerank = self.clock()
        if self.night is None:
            return None
        shortlist = self._shortlist(final=False)
        ids = [c["ztf_id"] for c in shortlist.candidates]
        if ids != self._shortlist_ids:
            entered, left = len(set(ids) - set(self._shortlist_ids)), len(set(self._shortlist_ids) - set(ids))
            self.publisher.publish(self.settings.shortlist_topic, key=shortlist.night, value=shortlist.to_dict())
            self.audit.live(shortlist)
            cutoff = shortlist.candidates[-1]["p_ia"] if shortlist.candidates else float("nan")
            log.info("night %s: %d alerts, %d eligible -> shortlist %d (+%d / -%d), lowest P(Ia) %.3f",
                     shortlist.night, shortlist.alerts_tonight, shortlist.eligible_tonight,
                     len(ids), entered, left, cutoff)
            self._shortlist_ids = ids
        return shortlist

    def close_night(self, reason: str) -> Shortlist:
        final = self._shortlist(final=True)
        self.publisher.publish(self.settings.selections_topic, key=final.night, value=final.to_dict())
        self.publisher.flush()                            # outputs are durable before inputs are committed
        self.audit.selection(final)
        self.audit.live(final)
        self.subscriber.commit(self._uncommitted)
        self._uncommitted.clear()
        self.store.evict(final.night)
        self.stats["nights_closed"] += 1
        log.info("night %s closed (%s): %d alerts, %d eligible, %d selected for follow-up",
                 final.night, reason, final.alerts_tonight, final.eligible_tonight, len(final.candidates))
        self.last_closed, self.night, self._shortlist_ids = final.night, None, []
        return final
