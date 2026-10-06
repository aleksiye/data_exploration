"""The classifier service against the in-memory broker, with a scripted model and a fake clock."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_alert
from ztf_pipeline.audit import AuditLog
from ztf_pipeline.service import ClassifierService
from ztf_pipeline.transport import InMemoryBroker


class ScriptedModel:
    """Returns a preset P(Ia) per ztf_id."""
    version, threshold = "test-model", 0.8

    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def score(self, alerts: pd.DataFrame) -> np.ndarray:
        return alerts["ztf_id"].map(self.scores).to_numpy(dtype=float)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def setup(settings):
    settings = replace(settings, daily_capacity=3)
    broker = InMemoryBroker()
    clock = FakeClock()
    audits = []

    def start(scores: dict[str, float]) -> ClassifierService:
        audit = AuditLog(settings)
        audits.append(audit)
        return ClassifierService(settings, ScriptedModel(scores), broker.subscriber(settings.consumer_group,
                                 [settings.alerts_topic]), broker.publisher(), audit, clock=clock)

    yield settings, broker, clock, start
    for audit in audits:
        audit.close()


def publish(broker, settings, alerts):
    publisher = broker.publisher()
    for a in alerts:
        publisher.publish(settings.alerts_topic, a["ztf_id"], a)


def drain(service, steps=5):
    for _ in range(steps):
        service.step()


def selections(broker, settings) -> dict[str, list[str]]:
    return {s["night"]: [c["ztf_id"] for c in s["candidates"]] for s in broker.messages(settings.selections_topic)}


def test_selects_the_top_eligible_candidates_per_night(setup):
    settings, broker, _, start = setup
    scores = {"a1": 0.95, "a2": 0.70, "a3": 0.99, "a4": 0.85, "a5": 0.90, "b1": 0.81, "b2": 0.10}
    service = start(scores)
    publish(broker, settings, [make_alert(z, "2024-01-01") for z in ["a1", "a2", "a3", "a4", "a5"]]
            + [make_alert(z, "2024-01-02") for z in ["b1", "b2"]])
    drain(service)
    # night 1 closed when night 2 began: capacity 3, threshold 0.8, best first
    assert selections(broker, settings) == {"2024-01-01": ["a3", "a1", "a5"]}
    assert len(broker.messages(settings.classifications_topic)) == 7


def test_night_closes_after_idle_time_and_offsets_commit_only_then(setup):
    settings, broker, clock, start = setup
    service = start({"a1": 0.9, "a2": 0.95})
    publish(broker, settings, [make_alert("a1", "2024-01-01"), make_alert("a2", "2024-01-01")])
    drain(service)
    assert broker._committed == {}                                    # night still open: nothing committed
    clock.now += settings.idle_close_s + 1
    service.step()
    assert selections(broker, settings) == {"2024-01-01": ["a2", "a1"]}
    assert broker._committed[(settings.consumer_group, settings.alerts_topic)] == 2


def test_restart_mid_night_replays_and_loses_nothing(setup):
    settings, broker, clock, start = setup
    scores = {"a1": 0.99, "a2": 0.98, "a3": 0.97, "b1": 0.90, "b2": 0.95, "c1": 0.5}
    first = start(scores)
    publish(broker, settings, [make_alert(z, "2024-01-01") for z in ["a1", "a2", "a3"]]
            + [make_alert(z, "2024-01-02") for z in ["b1", "b2"]])
    drain(first)                     # night 1 closed; night 2 open with b1, b2 in memory... then a crash
    del first

    second = start(scores)           # same consumer group: resumes from the last commit (end of night 1)
    publish(broker, settings, [make_alert("c1", "2024-01-03")])
    drain(second)
    assert selections(broker, settings)["2024-01-02"] == ["b2", "b1"]     # b1, b2 were replayed, not lost


def test_bad_messages_go_to_the_dead_letter_topic(setup):
    settings, broker, clock, start = setup
    service = start({"a1": 0.9})
    publisher = broker.publisher()
    publisher.publish(settings.alerts_topic, "x", {"schema": "ztf.alert.v1", "ztf_id": "x"})
    publisher.publish(settings.alerts_topic, "a1", make_alert("a1", "2024-01-01"))
    drain(service)
    dead = broker.messages(settings.dead_letter_topic)
    assert len(dead) == 1 and "missing fields" in dead[0]["error"]
    assert service.stats["alerts_scored"] == 1


def test_late_alerts_are_classified_but_not_selected(setup):
    settings, broker, clock, start = setup
    service = start({"a1": 0.9, "b1": 0.9, "late": 0.99})
    publish(broker, settings, [make_alert("a1", "2024-01-01"), make_alert("b1", "2024-01-02"),
                               make_alert("late", "2024-01-01")])
    drain(service)
    clock.now += settings.idle_close_s + 1
    service.step()
    assert selections(broker, settings) == {"2024-01-01": ["a1"], "2024-01-02": ["b1"]}
    assert service.stats["late_alerts"] == 1
    assert "late" in {c["ztf_id"] for c in broker.messages(settings.classifications_topic)}


def test_live_shortlist_is_published_when_it_changes(setup):
    settings, broker, clock, start = setup
    service = start({"a1": 0.9, "a2": 0.95, "a3": 0.5})
    publish(broker, settings, [make_alert("a1", "2024-01-01")])
    drain(service, 1)
    service.rerank()
    publish(broker, settings, [make_alert("a2", "2024-01-01"), make_alert("a3", "2024-01-01")])
    drain(service, 1)
    service.rerank()
    service.rerank()                                  # unchanged: not published again
    live = broker.messages(settings.shortlist_topic)
    assert [[c["ztf_id"] for c in s["candidates"]] for s in live] == [["a1"], ["a2", "a1"]]
    assert all(not s["final"] for s in live)
