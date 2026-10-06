"""Message transport: Kafka in production, an in-process broker for tests and Kafka-free runs.

Both implement the same two small interfaces, so the producer and the classifier service don't know
which one they're talking to.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

log = logging.getLogger(__name__)

# "localhost" also resolves to IPv6 ::1, where the local broker isn't listening; skip the failed attempt.
CLIENT_DEFAULTS = {"broker.address.family": "v4"}


@dataclass(frozen=True)
class Record:
    topic: str
    partition: int
    offset: int
    key: str | None
    value: dict | None          # None when the payload wasn't valid JSON
    raw: bytes = b""


class Publisher(Protocol):
    def publish(self, topic: str, key: str, value: dict) -> None: ...
    def flush(self, timeout: float = 30.0) -> None: ...


class Subscriber(Protocol):
    def poll(self, max_records: int = 500, timeout: float = 0.5) -> list[Record]: ...
    def commit(self, records: Iterable[Record]) -> None: ...    # commit through these records (inclusive)
    def close(self) -> None: ...


def encode(value: dict) -> bytes:
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")


def decode(payload: bytes) -> dict | None:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def next_offsets(records: Iterable[Record]) -> dict[tuple[str, int], int]:
    """For each (topic, partition), the offset to commit: one past the highest record."""
    offsets: dict[tuple[str, int], int] = {}
    for r in records:
        key = (r.topic, r.partition)
        offsets[key] = max(offsets.get(key, 0), r.offset + 1)
    return offsets


# --- Kafka ---------------------------------------------------------------------------------------

class KafkaPublisher:
    def __init__(self, bootstrap_servers: str):
        from confluent_kafka import Producer
        self._producer = Producer({
            **CLIENT_DEFAULTS,
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,     # no duplicates from retries; implies acks=all
            "linger.ms": 5,
            "compression.type": "lz4",
        })
        self._failures = 0

    def _on_delivery(self, err, msg) -> None:
        if err is not None:
            self._failures += 1
            log.error("delivery to %s failed: %s", msg.topic(), err)

    def publish(self, topic: str, key: str, value: dict) -> None:
        while True:
            try:
                self._producer.produce(topic, key=key.encode("utf-8"), value=encode(value),
                                       on_delivery=self._on_delivery)
                break
            except BufferError:             # local queue full: let it drain, then retry
                self._producer.poll(0.5)
        self._producer.poll(0)

    def flush(self, timeout: float = 30.0) -> None:
        remaining = self._producer.flush(timeout)
        if remaining or self._failures:
            raise RuntimeError(f"{remaining} messages undelivered, {self._failures} delivery failures")


class KafkaSubscriber:
    def __init__(self, bootstrap_servers: str, group_id: str, topics: list[str], start_from: str = "earliest"):
        from confluent_kafka import Consumer
        self._consumer = Consumer({
            **CLIENT_DEFAULTS,
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "enable.auto.commit": False,     # offsets are committed explicitly, see ClassifierService
            "auto.offset.reset": start_from,      # where a group with no committed offsets starts
        })
        self._consumer.subscribe(topics)

    def poll(self, max_records: int = 500, timeout: float = 0.5) -> list[Record]:
        from confluent_kafka import KafkaError, KafkaException
        records = []
        for msg in self._consumer.consume(num_messages=max_records, timeout=timeout):
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())
            key = msg.key().decode("utf-8") if msg.key() else None
            records.append(Record(msg.topic(), msg.partition(), msg.offset(), key, decode(msg.value()), msg.value()))
        return records

    def commit(self, records: Iterable[Record]) -> None:
        from confluent_kafka import TopicPartition
        offsets = [TopicPartition(t, p, o) for (t, p), o in next_offsets(records).items()]
        if offsets:
            self._consumer.commit(offsets=offsets, asynchronous=False)

    def close(self) -> None:
        self._consumer.close()


def create_topics(bootstrap_servers: str, topics: list[str], recreate: bool = False) -> None:
    """Create the topics with one partition each.

    One partition keeps alerts in arrival order, which the service relies on to know when a night has
    ended. Scaling out would mean partitioning by night, or tracking a watermark per partition.
    """
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({**CLIENT_DEFAULTS, "bootstrap.servers": bootstrap_servers})
    existing = set(admin.list_topics(timeout=10).topics)
    if recreate and existing & set(topics):
        for topic, future in admin.delete_topics(sorted(existing & set(topics))).items():
            future.result()
            log.info("deleted topic %s", topic)
        _wait_until_gone(admin, set(topics))
        existing -= set(topics)
    new = [NewTopic(t, num_partitions=1, replication_factor=1) for t in topics if t not in existing]
    for topic, future in (admin.create_topics(new).items() if new else []):
        future.result()
        log.info("created topic %s", topic)


def skip_to_end(bootstrap_servers: str, group_id: str, topic: str) -> int:
    """Move a consumer group to the end of a topic so it ignores everything already there.

    The same as `kafka-consumer-groups --reset-offsets --to-latest`. Run it while the group's consumers
    are stopped. Returns how many messages the group will skip.
    """
    from confluent_kafka import Consumer, TopicPartition
    consumer = Consumer({**CLIENT_DEFAULTS, "bootstrap.servers": bootstrap_servers, "group.id": group_id,
                         "enable.auto.commit": False})
    try:
        partitions = consumer.list_topics(topic, timeout=10).topics[topic].partitions
        ends = [TopicPartition(topic, p, consumer.get_watermark_offsets(TopicPartition(topic, p), timeout=10)[1])
                for p in partitions]
        committed = consumer.committed([TopicPartition(topic, p) for p in partitions], timeout=10)
        skipped = sum(end.offset - max(c.offset, 0) for end, c in zip(ends, committed))
        consumer.commit(offsets=ends, asynchronous=False)
        return skipped
    finally:
        consumer.close()


def _wait_until_gone(admin, topics: set[str], timeout: float = 30.0) -> None:
    import time
    deadline = time.monotonic() + timeout
    while topics & set(admin.list_topics(timeout=10).topics):
        if time.monotonic() > deadline:
            raise TimeoutError(f"topics still exist after deletion: {sorted(topics)}")
        time.sleep(0.5)


# --- in-process broker ---------------------------------------------------------------------------

class InMemoryBroker:
    """A minimal single-partition broker with consumer-group offsets, safe to use across threads."""

    def __init__(self):
        self._logs: dict[str, list[tuple[str, bytes]]] = defaultdict(list)
        self._committed: dict[tuple[str, str], int] = {}       # (group, topic) -> next offset
        self._changed = threading.Condition()

    def publisher(self) -> InMemoryPublisher:
        return InMemoryPublisher(self)

    def subscriber(self, group_id: str, topics: list[str]) -> InMemorySubscriber:
        return InMemorySubscriber(self, group_id, topics)

    def messages(self, topic: str) -> list[dict]:
        with self._changed:
            return [decode(raw) for _, raw in self._logs[topic]]


class InMemoryPublisher:
    def __init__(self, broker: InMemoryBroker):
        self._broker = broker

    def publish(self, topic: str, key: str, value: dict) -> None:
        with self._broker._changed:
            self._broker._logs[topic].append((key, encode(value)))
            self._broker._changed.notify_all()

    def flush(self, timeout: float = 30.0) -> None:
        pass


class InMemorySubscriber:
    def __init__(self, broker: InMemoryBroker, group_id: str, topics: list[str]):
        self._broker, self._group, self._topics = broker, group_id, topics
        with broker._changed:     # start from the group's committed position, like a Kafka consumer
            self._position = {t: broker._committed.get((group_id, t), 0) for t in topics}

    def _available(self) -> bool:
        return any(len(self._broker._logs[t]) > self._position[t] for t in self._topics)

    def poll(self, max_records: int = 500, timeout: float = 0.5) -> list[Record]:
        with self._broker._changed:
            self._broker._changed.wait_for(self._available, timeout=timeout)
            records = []
            for topic in self._topics:
                log_ = self._broker._logs[topic]
                while self._position[topic] < len(log_) and len(records) < max_records:
                    offset = self._position[topic]
                    key, raw = log_[offset]
                    records.append(Record(topic, 0, offset, key, decode(raw), raw))
                    self._position[topic] += 1
            return records

    def commit(self, records: Iterable[Record]) -> None:
        with self._broker._changed:
            for (topic, _), offset in next_offsets(records).items():
                self._broker._committed[(self._group, topic)] = offset

    def close(self) -> None:
        pass
