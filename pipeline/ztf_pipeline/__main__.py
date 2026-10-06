"""Command-line entry point: python -m ztf_pipeline <command> [options]."""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import uuid

from .audit import AuditLog, clear_outputs
from .config import Settings

log = logging.getLogger("ztf_pipeline")


def _stop_on_signals() -> threading.Event:
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    return stop


def cmd_prepare(settings: Settings, args) -> int:
    from .prepare import prepare
    print(json.dumps(prepare(settings), indent=2))
    return 0


def cmd_train(settings: Settings, args) -> int:
    from .train import train
    card = train(settings)
    print(json.dumps({k: card[k] for k in ("model_version", "selection_policy", "validation")}, indent=2))
    return 0


def cmd_topics(settings: Settings, args) -> int:
    from .transport import create_topics
    create_topics(settings.bootstrap_servers, settings.topics, recreate=args.recreate)
    return 0


def _fresh_kafka_run(settings: Settings) -> None:
    """Make sure the topics exist, move the classifier past every alert already in Kafka, clear outputs.

    Old messages stay in the topics; the classifier's consumer group simply starts after them. (Deleting
    topics would also work, but it crashes Kafka brokers running natively on Windows.)
    """
    from .transport import create_topics, skip_to_end
    create_topics(settings.bootstrap_servers, settings.topics)
    skipped = skip_to_end(settings.bootstrap_servers, settings.consumer_group, settings.alerts_topic)
    clear_outputs(settings)
    log.info("consumer group %s will skip %d earlier alerts; outputs cleared", settings.consumer_group, skipped)


def cmd_reset(settings: Settings, args) -> int:
    """Start a fresh run. Stop the classifier service first."""
    _fresh_kafka_run(settings)
    return 0


def cmd_produce(settings: Settings, args) -> int:
    from .replay import replay
    from .transport import KafkaPublisher
    replay(settings.stream_path, KafkaPublisher(settings.bootstrap_servers), settings.alerts_topic,
           alerts_per_night=None if args.real_nights else args.alerts_per_night,
           seconds_per_night=args.seconds_per_night, limit=args.limit, stop=_stop_on_signals())
    return 0


def cmd_classify(settings: Settings, args) -> int:
    from .model import ModelBundle
    from .service import ClassifierService
    from .transport import KafkaPublisher, KafkaSubscriber
    subscriber = KafkaSubscriber(settings.bootstrap_servers, settings.consumer_group, [settings.alerts_topic])
    audit = AuditLog(settings)
    service = ClassifierService(settings, ModelBundle.load(settings), subscriber,
                                KafkaPublisher(settings.bootstrap_servers), audit)
    try:
        service.run(_stop_on_signals(), exit_when_idle=args.exit_when_idle)
    finally:
        subscriber.close()
        audit.close()
    return 0


def cmd_watch(settings: Settings, args) -> int:
    """Print tonight's shortlist as the service publishes it (a stand-in for a team dashboard)."""
    from .transport import KafkaSubscriber
    subscriber = KafkaSubscriber(settings.bootstrap_servers, f"watch-{uuid.uuid4().hex[:8]}",
                                 [settings.shortlist_topic, settings.selections_topic], start_from="latest")
    stop = _stop_on_signals()
    try:
        while not stop.is_set():
            for record in subscriber.poll(timeout=1.0):
                s = record.value
                top = ", ".join(f"{c['ztf_id']} ({c['p_ia']:.3f})" for c in s["candidates"][:args.top])
                print(f"{s['night']} {'FINAL' if s['final'] else 'live '} | {s['alerts_tonight']:>4} alerts, "
                      f"{s['eligible_tonight']:>4} eligible, {len(s['candidates']):>3} on the list | top: {top}",
                      flush=True)
    finally:
        subscriber.close()
    return 0


def cmd_dashboard(settings: Settings, args) -> int:
    import webbrowser
    from .dashboard import make_server
    server = make_server(settings, args.host, args.port)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{server.server_port}/"
    if args.host not in ("127.0.0.1", "localhost"):
        log.warning("serving on %s: anyone who can reach this machine can change the worklist", args.host)
    log.info("dashboard running at %s (Ctrl+C to stop)", url)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def cmd_evaluate(settings: Settings, args) -> int:
    from .evaluate import evaluate
    return 0 if evaluate(settings) else 1


def cmd_demo(settings: Settings, args) -> int:
    """The whole stream in one process: replay in a background thread, classify, then evaluate."""
    from .evaluate import evaluate
    from .model import ModelBundle
    from .replay import replay
    from .service import ClassifierService
    from .transport import InMemoryBroker, KafkaPublisher, KafkaSubscriber

    clear_outputs(settings)
    if args.broker == "memory":
        broker = InMemoryBroker()
        replay_publisher, service_publisher = broker.publisher(), broker.publisher()
        subscriber = broker.subscriber(settings.consumer_group, [settings.alerts_topic])
    else:
        _fresh_kafka_run(settings)
        replay_publisher = KafkaPublisher(settings.bootstrap_servers)
        service_publisher = KafkaPublisher(settings.bootstrap_servers)
        subscriber = KafkaSubscriber(settings.bootstrap_servers, settings.consumer_group, [settings.alerts_topic])

    stop = _stop_on_signals()
    producer = threading.Thread(name="replay", daemon=True, target=replay, kwargs=dict(
        stream_path=settings.stream_path, publisher=replay_publisher, topic=settings.alerts_topic,
        alerts_per_night=args.alerts_per_night, seconds_per_night=args.seconds_per_night, stop=stop))
    audit = AuditLog(settings)
    service = ClassifierService(settings, ModelBundle.load(settings), subscriber, service_publisher, audit)
    producer.start()
    try:
        service.run(stop, exit_when_idle=True)
    finally:
        stop.set()
        producer.join(timeout=10)
        subscriber.close()
        audit.close()
    return 0 if evaluate(settings) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ztf_pipeline", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("prepare", help="clean bts.csv, split by time, write train.parquet and stream.csv")
    commands.add_parser("train", help="train the classifier and choose the precision threshold")
    p = commands.add_parser("topics", help="create the Kafka topics")
    p.add_argument("--recreate", action="store_true",
                   help="delete and recreate them, dropping all messages (crashes a native Windows broker)")
    commands.add_parser("reset", help="fresh run: skip alerts already in Kafka and clear local outputs")

    p = commands.add_parser("produce", help="replay stream.csv into the alerts topic in (simulated) real time")
    p.add_argument("--alerts-per-night", type=int, default=333, help="pack alerts into nights of this size")
    p.add_argument("--real-nights", action="store_true", help="use real discovery dates instead")
    p.add_argument("--seconds-per-night", type=float, default=20.0, help="wall-clock length of a night (0 = no waiting)")
    p.add_argument("--limit", type=int, help="only replay the first N alerts")

    p = commands.add_parser("classify", help="run the classifier service")
    p.add_argument("--exit-when-idle", action="store_true", help="stop once the stream goes idle")

    p = commands.add_parser("watch", help="print the live shortlist as it is published")
    p.add_argument("--top", type=int, default=3, help="how many top candidates to print")

    p = commands.add_parser("dashboard", help="web page for the follow-up team: nightly worklists and search")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share it on your network (no login!)")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--open", action="store_true", help="open it in the browser")

    commands.add_parser("evaluate", help="cross-reference the results with bts.csv and check the targets")

    p = commands.add_parser("demo", help="replay + classify + evaluate in one process")
    p.add_argument("--broker", choices=["memory", "kafka"], default="memory")
    p.add_argument("--alerts-per-night", type=int, default=333)
    p.add_argument("--seconds-per-night", type=float, default=5.0)

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), stream=sys.stderr,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    settings = Settings.from_env()
    handler = globals()[f"cmd_{args.command}"]
    return handler(settings, args)


if __name__ == "__main__":
    sys.exit(main())
