# Streaming SN Ia candidate selection

A production-style version of notebooks 01–05. Discoveries are replayed through **Kafka** as if they were happening live, a classifier service scores each alert as it arrives, and every night it hands the follow-up team **at most 100 candidates**, chosen to be **≥ 95% SN Ia**. Afterwards, an evaluator cross-references every decision with the original catalogue.

```
 stream.csv ──► replay ──► [ztf.alerts] ──► classifier service ──► [ztf.classifications]  every alert's P(Ia)
 (2024–25,                                   │  in-memory SQLite  ──► [ztf.shortlist]        live top 100, re-ranked
  no labels)                                 │  of tonight's      ──► [ztf.selections]       final list per night
                                             │  candidates        ──► [ztf.alerts.dlq]       malformed alerts
                                             └─► output/*.jsonl (audit log) ─┬─► evaluate ◄── bts.csv (labels)
                                                                              └─► dashboard  (the team's worklist)
```

## Results

From a full run: 1,791 alerts over 6 simulated nights of 333 alerts (about 10,000 a month). The classifier was killed in the middle of night 3 and restarted.

| Check | Result |
|---|---|
| Every streamed alert classified exactly once (after de-duplication) | ✅ 1,791 / 1,791 |
| No streamed object in the training data | ✅ |
| Streamed scores identical to offline scoring (no training/serving skew) | ✅ largest difference 5 × 10⁻⁷ (rounding) |
| At most 100 candidates per night | ✅ 100, 100, 98, 100, 100, 34 |
| **Precision ≥ 95%** | ✅ **96.2%**: 512 of 532 selected were SN Ia (95% interval 94.3–97.6%) |
| Nothing lost after the crash | ✅ night 3 was replayed from Kafka and selected exactly as in an uninterrupted run |

Two things to know about the precision:
* **It varies from night to night:** 96, 93, 96, 98, 97 and 100%. The target is met overall, not on every single night. Notebook 05 shows how to add a safety margin if every night must clear 95%.
* **"95% precision" means 5% of the *selected* candidates are wrong** (the false discovery rate). The false-*positive* rate is something different: the share of all non-Ia that get selected. Here it's 20 of 476, or 4.2%.

The 20 false positives were 11 SN II, 7 SN Ib/c, 1 SLSN and 1 peculiar SN. These are the classes the notebooks found hardest to separate from Ia using light-curve summaries.

## Running it

All commands run from this `pipeline/` folder, using the project's virtual environment (`..\.venv\Scripts\python`).

**1. Build the data and the model** (no Kafka needed):

```powershell
..\.venv\Scripts\python -m ztf_pipeline prepare     # clean bts.csv, split by time, write data/train.parquet + data/stream.csv
..\.venv\Scripts\python -m ztf_pipeline train       # train, choose the 95%-precision threshold, write artifacts/
```

**2. Start a Kafka broker on `localhost:9092`.** Use either of these:

```powershell
docker compose up -d                                    # with Docker
powershell -ExecutionPolicy Bypass -File scripts\start-kafka.ps1   # natively on Windows (needs Java 17+)
```

The script downloads Kafka 4.3.1 once into `.kafka\` and verifies its checksum. It runs in the foreground; stop it with Ctrl+C. Add `-Wipe` to start with empty storage.

**3. Run the stream.** Use one terminal per process, as in production:

```powershell
..\.venv\Scripts\python -m ztf_pipeline reset       # fresh run: skip old alerts, clear output/
..\.venv\Scripts\python -m ztf_pipeline classify    # terminal 1: the classifier service (Ctrl+C to stop)
..\.venv\Scripts\python -m ztf_pipeline watch       # terminal 2: live shortlist, a stand-in for a dashboard
..\.venv\Scripts\python -m ztf_pipeline produce --seconds-per-night 10   # terminal 3: replay the alerts
..\.venv\Scripts\python -m ztf_pipeline evaluate    # afterwards: cross-check with bts.csv -> output/evaluation_report.md
```

Or run everything in one process. That's handy for a quick check, and `--broker memory` doesn't even need Kafka:

```powershell
..\.venv\Scripts\python -m ztf_pipeline demo --broker memory
..\.venv\Scripts\python -m ztf_pipeline demo --broker kafka
```

**4. The team's dashboard.** This works whenever there are closed nights in `output/`:

```powershell
..\.venv\Scripts\python -m ztf_pipeline dashboard --open      # http://127.0.0.1:8050
```

* **It opens on last night's backlog:** the selected candidates, best first, with coordinates (copy-paste ready for the telescope), peak magnitude, rise/fade times and links to ALeRCE, Fink and Lasair.
* **Everyone works from one list.** Mark each candidate *to do*, *observed* or *skipped*, and add a note. Your name (set once, top right) is recorded with each change. The page picks up teammates' changes every 30 s, and progress is saved in `output/worklist.db`.
* **Use the filter chips** (To do, Observed, Skipped, All) to focus on what's left. *Copy to-do targets* puts a plain-text target list on the clipboard; *Export CSV* saves the whole night.
* **Earlier nights** are in the sidebar, with their progress.
* **Search** finds any classified object by ZTF ID across all nights, including ones that weren't selected, with the reason (below the threshold, or the night was full). A result that made the list opens its night with the row highlighted.
* **Links can be bookmarked or shared:** `#night=2024-01-05`, or `#q=ZTF25acg`.

It binds to `127.0.0.1` by default. `--host 0.0.0.0` shares it on your network, but there's no login, so only do that on a trusted network.

**Tests:** `..\.venv\Scripts\python -m pytest` (32 tests, about 20 s, no Kafka needed).

**Configuration:** every setting in `ztf_pipeline/config.py` can be overridden with an environment variable, for example `$env:ZTF_DAILY_CAPACITY = 150` or `$env:ZTF_BOOTSTRAP_SERVERS = "broker:9092"`.

## Design decisions

**Leakage is blocked by an allow-list.** `features.py` lists the *only* catalogue columns allowed into an alert. Anything not on the list never reaches the stream: type, IAU name, redshift and the redshift-derived columns. A second list, `LEAKY_COLUMNS`, records *why* each excluded column leaks, and tests check that the two lists never overlap. The labels for streamed objects exist only in the original `bts.csv`, and only the evaluator reads them.

**Train on the past, stream the future.** The model trains on objects discovered before 2024 (6,041). The stream is the labelled objects from 2024–25 (1,791). This is the honest time-based test from notebooks 03–05; a random split would overstate performance.

**One feature function for training and serving.** `build_features` is used both by `train` and inside the service. The evaluator re-scores the stream offline and confirms that the streamed probabilities match. Code that builds features twice tends to drift apart; this check would catch it.

**The precision threshold is chosen on training data only.** It comes from out-of-fold predictions on the training set. It's the lowest P(Ia) above which precision stays ≥ 95% *for every higher threshold too*, which avoids being fooled by a lucky bump in the precision curve. Here it's P(Ia) ≥ 0.885. The threshold, hyperparameters, data hash and validation scores are saved in `artifacts/model_card.json`, and every message carries the model version.

**Tonight's candidates live in an in-memory SQLite database.** Each alert is scored on arrival and upserted, together with what the team needs to observe it (coordinates, magnitude, timescales). Every selection list therefore stands on its own. Every 2 s the service re-ranks: it takes the top 100 above the threshold and publishes the list to `ztf.shortlist` whenever it changes. A night closes when the first alert of the next night arrives, or after 10 s of silence. Its final list then goes to `ztf.selections`, and the night is removed from the database.

**At-least-once delivery, with no lost candidates.** Input offsets are committed *only when a night closes*, after its outputs have been flushed. If the service dies mid-night, a restart replays that night from Kafka and rebuilds the database; the crash test above does exactly this. The price is that some classifications are published twice, so consumers keep the latest message per `ztf_id` (or per night for selections).

**Bad input doesn't stop the service.** Alerts are validated against a schema (`messages.py`). Malformed ones go to a dead-letter topic with the error, and processing continues.

**One partition per topic.** The service decides that a night is over from the order in which alerts arrive, and Kafka only guarantees order within a partition. Scaling out would mean partitioning by night, or tracking a "watermark" per partition.

**A fresh run skips old messages instead of deleting topics.** `reset` moves the classifier's consumer group to the end of the alerts topic, the same as `kafka-consumer-groups --reset-offsets --to-latest`. Deleting topics also works on Linux, but crashes a Kafka broker running natively on Windows, because Windows won't rename files that are still open.

## Files

```
ztf_pipeline/
  config.py      settings, overridable with ZTF_* environment variables
  cleaning.py    notebook 01's cleaning, as functions
  features.py    alert allow-list, leaky columns with reasons, build_features
  prepare.py     time split -> data/train.parquet, data/stream.csv, data/manifest.json
  train.py       model, threshold selection, artifacts/model.joblib + model_card.json
  model.py       loads the model and checks it matches the feature code
  messages.py    Alert / Classification / Shortlist schemas and validation
  transport.py   Kafka publisher/subscriber, topic admin, and an in-memory broker
  replay.py      the producer: replays stream.csv night by night in simulated real time
  store.py       in-memory SQLite store of tonight's candidates
  service.py     the classifier service
  audit.py       local audit log (output/*.jsonl, output/shortlist_live.json)
  evaluate.py    cross-reference with bts.csv -> output/evaluation_report.md (+ .json)
  dashboard.py   the team's web dashboard: JSON API + worklist (output/worklist.db), standard library only
  dashboard.html the page itself (plain HTML/CSS/JavaScript, no external dependencies)
  __main__.py    command-line interface
tests/           unit tests and service tests on the in-memory broker
scripts/start-kafka.ps1   native Windows Kafka broker
docker-compose.yml        Kafka broker in Docker
```

## Limitations and next steps

* **Rise, fade and duration aren't known at discovery.** A real alert carries a growing light curve, not final summaries. The honest next step is to score each alert repeatedly as its light curve develops (notebook 03's "early classification").
* **Only labelled objects are streamed,** so that every decision can be checked. A real stream would also contain objects that never get a spectrum, and probably a different Ia fraction. Notebook 05 shows how much that matters.
* **Monitoring:** the team's spectra are free labels. Feeding them back as a `ztf.followup` topic, and alerting on the 7-day false-positive rate, would complete the loop from notebook 05.
* **Model registry and rollout:** the model is a file. A real deployment would version models in a registry and switch between them deliberately; the `model_version` field on every message is already there for that.
