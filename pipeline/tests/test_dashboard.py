"""The dashboard's HTTP API, served for real on a random local port."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from ztf_pipeline.dashboard import make_server


def candidate(rank, ztf_id, p):
    return {"rank": rank, "ztf_id": ztf_id, "p_ia": p, "ra_deg": 150.0, "dec_deg": -2.5, "peak_mag": 18.2,
            "peak_filter": "g", "rise_days": 9.0, "rise_is_limit": False, "fade_days": 14.0, "fade_is_limit": True,
            "discovered_at": "2024-01-01 00:00:00"}


def selection(night, candidates, final=True):
    return {"schema": "ztf.shortlist.v1", "night": night, "final": final, "model_version": "m1", "threshold": 0.885,
            "capacity": 100, "alerts_tonight": 333, "eligible_tonight": len(candidates), "candidates": candidates,
            "generated_at": "2024-01-02T00:00:00Z"}


@pytest.fixture
def server(settings):
    settings.output_dir.mkdir(parents=True)
    selections = [selection("2024-01-01", [candidate(1, "ZTF24aaa", 0.97)]),
                  selection("2024-01-02", [candidate(1, "ZTF24bbb", 0.99), candidate(2, "ZTF24ccc", 0.95)]),
                  selection("2024-01-02", [candidate(1, "ZTF24bbb", 0.99), candidate(2, "ZTF24ccc", 0.95)]),  # replayed
                  selection("2024-01-03", [], final=False)]                                                  # live, not final
    classifications = [{"ztf_id": z, "night": n, "p_ia": p, "eligible": p >= 0.885, "model_version": "m1"}
                       for z, n, p in [("ZTF24aaa", "2024-01-01", 0.97), ("ZTF24bbb", "2024-01-02", 0.99),
                                       ("ZTF24ccc", "2024-01-02", 0.95), ("ZTF24ddd", "2024-01-02", 0.30)]]
    settings.selections_log.write_text("".join(json.dumps(s) + "\n" for s in selections), encoding="utf-8")
    settings.classifications_log.write_text("".join(json.dumps(c) + "\n" for c in classifications), encoding="utf-8")

    srv = make_server(settings, port=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as r:
        return json.loads(r.read())


def test_page_is_served(server):
    with urllib.request.urlopen(server + "/") as r:
        assert r.status == 200 and b"SN Ia follow-up" in r.read()


def test_nights_are_listed_newest_first_and_only_once(server):
    nights = get(server + "/api/nights")
    assert [n["night"] for n in nights] == ["2024-01-02", "2024-01-01"]       # live night 3 not shown
    assert nights[0]["selected"] == 2 and nights[0]["observed"] == 0


def test_progress_is_saved_and_shown(server):
    saved = post(server + "/api/progress", {"night": "2024-01-02", "ztf_id": "ZTF24ccc", "status": "observed",
                                            "note": "good S/N", "by": "Ana"})
    assert saved["status"] == "observed" and saved["updated_by"] == "Ana"
    night = get(server + "/api/night?night=2024-01-02")
    by_id = {c["ztf_id"]: c for c in night["candidates"]}
    assert by_id["ZTF24ccc"]["note"] == "good S/N" and by_id["ZTF24bbb"]["status"] == "todo"
    assert by_id["ZTF24bbb"]["ra_deg"] == 150.0                                 # observing details included
    assert get(server + "/api/nights")[0]["observed"] == 1


@pytest.mark.parametrize("body", [
    {"night": "2024-01-02", "ztf_id": "ZTF24ccc", "status": "done"},          # unknown status
    {"night": "2024-01-02", "ztf_id": "ZTF24ddd", "status": "observed"},      # not selected that night
    {"night": "2024-01-02", "ztf_id": "ZTF24ccc", "status": "observed", "note": "x" * 501},
])
def test_bad_updates_are_rejected(server, body):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server + "/api/progress", body)
    assert exc.value.code == 400


def test_search_finds_selected_and_unselected_objects(server):
    results = {r["ztf_id"]: r for r in get(server + "/api/search?q=ztf24")}
    assert results["ZTF24bbb"]["selected_night"] == "2024-01-02" and results["ZTF24bbb"]["rank"] == 1
    assert results["ZTF24ddd"]["selected_night"] is None and not results["ZTF24ddd"]["eligible"]
    with pytest.raises(urllib.error.HTTPError):
        get(server + "/api/search?q=zt")                                        # too short


def test_unknown_night_is_a_bad_request(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server + "/api/night?night=1999-01-01")
    assert exc.value.code == 400
