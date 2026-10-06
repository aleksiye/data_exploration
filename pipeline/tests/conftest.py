from __future__ import annotations

from dataclasses import replace

import pytest

from ztf_pipeline.config import Settings


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings that write everything under a temporary directory (the raw CSV is the real one)."""
    return replace(Settings(), data_dir=tmp_path / "data", artifacts_dir=tmp_path / "artifacts",
                   output_dir=tmp_path / "output", rerank_interval_s=1.0, idle_close_s=5.0)


@pytest.fixture(scope="session")
def prepared(tmp_path_factory) -> Settings:
    """Run prepare + train once for the whole session."""
    from ztf_pipeline.prepare import prepare
    from ztf_pipeline.train import train
    root = tmp_path_factory.mktemp("prepared")
    s = replace(Settings(), data_dir=root / "data", artifacts_dir=root / "artifacts", output_dir=root / "output")
    prepare(s)
    train(s)
    return s


def make_alert(ztf_id: str, night: str, **overrides) -> dict:
    alert = {
        "schema": "ztf.alert.v1", "ztf_id": ztf_id, "night": night, "event_time": f"{night}T00:00:00Z",
        "discovered_at": f"{night} 00:00:00", "ra_deg": 10.0, "dec_deg": 20.0, "gal_b": 45.0, "a_v": 0.1,
        "peak_filter": "g", "peak_mag": 18.0, "rise_days": 10.0, "rise_is_limit": False, "fade_days": 14.0,
        "fade_is_limit": False, "duration_days": 24.0, "duration_is_limit": False,
    }
    return {**alert, **overrides}
