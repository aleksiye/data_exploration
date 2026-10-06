"""Local audit trail of everything the service decides, for the evaluator and for humans."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .config import Settings
from .messages import Classification, Shortlist


class AuditLog:
    def __init__(self, settings: Settings):
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self._classifications = open(settings.classifications_log, "a", encoding="utf-8", buffering=1)
        self._selections = open(settings.selections_log, "a", encoding="utf-8", buffering=1)
        self._live_path = settings.live_shortlist_path

    def classification(self, c: Classification) -> None:
        self._classifications.write(json.dumps(c.to_dict()) + "\n")

    def selection(self, s: Shortlist) -> None:
        self._selections.write(json.dumps(s.to_dict()) + "\n")

    def live(self, s: Shortlist) -> None:
        # Write to a temporary file and swap it in, so a reader never sees a half-written file.
        tmp = self._live_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(s.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, self._live_path)

    def close(self) -> None:
        self._classifications.close()
        self._selections.close()


def clear_outputs(settings: Settings) -> None:
    for path in (settings.classifications_log, settings.selections_log, settings.live_shortlist_path,
                 settings.report_path, settings.report_path.with_suffix(".json"), settings.worklist_path):
        try:
            Path(path).unlink(missing_ok=True)
        except PermissionError as exc:          # Windows: the file is open, e.g. by a running dashboard
            raise RuntimeError(f"can't delete {path}: stop the dashboard and the classifier first") from exc
