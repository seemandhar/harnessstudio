"""GPU-hour ledger. The cap lives in the ledger file (`cap_hours`); CAP_HOURS is only the default for a new file."""
from __future__ import annotations

import json
from pathlib import Path

CAP_HOURS = 6.0


def used_hours(runs: list) -> float:
    total = 0.0
    for run in runs:
        state = Path(run) / "state.json"
        if state.is_file():
            total += float(json.loads(state.read_text()).get("gpu_hours_upper_bound", 0.0))
    return total


class Ledger:
    def __init__(self, path: Path, cap: float = CAP_HOURS):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.is_file() else {"runs": [], "cap_hours": cap}
        self.cap = float(self.data.get("cap_hours", cap))

    def set_cap(self, hours: float, reason: str) -> None:
        """Change the cap (a user budget decision), keeping the history in the file."""
        self.data.setdefault("cap_history", []).append({"from": self.cap, "to": float(hours), "reason": reason})
        self.data["cap_hours"] = self.cap = float(hours)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def used(self) -> float:
        manual = sum(float(a["hours"]) for a in self.data.get("adjustments", []))
        return used_hours(self.data["runs"]) + manual

    def adjust(self, run_dir, hours: float, reason: str) -> None:
        """Count GPU time a run consumed but did not record (e.g. an attempt whose state was not finalized)."""
        self.data.setdefault("adjustments", []).append({"run": str(run_dir), "hours": float(hours), "reason": reason})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def require(self, projected_hours: float) -> None:
        if self.used() + projected_hours > self.cap:
            raise RuntimeError(f"GPU ledger: {self.used():.2f} h used + {projected_hours:.2f} h projected "
                               f"exceeds the {self.cap} h cap")

    def add(self, run_dir) -> None:
        if str(run_dir) not in self.data["runs"]:
            self.data["runs"].append(str(run_dir))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))
