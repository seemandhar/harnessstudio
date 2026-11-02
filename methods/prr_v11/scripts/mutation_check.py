"""Mutation check (PITFALLS P-20261031-21: prove a checker by making it fail). Each mutation is applied to a temporary
copy of hs_prr_v11 placed first on PYTHONPATH; the package tests must FAIL for every mutation and PASS for the unmutated copy.

    harnessstudio/.venv/bin/python mutation_check.py OUT.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
PY = sys.executable
MUTATIONS = [
    ("M1 confirmation 10 steps instead of 15", "planner.py",
     "if d >= self.U + p.confirm_steps:", "if d >= self.U + p.confirm_steps - 5:"),
    ("M2 reversal rows not reversed", "planner.py",
     "for s in range(f - 1, cp - 1, -1)]", "for s in range(cp, f)]"),
    ("M3 gripper latch dropped (replays window gripper)", "planner.py",
     "np.concatenate([-self.rows[s][:6], [latch]])", "np.concatenate([-self.rows[s][:6], [self.rows[s][6]]])"),
    ("M4 C2 custody window disabled", "planner.py",
     "if dist > p.custody_tol_m:", "if dist > 1e9:"),
    ("M5 C4 horizon guard disabled", "planner.py",
     "elif horizon_end > p.horizon_guard_last_step:", "elif False:"),
    ("M6 re-arm without the 30-step gap", "planner.py",
     "self.a_armed, self.a_search, self.U = True, h + self.p.rearm_gap_steps, None",
     "self.a_armed, self.a_search, self.U = True, h, None"),
    ("M7 failed confirmation does not disarm route A", "planner.py",
     "                        self.a_armed = False\n                        self.events.append((\"prr_cancel\"",
     "                        pass\n                        self.events.append((\"prr_cancel\""),
    ("M8 ortho coin parity flipped", "controller.py",
     "return (1 if int(h, 16) % 2 == 0 else -1), text, h", "return (-1 if int(h, 16) % 2 == 0 else 1), text, h"),
    ("M9 hold executes PRR rows", "controller.py",
     'if self.variant == "hold":', 'if self.variant == "hold_disabled":'),
    ("M10 correction position law sign flipped", "controller.py",
     "(self.kp_pos * (np.asarray(cp_state[:3]) - st[:3]) - self.kd_pos * v)", "(self.kp_pos * (st[:3] - np.asarray(cp_state[:3])) - self.kd_pos * v)"),
    ("M14 v1.0.0 per-element deadbeat law restored (position)", "controller.py",
     "u = (self.kp_pos * (np.asarray(cp_state[:3]) - st[:3]) - self.kd_pos * v) / self.pos_scale\n            a[:3] = self._sat(u)",
     "a[:3] = np.clip((np.asarray(cp_state[:3]) - st[:3]) / 0.0054, -1.0, 1.0)"),
    ("M15 v1.0.0 per-element deadbeat law restored (rotation)", "controller.py",
     "u = (self.kp_rot * rot_error(cp_state[3:6], st[3:6]) - self.kd_rot * w) / self.rot_scale\n            a[3:6] = self._sat(u)",
     "a[3:6] = np.clip(rot_error(cp_state[3:6], st[3:6]) / 0.0324, -1.0, 1.0)"),
    ("M16 velocity damping sign flipped (anti-damping)", "controller.py",
     "- self.kd_pos * v) / self.pos_scale", "+ self.kd_pos * v) / self.pos_scale"),
    ("M17 rotation velocity estimate frame flipped", "controller.py",
     "w = rot_error(st[3:6], prev[3:6])", "w = rot_error(prev[3:6], st[3:6])"),
    ("M11 checkpoint ignores gripper sign", "planner.py",
     "and self.gs.get(k - 1) == latch)", ")"),
    ("M12 route B window may start at the hand-back (settle-produced sample)", "planner.py",
     "if start < p.stall_earliest_step or (self.h is not None and start <= self.h):",
     "if start < p.stall_earliest_step or (self.h is not None and start < self.h):"),
    ("M13 checkpoint may be the hand-back decision", "planner.py",
     "(self.h is None or k > self.h)),", "(self.h is None or k >= self.h)),"),
]


def run(tmp: Path) -> tuple:
    # cwd = the temporary copy: 'python -m' puts the cwd first on sys.path, so the mutated hs_prr is the one imported
    env = dict(os.environ, PYTHONPATH=str(tmp), PYTHONDONTWRITEBYTECODE="1")
    t = PKG / "tests"
    probe = subprocess.run([PY, "-c", "import hs_prr_v11; print(hs_prr_v11.__file__)"], cwd=tmp, env=env, capture_output=True,
                           text=True).stdout.strip()
    assert probe.startswith(str(tmp)), ("mutated copy not imported", probe)
    fast = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(t / "test_planner.py"),
                           str(t / "test_replay_lock.py")], cwd=tmp, env=env, capture_output=True, text=True)
    if fast.returncode != 0:
        return fast.returncode, "fast", fast.stdout.strip().splitlines()[-1:]
    slow = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", str(t / "test_fake_stack.py")],
                          cwd=tmp, env=env, capture_output=True, text=True)
    return slow.returncode, "fake_stack", slow.stdout.strip().splitlines()[-1:]


def main(out: str) -> int:
    results = []
    for label, fname, old, new in [("M0 unmutated", None, None, None)] + MUTATIONS:
        with tempfile.TemporaryDirectory(prefix="prr-mut-") as d:
            tmp = Path(d)
            shutil.copytree(PKG / "hs_prr_v11", tmp / "hs_prr_v11", ignore=shutil.ignore_patterns("__pycache__"))
            if fname:
                f = tmp / "hs_prr_v11" / fname
                text = f.read_text()
                assert text.count(old) == 1, (label, text.count(old))
                f.write_text(text.replace(old, new))
            code, stage, tail = run(tmp)
        ok = (code == 0) if fname is None else (code != 0)
        results.append({"mutation": label, "file": fname, "tests_exit": code, "failed_at": stage if code else None,
                        "tail": tail, "as_expected": ok})
        print(json.dumps(results[-1]), flush=True)
    doc = {"all_as_expected": all(r["as_expected"] for r in results), "results": results}
    Path(out).write_text(json.dumps(doc, indent=1))
    return 0 if doc["all_as_expected"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
