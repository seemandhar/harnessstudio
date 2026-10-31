# Deferred core changes

Changes that would alter behaviour under runs still in progress are applied only when no paper run is active.

| # | From | Change | Why deferred |
|---|---|---|---|
| 1 (DONE 2026-10-31 ~11:00, merged) | Harness VLA results (Core request 1) | Case key includes the benchmark's `task_id` when the EPISODE_START task payload has one (`suite/name#task_id`), and trace file names use the same key. LIBERO-Pro `libero_goal_task` has two tasks with the same instruction, so keys collide and `summarize` raises "duplicate case keys". | Changes every LIBERO case key; SCALE pairs separately run arms and would mix old and new keys mid-run. |
| 2 | FastHarness results (Core request 5) | Per-dimension action clip bounds in contracts (RoboDojo clips the gripper to [0, 1]). | Nice to have. |
| 3 | FastHarness results (Core request 6) | A RoboDojo benchmark profile in the core (Kit render-sync settings, fast exit), replacing the overlay hook. | Needs the FastHarness run's findings first. |
| 4 | SCALE results (Core request 2) | `hs eval` episode offsets / `--cases`. | vla-eval always runs episodes 0..N-1; needs a benchmark-side option. |
| 5 | Package audit (2026-10-31) | Fingerprint the method *source* (not only the config), record the benchmark class and its params in every run manifest, and refuse to pair records whose method/protocol fingerprints differ (`eval/pairing.py`). | Would have flagged FastHarness Stage A's two fingerprints and SCALE's offset benchmark automatically. |
| 6 | Package audit | A charged `ctx.tools` entry for external tool servers (e.g. SAM3), enforcing `budgets.tool_calls` and recording input hashes and wall time. | Harness VLA calls SAM3 over HTTP directly. |
| 7 | Package audit | Make development-case refusal namespace-aware for replayed scenes (a benchmark that pins an init state should declare which evaluation keys it covers). | Harness VLA few-shot relies on `episode_offset` by construction. |
