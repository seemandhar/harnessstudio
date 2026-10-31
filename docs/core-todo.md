# Deferred core changes

Changes that would alter behaviour under runs still in progress are applied only when no paper run is active.

| # | From | Change | Why deferred |
|---|---|---|---|
| 1 (ready on branch v2-casekey, merge after the running agents finish) | Harness VLA results (Core request 1) | Case key includes the benchmark's `task_id` when the EPISODE_START task payload has one (`suite/name#task_id`), and trace file names use the same key. LIBERO-Pro `libero_goal_task` has two tasks with the same instruction, so keys collide and `summarize` raises "duplicate case keys". | Changes every LIBERO case key; SCALE pairs separately run arms and would mix old and new keys mid-run. |
| 2 | FastHarness results (Core request 5) | Per-dimension action clip bounds in contracts (RoboDojo clips the gripper to [0, 1]). | Nice to have. |
| 3 | FastHarness results (Core request 6) | A RoboDojo benchmark profile in the core (Kit render-sync settings, fast exit), replacing the overlay hook. | Needs the FastHarness run's findings first. |
| 4 | SCALE results (Core request 2) | `hs eval` episode offsets / `--cases`. | vla-eval always runs episodes 0..N-1; needs a benchmark-side option. |
