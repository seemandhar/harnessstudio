"""Compare two vla-eval result directories step by step (benchmark-side recordings).

    python scripts/compare_vla_eval_runs.py RUN_A RUN_B

For every episode (matched by task and episode id) it reports outcome, length and the first step where
the recorded action or state differs. Used for the M1 transparency gate (direct vs through the harness).
"""
import glob
import json
import sqlite3
import sys


def load(run_dir):
    dbs = glob.glob(f"{run_dir}/**/recording-*.sqlite", recursive=True)
    if len(dbs) != 1:
        raise SystemExit(f"{run_dir}: expected one recording-*.sqlite, found {len(dbs)}")
    con = sqlite3.connect(dbs[0])
    cols = [r[1] for r in con.execute("pragma table_info(episode_results)")]
    out = {}
    for row in con.execute("select * from episode_results"):
        rec = dict(zip(cols, row))
        steps = [json.loads(f) for (f,) in con.execute(
            "select fields from step_rows where sid=? and eid=? order by step_id", (rec["sid"], rec["eid"]))]
        key = (rec.get("task_idx", rec.get("task")), rec["episode_id"])
        out[key] = {"steps": rec["steps"], "status": rec["status"], "rows": steps}
    return out


def main(a_dir, b_dir):
    a, b = load(a_dir), load(b_dir)
    identical = 0
    for key in sorted(set(a) | set(b), key=str):
        if key not in a or key not in b:
            print(f"{key}: only in {'A' if key in a else 'B'}")
            continue
        ra, rb = a[key]["rows"], b[key]["rows"]
        first = next((i for i in range(min(len(ra), len(rb)))
                      if ra[i].get("action") != rb[i].get("action") or ra[i].get("state") != rb[i].get("state")), None)
        if first is None and len(ra) != len(rb):
            first = min(len(ra), len(rb))
        same = first is None
        identical += same
        print(f"{key}: A {a[key]['status']}/{a[key]['steps']} steps, B {b[key]['status']}/{b[key]['steps']} steps, "
              f"{'IDENTICAL' if same else f'first difference at step {first}'}")
    print(f"identical episodes: {identical}/{len(set(a) & set(b))}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
