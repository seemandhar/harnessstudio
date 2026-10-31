"""Differential check: hs_fast_harness's memory / scheduler / recovery vs the original fast_harness code.

usage: python check_port.py /path/to/fast-harness [N_CASES]
Drives both implementations with the same randomized operation sequences (memory updates, stage matching,
review-interval decisions under every gating flag, recovery state transitions) and requires identical outputs.
"""
from __future__ import annotations

import math
import random
import sys
from dataclasses import asdict


def main(repo: str, cases: int = 300) -> int:
    sys.path.insert(0, repo)
    import fast_harness.memory as om
    import fast_harness.recovery as orc
    import fast_harness.scheduler as osch
    import fast_harness.types as ot
    import hs_fast_harness.memory as nm
    import hs_fast_harness.scheduler as nsch

    checks = mismatches = 0

    def same(a, b, what):
        nonlocal checks, mismatches
        checks += 1
        if a != b and not (isinstance(a, float) and isinstance(b, float) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)):
            mismatches += 1
            if mismatches <= 10:
                print(f"MISMATCH {what}: original={a!r} port={b!r}")

    for case in range(cases):
        rng = random.Random(case)
        dim = 12
        centers = {s: [rng.random() for _ in range(dim)] for s in ("approach", "grasp", "place")}
        decay = rng.choice([1.0, 0.98, 0.9])
        old = om.Memory(":memory:", "ns", decay=decay, examples_per_stage=rng.choice([4, 128]))
        new = nm.Memory("ns", decay=decay, examples_per_stage=old.examples_per_stage)
        cfg_kw = dict(contact_gating=rng.random() < .5, commit_gating=rng.random() < .5,
                      persist_error_review=rng.random() < .5, relax_known_stage_change=rng.random() < .3,
                      gate_stage_change_on_contact_only=rng.random() < .3, confidence_cap=rng.choice([1.0, 2.0]),
                      audit_probability=rng.choice([0.0, 0.05, 0.3]), seed=case, commit_cap=rng.choice([6, 12]))
        osc, nsc = osch.Scheduler(osch.Config(**cfg_kw)), nsch.Scheduler(nsch.Config(**cfg_kw))

        def feat(stage):
            return tuple(min(1, max(0, c + rng.gauss(0, rng.choice([0.01, 0.05, 0.2])))) for c in centers[stage])

        for step in range(60):
            stage = rng.choice(list(centers))
            f = feat(stage)
            op = rng.random()
            if op < 0.3:
                old.learn_stage(stage, f)
                new.learn_stage(stage, f)
            elif op < 0.65:
                kw = dict(episode=f"e{step % 5}", step=step // 2, kind=rng.choice(["execution", "intent"]), stage=stage,
                          features=f, outcome=rng.choice(["ok", "ok", "ok", "error", "unknown"]), evidence=f"ev{step}",
                          student=rng.random() < .8, error_type=rng.choice(["", "slip", "miss"]))
                same(old.record(**kw), new.record(**kw), "record")
            elif op < 0.75:
                n = rng.randint(1, 8)
                old.record_span(stage, n)
                new.record_span(stage, n)
            elif op < 0.8:
                ids = {e["case_id"] for e in old.errors(limit=50)}
                pick = set(rng.sample(sorted(ids), min(len(ids), 2)))
                old.recovered(pick, "rec")
                new.recovered(pick, "rec")
            thr, margin = rng.choice([0.06, 0.15, 0.3]), rng.choice([0.0, 0.01, 0.05])
            mo, mn = old.match(f, thr, margin), new.match(f, thr, margin)
            same((mo.stage, mo.reason), (mn.stage, mn.reason), "match")
            if math.isfinite(mo.distance) or math.isfinite(mn.distance):
                same(round(mo.distance, 12), round(mn.distance, 12), "match.distance")
            same(asdict(old.statistics(stage)), asdict(new.statistics(stage)), "statistics")
            same(old.near_error(stage, f, 0.04), new.near_error(stage, f, 0.04), "near_error")
            same([dict(e) for e in old.errors(stage)], [dict(e) for e in new.errors(stage)], "errors")
            same(old.span(stage), new.span(stage), "span")
            same(old.stages(), new.stages(), "stages")
            # scheduler decision with a random proposal
            traj = [{arm: {"position": [0, 0, 1.0 - 0.02 * i * rng.random()], "gripper_opening_command": rng.choice([1.0, 0.9, 0.2])}
                     for arm in ("left", "right")} for i in range(rng.randint(0, 6))]
            payload = {"student_eef_trajectory": traj, "current_eef": {a: {"gripper_opening_command": 1.0} for a in ("left", "right")}}
            warnings = ("jump",) if rng.random() < .1 else ()
            obs_o = ot.Observation("e", step * 15, f, {})
            obs_n = nsch.Observation("e", step * 15, f, {})
            prop_o = ot.Proposal("r", obs_o, payload, warnings=warnings)
            prop_n = nsch.Proposal("r", obs_n, payload, warnings=warnings)
            kw = dict(last_review_step=None if step == 0 else max(0, step * 15 - rng.choice([15, 45, 90, 200])),
                      chunks_since_review=rng.randint(0, 8), previous_stage=rng.choice([stage, None, "grasp"]),
                      recovery_active=rng.random() < .1, force_review=rng.random() < .1, stalled=rng.random() < .05)
            same(osc.reason(old, mo, obs_o, prop_o, **kw), nsc.reason(new, mn, obs_n, prop_n, **kw), "scheduler.reason")
            same(osch.interval(old.statistics(stage), osch.Config(**cfg_kw)),
                 nsch.interval(new.statistics(stage), nsch.Config(**cfg_kw)), "interval")

        # recovery state machine
        ro, rn = orc.Recovery(rng.choice([2, 8]), rng.choice([6, 24])), nsch.Recovery(0, 0)
        rn.attempt_limit, rn.total_limit = ro.attempt_limit, ro.total_limit
        cps_o = [ot.Checkpoint(f"e:{i}", ot.Observation("e", i * 10, (0.0,), {}), "s", f"g{i}", ()) for i in range(rng.randint(0, 4))]
        cps_n = [nsch.Checkpoint(c.id, nsch.Observation("e", c.observation.step, (0.0,), {}), "s", c.goal, ()) for c in cps_o]
        for t in range(40):
            r = rng.random()
            if r < 0.15 and not ro.active:
                ro.start(50 + t, "goal")
                rn.start(50 + t, "goal")
            elif r < 0.45:
                ro.executed()
                rn.executed()
            else:
                ids = [c.id for c in cps_o] + [None, "bogus"]
                rev = dict(stage="s", last_outcome=rng.choice(["ok", "error", "unknown"]), error_type="", evidence="",
                           response=None, recovery_status=rng.choice(["continue", "succeeded", "failed"]),
                           selected_checkpoint_id=rng.choice(ids))
                res = []
                for machine, cps, review in ((ro, cps_o, ot.Review(**rev)), (rn, cps_n, ot.Review(**rev))):
                    try:
                        res.append(("ok", machine.update(review, cps)))
                    except ValueError as exc:
                        res.append(("ValueError", str(exc)))
                same(res[0], res[1], "recovery.update")
            same(ro.context(), rn.context(), "recovery.context")
            same(tuple(c.id for c in ro.candidates(cps_o)), tuple(c.id for c in rn.candidates(cps_n)), "recovery.candidates")
        # artifact round trip preserves everything
        back = nm.Memory.from_json(new.to_json())
        same(new.to_json(), back.to_json(), "artifact round trip")
    print(f"{cases} cases, {checks} checks, {mismatches} mismatches")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300))
