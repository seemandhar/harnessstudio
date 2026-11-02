# hs-bcp: H1-BCP Boundary Clause Prompt (rtcfpp-20261101, round 8)

This package is development-only. It is a HarnessStudio v2 controller for openpi `pi0_fast_libero` on vla-eval
LIBERO-10 (z-score server, greedy FAST, 5 of 10 rows per call, horizon 520). The hypothesis is in
`runs/rtcfpp-20261101/ideation/r8-task.json` (H1-BCP). Approval covers the K1 shadow kill test only, under
conditions K1-c1..K1-c6, in `runs/rtcfpp-20261101/reviews/r8-review.json`. K2 needs a K1 pass and the K2-c1 review.

| Entry | Mode | Use |
|---|---|---|
| `bcp` | `shadow_probe` | **K1.** Run only as `hs eval bcp --arms null`. Executes native actions. For 10 decisions after the first custody boundary it also decodes the clause prompt and the surface control (instruction + `" ."`), logs both and discards them. |
| `bcp-switch` | `switch` | K2 candidate. After the boundary, every decision uses the clause. Not run now. |
| `bcp-switch-dot` | `switch_control` | K2 churn control. After the boundary, every decision uses the instruction + `" ."`. Not run now. |

The rules are frozen in `hs_bcp/rules.py`:
- **Clause split.** Copied verbatim from the scientist's `boundary_exposure.py`. It splits at `' and VERB'` for VERB in put, close, open, turn, pick, place, push. A clause-2 `it` resolves to the last in/on/into object of clause 1. Object pronouns are refused. The rule splits t2, t3, t4, t6 and t9; the other tasks get no split, and the controller returns at once.
- **Custody boundary.** The first run of executed gripper-close rows (action[6] > 0) that is at least 15 steps long, whose finger-width median over rows [a+10, b), rounded to 4 decimals, is at least 0.003 m, and that ends with an executed open command at row b.

**K1 must run in the null arm.** In the candidate arm, every extra `ctx.policy` call advances the framework's decision
counter, which would shift later native CRN seeds. In the shadow (null) arm, every controller call goes to its own
policy session, and the framework executes the native action stream unchanged. A shadow controller does not see the
executed actions. It therefore replays the native decode once per decision (greedy decoding gives identical bytes on a
deterministic server) and logs the replay's fingerprint. The K1 checkers bind each replay to the framework's `native`
event at the same step, and re-derive the boundary from the vla-eval step files.

Checks: `tests/test_bcp.py` (15 tests: the rule on all 10 strings, scripted custody streams, the offline-rule
equivalence, measures, fake-stack stream isolation, executed-row visibility, seed alignment in the switch mode), and
`hs verify bcp|bcp-switch|bcp-switch-dot --tier 0` and tier 1. Run outputs and the manifest are in
`runs/rtcfpp-20261101/implementation/bcp/`.
