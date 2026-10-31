# subtask-sequencer

Controller method. It prompts the frozen policy with one step of a composite instruction at a time. It is a
training-free variant (`tf_variant`) of Inner Monologue (Huang et al., CoRL 2022, arXiv:2207.05608v1), with no
LLM and no extra policy inference.

**Why.** Plain GR00T N1.5 on RoboCasa365 Composite-Unseen often finishes the first step of a multi-step
instruction and then stalls. A reactive policy sees the same full instruction at every step, so it has no
notion of which part is already done.

## Mechanism

1. `split_steps` splits the instruction into self-contained imperative steps using fixed text rules. Cuts happen
   at sentence ends, at `then`, and at `, and VERB` / `, VERB` when the clause starts with a known verb and does
   not refer back with a pronoun. If nothing separates cleanly, it returns `[instruction]`.
2. Each step runs as whole DecisionLoop decisions with that step as the prompt: `await ctx.decide(instruction=step)`.
3. The controller moves to the next step once the public end-effector position stalls. A stall means the step has
   run for at least 96 steps and the end effector stayed within 1 cm over the last 48 steps (checked at decision
   boundaries). The last step runs until the episode ends.
4. If the benchmark contract exposes no `eef_position` (`ctx.embodiment.eef_position(obs) is None`), every step
   except the last gets a fixed budget: `ctx.remaining // n_steps` when control starts, or
   `fallback_subtask_steps` if that is set. This is traced as an `eef_unavailable` event.
5. A single-step instruction (for example the fake benchmark's `reach the goal of fake_task_N`) makes `run`
   return immediately. The episode then finishes natively with the original instruction, exactly like the plain
   policy.

Trace events: `subtask_plan` (the steps), `subtask_start` (index, prompt), `subtask_advance` (index, rule, steps
on the subtask), `eef_unavailable`. `ctx.memory` holds `steps`, `advance_rule` and `advanced_at`.

## Substitutions (tf_variant)

| Paper | Here |
|---|---|
| LLM planner | fixed text rules split the instruction into imperative steps |
| success-detector feedback | public end-effector stall after a minimum dwell |
| skill library | the frozen robot policy prompted with the current step |
| (stall signal, when the benchmark has no eef position) | fixed per-subtask step budget |

## Arms and budget

`null: shadow`. In the null arm the controller runs, but `ctx.decide` executes native actions with the original
instruction, so the null reproduces the baseline at the baseline's cost. Requires `[act, chunk]`; it does not
need seeded sampling.

`budgets.policy_calls: 2000`. The method makes one policy sample per decision, so its calls equal its decisions,
which is at most ceil(horizon / executed chunk length). RoboCasa365's advisory horizon is 3000 steps; GR00T
executes 16 steps per decision, so an episode needs at most 188 calls, and 2000 covers any executed chunk of 2 or
more steps. LIBERO (520 steps) fits even at one step per decision. Reaching the cap ends control, and the episode
finishes natively.

## Run

```bash
uv pip install --python .venv/bin/python -e methods/subtask_sequencer
.venv/bin/hs verify subtask-sequencer --tier 0
.venv/bin/hs verify subtask-sequencer            # tier 1, fake stack (fake-t0)
```

Parameters (`ModuleRef` params of the controller): `min_dwell_steps=96`, `stall_window_steps=48`,
`stall_tolerance_m=0.01`, `fallback_subtask_steps=None`.

## v1 results (HarnessStudio v1, GR00T N1.5, Composite-Unseen, 16 tasks × trials 0–1)

The split ran on 9 of 16 tasks, into 2–3 steps each. (The v1 README says 10, but v1's own screen records show 9.)
Successes were 1/32 for the candidate vs 0/32 for the plain policy: one paired win, p = 0.5. That is not a
measured gain. The inference ratio was 1.009.

## Changelog

- **2.0.0** (card version 2): ported to v2.
  - Async controller on `EpisodeContext`; v1's `tools.act(step)` became `ctx.decide(instruction=step)`.
  - The stall signal now comes from the contract's `eef_position` instead of v1's `proprio` capability. When a
    benchmark exposes none, each subtask gets a fixed step budget.
  - The stall window is anchored on the last decision boundary at or before `latest - window`. This is identical
    to v1 when the window is a multiple of the executed chunk (GR00T: 48 = 3 × 16). v1 could never advance when it
    is not, for example with Pi0.5's 10-step chunks.
  - `min_act_steps` was dropped (v2 has no `act(steps=)`).
  - No advance is recorded on the decision that ends the episode.
- **1** HarnessStudio v1 `harnessstudio.methods.subtask_sequencer`.
