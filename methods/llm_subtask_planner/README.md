# llm-subtask-planner (tf_variant of Inner Monologue, arXiv:2207.05608v1)

A frozen LLM splits the instruction into at most 4 subtasks; the frozen policy executes each subtask as its
prompt (whole policy decisions via `ctx.decide(instruction=subtask)`); every 96 steps a frozen VLM looks at
one public camera image and answers whether the subtask is done. After the plan, the episode continues natively.

Substitutions (why `tf_variant`): NRP `qwen3-vl` replaces InstructGPT planner and success detector; the
policy's own prompt replaces the skill library; success detection uses one image, no scene descriptor.

Null arm (`shadow`): the controller runs and pays every LLM call; executed actions are native.

```bash
hs verify llm-subtask-planner                       # tier 1, scripted LLM, no network
NRP_API_KEY=... hs eval llm-subtask-planner --config libero_spatial.yaml \
   --policy-url ws://localhost:8101 --contract libero --profile pi05-libero-stock --out runs/planner
```
