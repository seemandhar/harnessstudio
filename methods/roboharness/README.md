# roboharness

RoboHarness (arXiv:2603.24060v3; earlier title SOMA) for HarnessStudio v2, ported from the authors' code
(github.com/LZY-1021/RoboHarness, MIT, commit 716defc).

A frozen VLA is wrapped by:

- **Dual-Memory RAG**: at episode start, the first agent-view frame and the instruction are embedded (CLIP
  ViT-L/14 0.6 + MiniLM 0.3 + task hash 0.1, 1168-d) and the top-1 success and top-2 failure experiences
  (cosine >= 0.4) are retrieved; the failure diagnoses go into the orchestrator prompt.
- **Orchestrator (VLM)**: every 10 steps it reads the frame, the task and the retrieved diagnoses and returns a
  tool chain. Visual tools edit the agent-view frame through SAM3 (`visual_overlay`: green highlight;
  `remove_distractor`: mask + Telea inpainting); a VLM verifier checks the edit and the orchestrator re-plans on
  rejection. It can also rewrite the instruction and trigger control flow.
- **Control flow**: Encore (replay the negated, averaged accumulated action for 40 steps, then 20 open-gripper
  hold steps), key-step timeout retry (50 + 25), subtask queue.
- **Memory**: a development pass on standard LIBERO-Spatial and LIBERO-Object (10 episodes per task) is
  diagnosed by the VLM (failure root cause / success summary from 5 frames) and frozen as `memory.json`.

## Fidelity

The port follows the released evaluation code, not the paper text, wherever they differ; `paper.yaml` lists
every substitution. `scripts/check_parity.py` runs the authors' modules against the port on randomized inputs
(control-flow state machines, `process_frame` with scripted VLM replies and an emulated SAM3 server including
every prompt message, memory retrieval, JSON/box parsers): 0 differences.

Main points:

| Item | Released code / paper | Here |
|---|---|---|
| Orchestration loop | `MAX_RETRIES = 0`: loop never runs, eval edits nothing | `max_attempts = 3` (documented retry behaviour) |
| VLM | Qwen3-VL-32B-Instruct | NRP `qwen3-vl` (Qwen3.8-Flash-Next), thinking off, original temperatures/token caps |
| Policy | LeRobot pi05 (checkpoint not released) | openpi `pi05_libero`, seeded, 10 actions per chunk |
| Memory | grows online from evaluated episodes (env success label) | frozen development artifact; evaluation never writes memory |
| Blank-inference attention check, Stage-2 consolidation | paper only | not implemented (not in released code) |
| Texture / background replacement | needs memory assets the logger never writes | same: never fires |
| Task chaining (LIBERO-RoboHarness) | hard-coded chain + timers, VLM off | not ported (benchmark not ported yet) |

## Run

Tool server (SAM3 + encoder; any env with torch, transformers, opencv, `sam3`):

```bash
GPU=4 PORT=8131 /home/sejain/research/cvpr27/hs2-gpu/roboharness/start_tool_server.sh
```

Development memory (4 shards in parallel, then one merged artifact):

```bash
hs prepare roboharness --config methods/roboharness/configs/dev_libero_spatial_a.yaml --contract libero \
  --policy-url ws://127.0.0.1:8106 --profile pi05-libero-hs --out DEV/spatial_a      # x4 shards
python methods/roboharness/scripts/merge_dev.py PREP DEV/spatial_a DEV/spatial_b DEV/object_a DEV/object_b
```

Evaluation (LIBERO-Pro, paper Table III cells):

```bash
hs eval roboharness --config methods/roboharness/configs/libero_pro_spatial_task.yaml --contract libero_pro \
  --policy-url ws://127.0.0.1:8106 --profile pi05-libero-hs --prepared PREP --arms baseline,candidate \
  --action-timeout 1800 --out OUT
```

`HS_ROBOHARNESS_TOOL_URL` overrides the tool server address.
