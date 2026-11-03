# EXT simpler-seeded (proposal, 2026-11-03): the PCD authors' seeded SIMPLER protocol, and an OpenVLA server mode that matches their inputs

> **Status (2026-11-03): APPROVED WITH EDITS, BUILT, NOT MERGED.** The independent `rmh_critic` review
> (`runs/pcd-port-20261103/reports/critic-simpler-ext.review.json`, verdict approve_with_edits) required S-1..S-8. They are
> applied below, each marked **[S#]**; section 10 lists every required edit and advisory with pointers and supersedes
> any conflicting text above. Built in worktree `../hs2-ext-simpler-seeded`, branch `ext2/simpler-seeded`, stacked on
> `ext2/openvla-contrast`: cut from `1fc5b10`, then REBASED onto its new head `da817ba` (tests-only openvla-contrast
> change, review E1; coordinator request 2026-11-03 ~12:25Z); commits `e088fd2`, `8feef86` and the closing commit; build report `runs/pcd-port-20261103/reports/simpler_build.md`
> (CPU tests, regress before/after, integrity, the evaluator's GPU plan). GPU checks, diff review and merge pending.

- **Author:** converter tasks `conv-simpler-ext` (proposal) and `conv-simpler-build` (edits + build), run `pcd-port-20261103`. The agent ref is in the supervisor ledger
  `runs/pcd-port-20261103/control/research/supervision`.
- **Inputs:**
  - `runs/pcd-port-20261103/reports/simpler_readiness.md` (sha256 `68776535...`). This proposal cites its sections as
    **[R§n]**.
  - `runs/pcd-port-20261103/SETTING.md` v2 (sha256 `0049cd2d...`): the "Protocol semantics" table, items (a)-(f).
  - The openvla-contrast critic ruling `runs/pcd-port-20261103/reports/critic-ext-proposal.review.json` (sha256
    `4a60a6a3...`). Its relevant ruling (`q5`): the SIMPLER contract and benchmark are "a separate extension,
    sequenced after this one"; any server option it needs "needs its own native-equivalence check"; "keep
    `openvla_logits_hs.py` benchmark-agnostic".
- **Paper and code:**
  - PCD, arXiv:2505.13255, Table 1 (any-step success) and Table 9 (max-step success).
  - Code `Koorye/PCD@31b027e`, read only, never installed or put on `sys.path`. It has no licence, so everything
    here is re-derived and cited by file:line.
- **Base:**
  - `harnessstudio` `main` `e4ae8b8` for the benchmark and contract;
  - branch `ext2/openvla-contrast` @ `da817ba` (worktree `../hs2-ext-openvla-contrast`, not merged) for the server
    subclass. **[S-7e]** The branch is cut from `ext2/openvla-contrast` (`1fc5b10`, rebased onto `da817ba`), not from `main`: a disclosed deviation from EVOLUTION
    step 3 (`-b ext2/EXT_ID main`), because the server subclass imports the unmerged `openvla_logits_hs.py`. The diff
    review covers only the simpler-seeded commits (`git diff da817ba..ext2/simpler-seeded`).
- **Paths:**
  - Paths without a prefix are relative to `harnessstudio/harnessstudio/`.
  - `vla_eval/...` means `/home/sejain/research/cvpr27/vla-eval-v0.8.0/src/vla_eval/...` (tag v0.8.0, `41bd8a44`). It
    is byte-identical to the copy installed in `harnessstudio/.venv` and to the PyPI wheel `vla_eval-0.8.0` (diffed,
    `_version.py` excluded).
  - `[A]` means the authors' repo.

**Short version.**
- **Gap.** vla-eval 0.8.0 ships a SIMPLER benchmark (`vla_eval/benchmarks/simpler/benchmark.py:48`) and an image.
  With its OpenVLA server, it does not run the authors' protocol: every Pick Coke and Close Drawer episode is the
  **same scene**. The OpenVLA server forces `seed=0` (`vla_eval/model_servers/openvla.py:75-76`), and the VM reset
  passes only `episode_id` (`benchmark.py:264-267`), which those environments ignore. Measured: identical obs0 hashes
  for episodes 0-3 [R§1.3].
  - **Other differences:** prompt, resize, gripper, rotation and attention all differ from the authors' code [R§1.2].
- **Benchmark half (this extension):** a HarnessStudio-owned benchmark subclass `SimplerSeededBenchmark`, plus a
  contract YAML. It needs no change to vla-eval and no change to any existing HarnessStudio file.
  - **Seeds:** one case per seed, with the seed in the case key. Each case is a fresh env reset with
    `env.reset(seed=s)` and no options.
  - **Post-processing:** the authors' rotation and Google relative sticky gripper, applied benchmark-side to the raw
    OpenVLA action.
  - **Success:** any-step and max-step success are both recorded; there is no early stop.
- **Server half (on top of `ext2/openvla-contrast`):** a NEW file, the subclass
  `policies/openvla_simpler_hs.py:HarnessOpenVLASimplerServer(HarnessOpenVLALogitsServer)`. It fixes the preprocessing
  mode `simpler_pcd_authors`:
  - raw instruction (+29871);
  - cv2 INTER_AREA resize;
  - eager attention;
  - raw un-normalized gripper.
  - It has its own equivalence checks.
  - `openvla_hs.py` and `openvla_logits_hs.py` stay **byte-identical**: one is Run X's pinned server, the other stays
    benchmark-agnostic per the ruling.
- **Server env.** It lives under `runs/pcd-port-20261103/venvs/`, never under `harnessstudio/` **[S-7a]** (the supervising
  session's 11:04Z rule "nothing compiled or binary under harnessstudio/", recorded in the COORDINATION.md log and
  PITFALLS P-20261103-06; it is not one of COORDINATION's binding Rules 1-5).
  - **Primary:** a pinned env with the authors' numerics, transformers 4.44.2 / torch 2.3.1 / eager. It is already built
    and CPU-checked: HS v2 server code imports, and the processor output is bitwise equal to the current stack's.
  - **Fallback** (SETTING Addendum 2 E-3): a RUN-LOCAL copy of the current env (transformers 4.40.1 / torch 2.10) plus
    opencv, at `runs/pcd-port-20261103/venvs/openvla-fallback`, built; used only for a failure of the numerics env
    itself, never for a code defect that fails in both envs (e.g. S-1, which is repaired).
- **CPU evidence** (prototype):
  - 13/13 fake-env unit tests pass, on the image's python 3.11 / numpy 1.24.4 / transforms3d 0.4.2 with vla-eval 0.8.0
    mounted.
  - Post-processing is **bitwise equal to the authors' own `OpenVLAInference.step` and
    `OpenVLAContrastInference.postprocess_actions`** over 22,600 steps, with 1,287 sticky onsets. Three sabotage
    controls are all detected.
  - A real-env CPU smoke gives obs0 equal to the readiness probe.
- **One framework-level numeric difference cannot be removed without touching core files.**
  - HS v2's protocol casts proposal actions to float32 (`policy_ext/protocol.py:61`), and so does vla-eval's
    `PredictModelServer` (`vla_eval/model_servers/predict.py:241`). The authors stay in float64.
  - Measured effect on the env action: at most 6e-8 absolute, and zero sticky-state differences over 11,300 steps.
  - It is disclosed and checked on the GPU (B7). Section 9 explains why no core change is proposed.

## 1. The mechanism needed (the authors' protocol, SETTING.md items a-f)

The PCD-vs-baseline gain in Table 1 is measured under the authors' own evaluation loop. A port that changes the
case distribution, inputs or action conventions cannot be compared with 16.8 / 18.0 / 47.3 [R§0]. The loop:

| item | authors' code ([A] = `/home/sejain/research/cvpr27/PCD-upstream/repo`) |
|---|---|
| (d) reset | `simpler_env.make(task)` = `gym.make(env_id, obs_mode="rgbd", prepackaged_config=True)` (`simpler_env/__init__.py:167-173`); `env.reset(seed=episode)` with no options (`parallel_inference.py:174`); one env per worker process, reused (`:137,270-290`) |
| (a) prompt | `prompt = task_description`: the raw instruction, no template (`simpler_env/policies/openvla/openvla_model.py:102`), then token 29871 appended if absent (`:108-109`) |
| (b) image | full-resolution `overhead_camera` RGB (`parallel_inference.py:12-23`), `cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)` (`openvla_model.py:173-175`), no JPEG, no crop |
| model load | `AutoModelForVision2Seq.from_pretrained(path, attn_implementation="eager", torch_dtype=bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)` (`openvla_model.py:43-50`); transformers 4.44.2 (`requirements_full_install.txt:23`), torch 2.3.1 |
| decode | greedy `generate(max_new_tokens=7, output_scores=True)`, then `scores.argmax(-1)` (`openvla_model.py:114-124`); de-tokenize and q01/q99 un-normalize with `fractal20220817_data` in float64 (`:209-218`; gripper mask False, so the gripper is the raw bin centre) |
| (c) actions | `world_vector * 1.0`; `euler2axangle(roll, pitch, yaw)` -> axis * angle (`:136-141`). Google gripper (`:143-164`): `relative = prev_open - cur_open` (0 at the first step); sticky for 15 steps once `abs(relative) > 0.5`. Env action = `concat(world_vector, rot_axangle, gripper)` (`parallel_inference.py:215-217`). The PCD arm uses the identical `postprocess_actions` (`contrast_policies/openvla_contrast.py:61-104`) |
| (e) termination | the loop runs until `truncated` (TimeLimit: coke 80, close/open drawer 113, move near 80) or until the policy terminates, which OpenVLA never does (`terminate_episode = 0`, `openvla_model.py:169`). Table 1 = any(`info["success"]`) (`utils.py:42-64`); Table 9 = the last step's (`utils.py:80-87`) |
| (f) env identity | vendored ManiSkill2_real2sim = upstream `ef7a4d4` plus the authors' edits, with no behavioural difference on these tasks under default lighting; SimplerEnv `make()` is equivalent; 274/275 data blobs are identical; the overlay PNGs cannot be verified [R§3] |

## 2. Why the existing v2 profiles and benchmarks cannot express it

1. **There is no SIMPLER contract or benchmark in HarnessStudio.**
   - `contracts/` has `fake`, `libero`, `libero_pro`, `libero_rgbd`, `robocasa365` and `robodojo`.
   - `benchmarks/` has only `libero_rgbd.py`.
   - `load_contract` accepts any YAML path (`contracts/__init__.py:78-81`), so the contract alone could be
     method-local. The protocol, however, lives in the benchmark.
2. **vla-eval's `SimplerEnvBenchmark` cannot be configured into the authors' reset:**
   - VM mode always passes `options={"obj_init_options": {"episode_id": idx}}` when `deterministic_episodes=True`
     (`benchmark.py:264-267`), and a single fixed `seed` for every episode (`:218-219`).
   - `deterministic_episodes=False` with `seed=None` gives an unseeded, irreproducible reset.
   - Nothing maps the episode to `reset(seed=s)`, and nothing supports a non-contiguous seed list such as the
     predeclared drawer seeds. Episodes are always `range(episodes_per_task)` (`vla_eval/orchestrator.py:348-349`).
   - The OpenVLA server's `observation_params = {"seed": 0}` (`model_servers/openvla.py:75-76`) is merged into any
     constructor that accepts `seed` (`orchestrator.py:90-113`). HarnessStudio forwards the policy server's
     observation params unchanged (`server/harness_server.py:120-121`), so HS runs inherit the collapse.
3. **Success and termination:**
   - `success_mode` is one of `truncation` (final step), `accumulate` (any step) or `early_stop`
     (`benchmark.py:383-394`). No mode records both any-step and max-step.
   - `get_step_result` returns `{"success"}` only. HS records keep only `metrics` (`server/harness_server.py:315-322`),
     so Table 9 cannot be recovered from an HS record.
4. **Actions:**
   - The benchmark passes rotation through unconverted (`benchmark.py:276-281`).
   - Its gripper is either `binary` (`:298`) or an *absolute* sticky map (`:282-296`); the authors use a *relative*
     map, prev - cur.
   - The server already maps the gripper to `-sign(2g-1)` (`model_servers/openvla.py:132-134`;
     `policies/openvla_hs.py:250`), which destroys the continuous value the relative map needs.
   - No combination of existing parameters gives the authors' env action.
5. **Policy inputs:**
   - Every OpenVLA path in HS v2 uses the template prompt `"In: What action should the robot take to {t}?\nOut:"`:
     - native: `model_servers/openvla.py:123`;
     - sampler: `policies/openvla_hs.py:173`;
     - logits: `ext2/openvla-contrast policies/openvla_logits_hs.py:154`.
   - They resize with PIL LANCZOS (`model_servers/openvla.py:103-104`).
   - They load with the default attention. Measured with the same classmethod `from_pretrained` uses, the default
     resolves to **SDPA** (`LlamaSdpaAttention`) on both transformers 4.40.1 and 4.44.2; `attn_implementation="eager"`
     selects `LlamaAttention` (`runs/pcd-port-20261103/simpler_ext/results/attn_check_{current,numerics}.json`).
   - The server args offer `jpeg_roundtrip` and `center_crop` only (`model_servers/openvla.py:44-59`). There is no
     prompt, resize or attention option.
6. **Docker isolation is not automatic here.** The host's default Docker runtime is `nvidia`, and the image bakes in
   `NVIDIA_VISIBLE_DEVICES=all` [R§2.1].
   - vla-eval adds `--gpus device=N` (`vla_eval/docker_resources.py:126-147`) and, only for `gpus: none`, an
     `NVIDIA_VISIBLE_DEVICES=void` override.
   - A GPU-rendered SIMPLER run therefore needs an explicit, checked single-GPU pin.

## 3. The smallest change

### 3.1 NEW `benchmarks/simpler_seeded.py`: `SimplerSeededBenchmark(SimplerEnvBenchmark)`

This is the prototype `runs/pcd-port-20261103/simpler_ext/pcd_simpler_seeded/simpler_seeded.py` (sha256
`26b67f05ebe48cb3...`), moved into `benchmarks/` with the review additions ([S-5b] action spec, [S-6] isolation, A-2,
A-3). **[S-8a]** `benchmarks/__init__.py` requires that importing the package needs only vla-eval and numpy:
transforms3d is imported lazily inside `GoogleAuthorsPostprocess.__call__`, so this holds. The package docstring
limited `benchmarks/` to *sensor* subclasses (spec 16.11); the build updates it (declared change, file list 3.8) to
admit *protocol* subclasses (reset / action post-processing / metrics), still with no simulator state, reward or success
to the policy or a method, and adds the same sentence to spec decision 16.11.

- **Constructor:** `__init__(self, task_name, seeds, action_post="openvla_google_authors", gpu_uuid=None)`, with **no
  `**kwargs`**. `gpu_uuid` is required for `render: gpu` [S-6].
  - `_accepted_init_params` stops at the first class without `**kwargs` (`orchestrator.py:77-87`), so the server's
    `seed` is dropped with vla-eval's warning. This is tested.
  - It calls `super().__init__(..., success_mode="accumulate", seed=None, deterministic_episodes=False)`.
  - It refuses tasks outside `{google_robot_pick_coke_can, google_robot_move_near, google_robot_open_drawer,
    google_robot_close_drawer}`. WidowX and Apple Drawer need their own rules: the camera, the binarized gripper, and
    the sub-task advance.
  - It refuses empty, duplicate or negative seed lists.
- **`get_tasks()`:** `[{"name": task_name, "task_name": task_name, "task_id": f"seed{s}"} for s in seeds]`. Run it with
  `episodes_per_task: 1`.
- **`reset(task)`:**
  1. Parse `s` from `task_id`. Refuse a seed outside the list and any `episode_idx != 0`.
  2. Reset the per-episode post-processing state and the success trace.
  3. Close the old env, then `simpler_env.make(task_name)` with **no kwargs**. The image's `make()` mutates its
     `ENVIRONMENT_MAP` kwargs in place [R§3].
  4. `env.reset(seed=s)` with no options.
  5. Read the instruction and record the first frame.
  - Fresh env versus the authors' reused env: equal obs0 in 15/15 measured cases [R§7].
- **`step(action)`:**
  1. Read `raw = np.asarray(actions, float64).reshape(-1)` and refuse any shape other than (7,).
  2. Compute `env_action = GoogleAuthorsPostprocess()(raw)`, a literal re-derivation of `openvla_model.py:128-169`
     that keeps the authors' array shapes and dtypes; for example, the first-step gripper is `np.array([0])`.
  3. `env.step(env_action)`, and append `bool(terminated)` to the trace.
  4. Re-read the instruction. If it changed, reset the gripper state, mirroring `openvla_model.py:94-96`; supported
     tasks never change it.
- **`check_done`:** `info["truncated"]` only, never success.
- **`get_step_result`:** `{"success": any(trace), "final_success": trace[-1], "first_success_step": i | -1,
  "steps": len(trace)}`.
- **`get_metadata`:** the parent's metadata plus `protocol: "pcd_authors_31b027e_seeded"`, `seeds`, `seeds_sha256`,
  `action_post` and `official_horizon`.
- **Observation:** the inherited `make_obs` (`benchmark.py:331-342`):
  - `images.primary` = overhead_camera RGB uint8 512x640x3 (measured);
  - `task_description`.

### 3.2 NEW `contracts/simpler_seeded.yaml`

Prototype `simpler_ext/contracts/simpler_seeded.yaml` (sha256 `d644d7e3...`):
- `public_keys: [images, task_description]`. `task_name` is not public.
- `action_dim: 7`, `control_mode: eef_delta`, layout position [0,3] / rotation [3,6] / gripper [6,7].
- `gripper_open: 1.0`, `gripper_close: 0.0`. This is the raw OpenVLA convention before the benchmark's relative map.
- `horizons`: coke 80, move near 80, open/close drawer 113. `default_horizon: 113`.
- Checked: `load_contract` loads it; `horizon()` reads `task["name"]`, so the `#seed` suffix of the case key does not
  affect it (`contracts/__init__.py:61-75`).

### 3.3 Case model, hashing, CRN and pairing

- **Case key:** `(simpler_seeded[@case_namespace], f"{task}#seed{s}", 0)` (`server/harness_server.py:131-140`). The seed
  is explicit in every record and every pairing key (`eval/pairing.py:8-9`). Methods never see it: case, seed and arm
  stay evaluator-side (INVARIANTS).
- **Case-list hash:** `sha256(json.dumps(sorted([[task, seed], ...])))`.
  - Setting A's predeclared 64 cases give `cf405fe2c90017d6fe196610cadf659565f90a11c598264697403274bf52c717`
    (`runs/pcd-port-20261103/readiness/case_list_cpu_obs0.json`).
  - The run manifest also records `seeds_sha256` per benchmark (from `get_metadata`), the image digest, the sha256 of
    `simpler_seeded.py`, and the GPU obs0 hashes (B1).
- **CRN:** `crn_seed(namespace, benchmark, task#seedS, 0, decision, sample)` (`context/episode.py:255-257`) is distinct
  per seed and identical across arms. Greedy OpenVLA ignores it; it fixes only harness-side randomness. The env is
  deterministic in s.
- **Arms interleaved per seed** (SETTING). Run one vla-eval config per (seed block, arm) on the same policy server, then
  pair the results with `hs pair --arm A=DIR ...` (`cli.py:190-206`). This needs no framework change.
- **M/R/X attribution** uses the per-step obs and action hashes. GPU SAPIEN rendering could branch between runs; that
  shows up as R, not X. B3 measures it before any candidate episode.

### 3.4 Success and termination

The benchmark keeps the episode running to the TimeLimit. The vla-eval runner breaks only on `is_done` (= truncated)
or at `max_steps` (`vla_eval/runners/sync_runner.py:55-63`).
- **`max_steps`:** set it in the config to the official horizon. That is redundant with the TimeLimit, and the test
  asserts that truncation governs even with a larger `max_steps`.
- **HS pairing** uses `metrics.success`, which is any-step (Table 1).
- **`metrics.final_success`** (Table 9) is read from `records.jsonl` by the run's analysis script. No pairing change is
  needed.

### 3.5 Image pin and the Docker GPU-isolation rule

- **Image:** `ghcr.io/allenai/vla-evaluation-harness/simpler@sha256:f77449671884afee5c997f2b933d0d2db503debe86961d2f6c378d47276d10fe`.
  - Pin it by digest: `:latest` was rebuilt on 2026-11-03.
  - It contains SimplerEnv `4ab7178`, ManiSkill2_real2sim `ef7a4d4`, sapien 2.2.2, python 3.11, numpy 1.24.4 and
    transforms3d 0.4.2.
  - It carries vla-eval **0.9.0**. Mount `vla-eval-v0.8.0/src:/workspace/src:ro` over it; the image's editable path is
    `/workspace/src`. Verified: `vla_eval.__version__ == 0.8.0` inside.
- **Mounts:** the HarnessStudio checkout `:/hs:ro` with `PYTHONPATH=/hs` (the `libero_rgbd` precedent). Use the
  worktree for the GPU checks before the merge and the merged checkout after it. Add `PYTHONDONTWRITEBYTECODE=1`.
- **Rendering: GPU** (supervisor decision 1).
  - Set `docker.gpus: "<one leased GPU>"`, never `all`, and add `-e NVIDIA_VISIBLE_DEVICES=<same index>` in
    `docker.env` as defence in depth.
  - Set `NVIDIA_DRIVER_CAPABILITIES=all`. **[S-6] Pin Vulkan to the NVIDIA ICD:** mount only the host's
    `/usr/share/vulkan/icd.d/nvidia_icd.json` at `/etc/hs-vulkan/nvidia_icd.json:ro` (NOT the whole host `icd.d`, which
    holds `lvp_icd.json` and 7 others; the image itself ships lavapipe) and set
    `VK_ICD_FILENAMES=VK_DRIVER_FILES=/etc/hs-vulkan/nvidia_icd.json`.
  - **[S-6] Per-launch isolation check** (every container = every (seed block, arm) config, not only B2):
    `SimplerSeededBenchmark._check_render_isolation()` runs on every `reset` in `render: gpu` mode and aborts the episode
    (infrastructure) unless `nvidia-smi --query-gpu=uuid` inside the container lists exactly `params.gpu_uuid` (the
    leased GPU's UUID from the host), and both Vulkan loader variables name only an existing `nvidia_icd.json` whose
    content names the NVIDIA driver and not lavapipe. The record (`render_isolation`: UUIDs, `NVIDIA_VISIBLE_DEVICES`,
    ICD) is in the benchmark metadata of every run. The Vulkan physical device SAPIEN selected is additionally read
    from the container log by B2.
- **CPU-only containers** (tests, reset probes): `--runtime runc -e NVIDIA_VISIBLE_DEVICES=void -e
  CUDA_VISIBLE_DEVICES= --network none`.
- **Templates:** `runs/pcd-port-20261103/simpler_ext/build/configs/simpler_seeded_{pick_coke_can,close_drawer}_gpu.yaml`
  (with the [S-6] ICD mount/env and `gpu_uuid`); vla-eval's `load_config` and `DockerConfig` parse both.

### 3.6 Server mode `simpler_pcd_authors`: NEW `policies/openvla_simpler_hs.py`, on top of `ext2/openvla-contrast`

```python
class HarnessOpenVLASimplerServer(HarnessOpenVLALogitsServer):   # mode simpler_pcd_authors, fixed per process
    harness_capabilities = HarnessOpenVLALogitsServer.harness_capabilities - {"decoding_hooks"}
```

- **Why a subclass and not a flag inside `openvla_logits_hs.py`:**
  - the ruling keeps that file benchmark-agnostic;
  - `openvla_hs.py` is Run X's pinned `frozen_sources`;
  - a flag would put SIMPLER-only branches into both, and every LIBERO path would then depend on new code;
  - a separate file leaves both parents byte-identical (sha pins `e7c6a4c3...` and `ddf04264...`, tested).

**Overrides** (as built, `harnessstudio/policies/openvla_simpler_hs.py`). The test
`tests/test_openvla_simpler_server.py` pins the exact set of methods the class defines (the allowlist below), and an
AST check of the module: no call to `predict_action`, exactly one `generate` call whose keywords are `input_ids`,
`pixel_values`, `max_new_tokens`, `do_sample` and `**extra` (only `logits_processor` / `stopping_criteria`), no
`attention_mask` anywhere in it, one `from_pretrained(..., attn_implementation="eager")`, and both cuDNN assignments.

| method | parent (file:line) | declared difference |
|---|---|---|
| `__init__` | `vla_eval/model_servers/openvla.py:44-73` + `policies/openvla_hs.py:99-105` | does NOT call the parent constructor (it would load SDPA): `PredictModelServer.__init__`, then `from_pretrained(local snapshot, attn_implementation="eager", torch_dtype=bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)`; **[S-2]** `torch.backends.cudnn.deterministic = True`, `cudnn.benchmark = False` (the authors' `setup_torch_seed`, `simpler_env/policies/__init__.py:7-15`, run by every `policy.reset`); refuses `jpeg_roundtrip` / `center_crop` and a non-directory model path; HS state as in the parent; optional `action_log` [S-4] |
| `_check_model` (new) | — | **[S-5c]** raises unless `config._attn_implementation == "eager"` AND `type(language_model.model.layers[0].self_attn).__name__ == "LlamaAttention"`, cuDNN deterministic with benchmark off, and the unnorm key exists |
| `_init_log`, `_log_action` (new) | — | **[S-4]** behaviour-free float64 pre-cast log: one JSONL line per decode with `seq`, `(session, episode)` ids (arm-agnostic), path (`native` / `native_seeded` / `logits`), tokens, the float64 action and its exact bytes (`action_f64_hex`). A write failure raises (evidence is never silently lost) |
| `get_observation_params` | `model_servers/openvla.py:75-76` | **[S-5a]** `{}` (no `seed`) |
| `get_action_spec` | `model_servers/openvla.py:78-79` | **[S-5a]** `POSITION_DELTA`, `ROTATION_EULER` (raw roll/pitch/yaw), `GRIPPER_RAW` (raw open-positive bin centre); equals the benchmark's spec, so `check_specs` warns nothing (tested) |
| `_preprocess_image(obs)` | `model_servers/openvla.py:84-116` | [b] the first image, uint8 HxWx3 asserted, `cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)` -> `PIL.Image.fromarray`; no JPEG, no crop |
| `_prompt(obs)` (new) | — | [a] `obs["task_description"]` verbatim |
| `_inputs(obs)` | `policies/openvla_hs.py:168-174` | refused (the sampler / SCALE path keeps the template prompt; samplers are not offered) |
| `_generate(obs, logits_processor=None, stopping_criteria=None)` (new) | [A] `openvla_model.py:105-124`, `openvla_contrast.py:107-123` | **[S-1]** `processor(raw prompt, image).to(device, bf16)`; append 29871 to `input_ids` if absent (the authors' rule, `openvla_model.py:108-109`); `model.generate(input_ids=..., pixel_values=..., max_new_tokens=7, do_sample=False[, logits_processor, stopping_criteria])` with **no attention_mask** (the authors pass none; `generate` then builds a full ones mask). `predict_action(**inputs)` is never called: it would pass the processor's mask, one token shorter than the ids after 29871 (`modeling_prismatic.py:386-390, 512-518`), which crashes the first eager forward. Tokens = the generated ids (= the authors' `scores.argmax`, same tensor under greedy). **[A-7]** fewer than 7 tokens (EOS) raises `SimplerDecodeShort`, an infrastructure event |
| `_actions(tokens)` | `policies/openvla_hs.py:238-251` | [c] float64, no gripper map; the authors' `_decode_actions` formula (`openvla_model.py:209-218`). `_bin_table` is inherited and therefore float64 and consistent with `_actions` |
| `predict(obs, ctx)` | `model_servers/openvla.py:118-135` | `_generate` + `_actions`; logs [S-4]; returns the float64 raw action (vla-eval's `_normalize_result` casts the wire to float32) |
| `sample_chunks` | `policies/openvla_hs.py:125-135`, `openvla_logits_hs.py:84-102` | sampler -> refused; any readout -> the logits parent (which refuses non-`logits` readouts); otherwise the native loop over `_generate` + `_actions` (float64; logged) |
| `_predict_recorded(obs)` | `openvla_logits_hs.py:143-167` | the recorders passed to `_generate`; raw prompt, no mask, float64, logged. The R2 routing and the R3 token/action self-check (`_recorded`, `_self_check`) are inherited unchanged |

- **The wire is still float32.** HS `make_response` casts proposals to float32 (`policy_ext/protocol.py:61`), and
  vla-eval's `PredictModelServer._normalize_result` does the same (`model_servers/predict.py:241`). The server computes
  in float64 like the authors, and the transport rounds.
  - Measured on 11,300 synthetic steps: max abs env-action difference 6e-8 (dims 0-6: 6.5e-9, 7.4e-9, 7.4e-9, 3.0e-8,
    2.0e-8, 1.9e-8, 5.9e-8); sticky decisions never differ
    (`simpler_ext/results/authors_post_parity_image.json`, `float32_transport_effect`).
  - Disclosed; B7 checks its effect on the simulator.
- **NEW profile `profiles/openvla-simpler-logits.yaml`:**
  ```yaml
  capabilities: [act, chunk, seeded_sampling, isolated_streams, logits_readout]
  chunking: harness
  execute_horizon: 1
  ```
  - It drops `decoding_hooks`, because this server refuses samplers.
  - Tier 0 only reports problems, so an extra profile changes no existing method (the openvla-contrast q3 finding).
- **Equivalence checks of the mode** (CPU: S1-S5; GPU: S6-S10, section 7):
  - **S1:** sha256 pins: `openvla_hs.py` == `e7c6a4c3...`, `openvla_logits_hs.py` == `ddf04264...`, unchanged by this
    extension.
  - **S2:** the method allowlist and the AST rules above.
  - **S3:** stub model (`__new__`, never `__init__`, `CUDA_VISIBLE_DEVICES=`, offline scratch HF caches; the
    openvla-contrast R6 pattern) around a tiny REAL eager `LlamaForCausalLM` with OpenVLA's sizes (32064 logits,
    vocab_size 32000, 256 bins) through the real transformers `generate`, in BOTH envs **[S-1]**:
    - the override path decodes under eager attention, no `attention_mask` reaches `generate` (kwargs-capturing stub),
      and the tokens equal an independent no-mask greedy reference;
    - negative control: the parent's `predict_action(**inputs)` path with the processor mask raises the eager size
      mismatch (`... at non-singleton dimension 3`); under SDPA the same call does not raise (why the LIBERO servers
      never hit it);
    - `predict`, the no-sampler `sample_chunks` and `_predict_recorded` all return the same float64 raw action;
    - the bin table equals `_actions` for the R1 boundary tokens (0, 31743-31746, 31999, 32000, 32063), and row 255
      equals row 254;
    - a sampler, other readouts, `_inputs` and non-uint8 images are refused; a short decode raises (A-7); the task
      config is ignored (A-9).
  - **S3b [S-3]:** the real server `_actions` and the `_bin_table` route against the authors' own `_decode_actions`
    (loaded by source) for every token id 0..32063 in every dimension, plus 20,000 mixed rows, bytes + dtype
    (`runs/pcd-port-20261103/simpler_ext/build/checks/decode_vs_authors.py`).
  - **S4:** the processor on the real tokenizer only (no weights). `"pick coke can"` -> `[1, 5839, 1302, 446, 508]`
    (+29871 by `predict_action`), equal in both envs. `pixel_values` of a fixed INTER_AREA image are bitwise equal
    across transformers 4.40.1 and 4.44.2 (fp32 `88fca69e...`, bf16 `8057babf...`; `results/stack_check_*.json`).
  - **S5:** `_preprocess_image` against the authors' expression `cv.resize(image, (224, 224),
    interpolation=cv.INTER_AREA)` on the 64 saved obs0 frames, bitwise.

### 3.7 Server env (outside `harnessstudio/`) and the numerics verdict

**Verdict: feasible; recommended as primary.** It is built and CPU-checked at
`/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/openvla-authors-numerics` (5.3 GB,
62 dists, `dists_digest` `dbe2e1a86dda730e1120e6db927f394b81be5bd871796eaa7d673f9a9a314355`, freeze
`simpler_ext/results/numerics_venv_freeze.txt` sha256 `dcfe715a...`).

Exact recipe (argv):
```bash
uv venv --python 3.11 /home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/openvla-authors-numerics
uv pip install --python /home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/openvla-authors-numerics/bin/python \
  "torch==2.3.1" "transformers==4.44.2" "tokenizers==0.19.1" "timm==0.9.10" "numpy==1.26.4" "pillow==10.4.0" \
  "accelerate==0.33.0" "opencv-python-headless==4.10.0.84" "vla-eval==0.8.0" "huggingface_hub==0.26.3" "transforms3d==0.4.2"
# resolved: torch 2.3.1+cu121 (nvidia-cublas-cu12 12.1.3.1, nvidia-cudnn-cu12 8.9.2.26), torchvision 0.18.1, safetensors 0.8.0
```

**What has been checked on CPU** (`CUDA_VISIBLE_DEVICES=`, offline, no weights loaded, nothing written under
`harnessstudio/`; `git status` before and after is equal):
- (A) torch `2.3.1+cu121`, transformers `4.44.2`, numpy `1.26.4`, CUDA hidden.
- (B) `harnessstudio.policies.openvla_hs` and `vla_eval.model_servers.openvla` import from the live checkout:
  **HS v2's server code runs on torch 2.3.1 / transformers 4.44.2 at import level.**
  - `_decode` and the SCALE ViT-attention patch are not exercised in this mode.
  - The recorder path uses `generate`'s logits-processor hook, which 4.44.2 still calls on
    `outputs.logits[:, -1, :].clone()` before `torch.argmax` (`transformers/generation/utils.py:2989-3022`).
  - Llama still returns float32 logits (`models/llama/modeling_llama.py:1209`).
- (C) Attention: default = SDPA, `attn_implementation="eager"` = `LlamaAttention`, in both stacks.
- (D) Processor outputs (token ids, pixel_values) are bitwise equal to the current stack's.
- vla-eval in this env is the PyPI 0.8.0 wheel (sha256 `52297503...`), whose package files equal the checkout.
- The host driver (580.173) supports cu121; the A6000 is sm_86.

**What is not checked (GPU, section 7):**
- the forward pass: whether tokens and actions agree with the authors' own `OpenVLAInference` loaded in the same env
  (S6);
- the startup probe hash (S8);
- the bundled cuBLAS being the one loaded: P-20261103-01, so set `inherit_env: false`, use no `LD_LIBRARY_PATH`, and
  list the `expected_libraries` (S9).

**Residual differences from the authors:**
- python 3.11 vs their unknown version (SimplerEnv documents 3.10);
- numpy 1.26.4 vs `numpy<2.0` (exact version unknown);
- the opencv, accelerate and pillow versions are unknown;
- their GPU model and cuDNN/cuBLAS builds are unknown;
- the server runs in a separate process over a websocket (float32 transport, section 3.6).

None of these can be pinned from their repository. They are disclosed, and the label stays "protocol-matched,
numerics close-matched".

**Fallback:** the current env (`~/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068`, transformers 4.40.1, torch
2.10.0, numpy 2.4.2). It has no opencv, so the fallback needs a sibling copy with `opencv-python-headless` under
`runs/pcd-port-20261103/venvs/`. Never `uv pip install` into the shared uv env: Run X's policy JSON pins its
`dists_sha256`.

**Choose before the first scored episode, never after outcomes.** S10 measures token agreement between the two envs on
the obs set; the choice is recorded in the method card.

**Policy JSON.** It may be run-local (critic ruling: the Campaign spec embeds the policy dict). It records:
- `python`: the numerics env;
- `script`: the merged `policies/openvla_simpler_hs.py`;
- `args`: the local snapshot path, `unnorm_key=fractal20220817_data`;
- `env`: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `PYTHONSAFEPATH=1`, `PYTHONDONTWRITEBYTECODE=1`, no
  `LD_LIBRARY_PATH`, `inherit_env: false`;
- `env_manifest.dists_sha256`;
- `verify_files`: the snapshot's remote code and weights (hashes in `runs/pcd-port-20261103/readiness/weights_manifest.json`);
- `frozen_sources`: `openvla_hs.py`, `openvla_logits_hs.py`, `openvla_simpler_hs.py` and vla-eval `openvla.py`;
- `startup_probe_sha16`: measured fresh (S8).

### 3.8 File list (the whole diff, on branch `ext2/simpler-seeded` from `ext2/openvla-contrast`)

| file | change |
|---|---|
| `harnessstudio/benchmarks/simpler_seeded.py` | NEW (section 3.1) |
| `harnessstudio/benchmarks/__init__.py` | docstring: admits protocol subclasses [S-8a] |
| `tests/test_openvla_logits.py` (openvla-contrast's test, on this branch only) | declared: its diff-scope allowlist `EXT_FILES` is extended by this branch's files, because a stacked branch's diff against `main` contains both extensions (found by the after-regress, `simpler_ext/build/regress-after-attempt1-8cb5ddb.json`); this branch's own scope is pinned by `tests/test_simpler_seeded.py::test_simpler_branch_touches_only_its_files` (diff against `da817ba`) |
| `harnessstudio/contracts/simpler_seeded.yaml` | NEW (section 3.2) |
| `harnessstudio/policies/openvla_simpler_hs.py` | NEW (section 3.6) |
| `harnessstudio/profiles/openvla-simpler-logits.yaml` | NEW (section 3.6) |
| `tests/test_simpler_seeded.py` | NEW (section 5): the prototype tests with a fake `simpler_env`; vla-eval only, no torch, no SAPIEN |
| `tests/test_openvla_simpler_server.py` | NEW: S1-S3, run with the server env and a stub model (the part-2 pattern) |
| `docs/writing-a-method.md`, `docs/extensions/simpler-seeded.md`, `docs/extensions/README.md`, `docs/superpowers/specs/2026-10-31-harnessstudio-v2-design.md` (decision 16.11 + the policy table) | the SIMPLER section and status |

Nothing else changes: no core, context, policy_ext, eval, server or other policy file, and nothing binary under
`harnessstudio/` (the 11:04Z rule [S-7a]). The env, configs and weights stay run-local or on `/bigdata`.

## 4. Compatibility claims

1. **No existing code path changes.** All four framework files are new. `openvla_hs.py` and `openvla_logits_hs.py`
   stay byte-identical (S1). No core, context, protocol, mixin, pairing or tier-1 file is edited.
2. **Identity and native stay bitwise on every existing profile.** LIBERO, RoboDojo and RoboCasa365 runs import
   nothing new: the new modules are only reachable by import string, or by `--profile openvla-simpler-logits`.
3. **Every installed method's tier 0/1 is unchanged.**
   - The new profile adds no capability name: `logits_readout` comes from openvla-contrast.
   - `tier1.default_profile` keeps first-match order.
   - `regress.py` must show per-method results equal before and after (section 6).
4. **The authors' post-processing is bitwise equal** given the same raw action: 22,600 steps, 0 mismatches. A known
   float32 transport rounding of at most 6e-8 remains. The claim is "protocol-matched with one disclosed numeric
   transport difference", not "bitwise identical to the authors' run".
5. **No privileged information reaches the policy or methods.**
   - `public_keys` holds only the camera image and the instruction.
   - The seed is in the case key only, which is evaluator-side.
   - Post-processing uses only the policy's own action history, as in the authors' policy wrapper.
6. **No new trainable component:** `trainable_parameters == 0`, and all weights are frozen and hash-pinned [R§6].

## 5. Tests

**Done (prototype, CPU, `runs/pcd-port-20261103/simpler_ext/`):**
- `tests/test_simpler_seeded.py`, 13/13 OK, inside the pinned image with vla-eval 0.8.0 mounted (log
  `simpler_ext/logs/tests_image.log`). The tests cover:
  - one task per seed and unique HS case keys;
  - constructor refusals, including `seed=` (TypeError, no `**kwargs`);
  - the server `{"seed": 0}` being dropped by the real `orchestrator._merge_observation_params`;
  - a fresh env per case with reset kwargs exactly `[{"seed": 16}, {"seed": 5}]`;
  - refusals for an unknown seed, `episode_idx != 0` and a missing task_id;
  - observation shape and instruction;
  - full episodes through the real `SyncEpisodeRunner`:
    - success at steps 10-12 still runs all 80 steps, with metrics `{success: True, final_success: False,
      first_success_step: 9, steps: 80}`;
    - success only at the last step;
    - never a success;
    - truncation at 113 even with `max_steps` 200;
  - env actions equal to the reference post-processing bitwise, with the first-step gripper 0 in every episode and
    the sticky mechanism firing;
  - an instruction change resetting the gripper state;
  - refusal of a non-7-D action.
- `parity/authors_post_parity.py`, OK. It runs inside the image with numpy 1.24.4 and transforms3d 0.4.2.
  - It loads the authors' `simpler_env/policies/__init__.py` (`152db35a...`), `openvla_model.py` (`abea979c...`) and
    `openvla_contrast.py` (`989a6e4d...`) **by source**, with stubs for torch, transformers, matplotlib and cv2. It
    drives `OpenVLAInference.step` (the baseline path) and `postprocess_actions` (the PCD path) on 200 synthetic
    113-step token sequences:
    - random tokens;
    - open-close-open gripper;
    - 3-step flicker;
    - deltas straddling 0.5;
    - the R1 boundary tokens.
  - **0 mismatches in 22,600 steps.** Three things are compared bitwise, with dtype:
    - the server decode formula (`modeling_prismatic.py:521-534`) against the authors' `_decode_actions`;
    - our env action against the authors' `step`;
    - `postprocess_actions` against `step`.
  - Coverage: 1,287 sticky onsets, 18,736 steps with a nonzero gripper.
  - Every sabotage control is detected: no euler2axangle (1,130 mismatching steps), a 14-step sticky (296), a
    float32 server action (1,130).
- `tests/real_env_smoke.py`: the real SimplerEnv in the image, render cpu, no policy.
  - **[S-8b]** Coke seed 0 matched the readiness probe in the prototype run; for seed 7 the prototype recorded no
    reference (`matches_readiness_probe: null`), although its hash `6cc7107e...` equals
    `readiness/case_list_cpu_obs0.json`. Re-recorded on the built worktree code with both references:
    `runs/pcd-port-20261103/simpler_ext/build/results/real_env_smoke_worktree.json` gives `true` for seeds 0 and 7.
  - The post-processed float64 action is accepted by `env.step`, `info["truncated"]` is present, and the metrics keys
    are as specified.
- `numerics/stack_check.py` and `numerics/attn_check.py` in both envs (section 3.7).

**To add in the worktree:**
- move the prototype tests to `tests/test_simpler_seeded.py`. They run under the shared `hs` interpreter when
  transforms3d is present, and are otherwise skipped with a reason; the image run stays the authoritative one;
- S1-S3 in `tests/test_openvla_simpler_server.py`;
- a golden-bytes test that `SimplerSeededBenchmark` is not importable from any existing profile or test path (import
  isolation).

## 6. Regress gate (CPU)

The same procedure as openvla-contrast section 7, with its R5/R6 prerequisites:
1. **Before any edit:** run `regress.py` on the new worktree at its base commit (`1fc5b10`; the after-regress runs on the rebased branch on `da817ba`, see the build report) and keep it as the
   "before" report. Copy the untracked method assets, sha-checked and not committed (at least
   `methods/tube_steer_vfb/hs_tube_steer_vfb/assets/visual_memory.npz`).
2. **After the build:**
   ```bash
   cd /home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio
   PYTHONDONTWRITEBYTECODE=1 nice -n 19 ionice -c3 python3 scripts/regress.py --source ../hs2-ext-simpler-seeded \
     --python /home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio/.venv/bin/python \
     --out ../hs2-ext-simpler-seeded/regress.json        # must report "ok": true; per-method results == before
   ```
3. **Server tests** with the numerics env, read-only:
   ```bash
   CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     HF_HOME=/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/hf_scratch/numerics \
     HF_MODULES_CACHE=/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/hf_scratch/numerics/modules \
     PYTHONPATH=../hs2-ext-simpler-seeded nice -n 19 \
     /home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/pcd-port-20261103/venvs/openvla-authors-numerics/bin/python -B \
     -m unittest tests.test_openvla_simpler_server -v
   ```
4. **Image tests:** `tests/test_simpler_seeded.py` and the parity script inside the pinned image, with `--runtime
   runc -e NVIDIA_VISIBLE_DEVICES=void --network none` and vla-eval 0.8.0 mounted.
5. **Integrity before and after:**
   - the live checkout (`git status --porcelain`, plus a Campaign-style tree hash of `harnessstudio/harnessstudio`);
   - the shared venv (`.pth` and dist-info list);
   - the numerics env's `dists_digest`.
   - Before and after must be equal.
6. **Commit inside `ext2/simpler-seeded`** with a message naming the EXT ID.

## 7. GPU checks (sole evaluator)

These run only after Run X closes, or after a GPU handoff that the supervising session logs in
`runs/scaleagent-shared/COORDINATION.md`. Each runs on one leased, process-free GPU among 4-7, against the **worktree**
package.

| id | check | pass rule |
|---|---|---|
| B1 | GPU obs0 re-hash: reset-only on the 64 cases (render gpu) | all 64 `(instruction, obs0)` distinct. If a duplicate appears, apply the edit 5 replacement rule on the GPU hashes, before any policy episode. Record the hashes. The CPU hashes [R§7] are advisory |
| B2 | container isolation (first launch; then every launch through the [S-6] per-reset check) | inside the benchmark container `nvidia-smi --query-gpu=uuid` lists exactly the leased GPU's UUID, `NVIDIA_VISIBLE_DEVICES` equals the leased index, both Vulkan loader variables name only the pinned `nvidia_icd.json`, and the SAPIEN/svulkan2 log names the NVIDIA device; any deviation aborts |
| B3 | render determinism | two reset-only passes give equal hashes; a fixed 80-step synthetic raw-action replay (no policy) twice gives equal per-step obs hashes |
| B4 | one complete seeded baseline episode through `hs eval identity --arms baseline` with the new server and contract (SETTING milestone) | valid record; 80/113 steps; metrics have success, final_success, first_success_step and steps; inspect frames, actions and timing |
| B5 | null vs baseline (shadow null of the PCD method) on two development cases | M (or R then M on one re-pair); X blocks |
| B6 | timing and memory | per-step policy, perception and render seconds; peak memory; renders per step on GPU |
| B7 | float32 transport effect **[S-4]** (rule fixed by SETTING Addendum 2 E-2) | replay open-loop, in separate container launches, the float64 raw-action sequences of the smoke episodes from the server's pre-cast log (at least the smoke baseline episode plus one predeclared non-subset drawer episode), once as float64 and once float32-cast, same seed. PASS ("no behavioral difference") = identical per-step success flags, identical gripper (sticky) state sequence, and max TCP position diff <= 1e-5 m and max TCP orientation diff <= 1e-5 rad over the episode. Any violation makes the Setting A label "adapted (float32 action wire)" |
| S6 | authors-path equivalence on the obs set (the 64 GPU obs0 frames plus the frames of the seed-99 smoke and B4 episodes, never subset seeds; A-6): load the authors' `OpenVLAInference` by source in the same env, own process after the server, same weights, eager attention, and call its `reset(instruction, seed=s)` as `parallel_inference.py:181` does [S-2] | tokens equal; raw action float64 bitwise equal to the server's pre-cast log entry [S-4] |
| S7 | logits path in the new mode | the openvla-contrast G2/G3/G3b analogues pass, with raw-prompt rows (`logits_tokens` = native tokens) |
| S8 | startup probe | a fresh `startup_probe_sha16`, recorded in the run-local policy JSON |
| S9 | loaded libraries | `/proc/PID/maps` shows the env's bundled cuBLAS 12.1, no `/usr/local/cuda-12.6` |
| S10 | numerics env vs fallback env | token agreement on the obs set (measurement); decide before the first scored episode |

## 8. Merge order and COORDINATION rule 2 [S-7]

1. **openvla-contrast first.** It is built and committed (`6cb581f`, `1fc5b10`, `da817ba`); its GPU checks G1-G9 and its diff
   review are pending. This branch sits on its head `da817ba` [S-7e]; if openvla-contrast changes before it merges,
   rebase and re-run S1-S3 and the regress gate.
2. **If ksample, or any edit to `openvla_hs.py`, lands first** (the openvla-contrast R8 situation): new review, new S1
   pins, and S6-S9 re-run.
3. **Merge `ext2/simpler-seeded` into `main` only when all hold:**
   - its independent diff review (scope: `git diff da817ba..ext2/simpler-seeded`) is closed with no open objection;
   - `python3 harnessstudio/scripts/regress.py --inflight` is empty;
   - **[S-7b]** COORDINATION rule 2 in full: no frozen non-terminal Campaign idea in **Run C**
     (`runs/scaleagent-20261103`), **Run X** (`runs/scaleagent-codex-20261103`) **or this run** (authority-amendment-1
     `framework_merge`); the other runs are checked by yes/no queries only (grep -q / grep -c, or the COORDINATION
     log), never by printing their files;
   - nothing compiled or binary under `harnessstudio/` (the supervisor's 11:04Z rule, logged in COORDINATION.md, not a
     numbered Rule [S-7a]);
   - **[S-7c]** the log line (time UTC, run, what, commit) is appended to `runs/scaleagent-shared/COORDINATION.md`
     **before** the merge;
   - after the merge, re-run `regress.py` on `main`.
4. **[S-7d] Push** only after `regress.py` passes on `main` AND only after Run X's Campaign is terminal, not merely
   between frozen ideas (authority-amendment-2 `framework_push`).
5. **The benchmark half could merge alone,** since it does not depend on openvla-contrast. One stacked branch is kept
   (one review, one merge); the critic accepted this order.
6. **Scored runs** start only from the merged files. SIMPLER has no v2 Campaign case model (A-5), so the run records by
   hand under `harnessstudio/docs/results/`: the method fingerprint, the merged package tree hash, the sha256 of
   `simpler_seeded.py` and `openvla_simpler_hs.py`, the image digest, the policy JSON sha, the server env `dists_digest`
   and the case-list hash (both definitions, A-4).

## 9. Risks, rejected alternatives and open points

1. **float32 transport (rejected alternative: preserve float64 on the wire).** Changing
   `policy_ext/protocol.py:61` and vla-eval's `predict.py:241` would touch a core file every running method imports,
   plus an external package. The measured effect is at most 6e-8 with no gripper-state change. Disclose it and measure
   it (B7); do not change the core.
2. **Rejected: sending tokens instead of actions,** with the benchmark de-tokenizing in float64. That would be exact,
   but it changes the meaning of "action" for every HS mechanism (embodiment, eef deltas, analytic skills) and puts
   the policy's normalization statistics into the benchmark.
3. **Rejected: a `preprocess` flag in `openvla_logits_hs.py` or `openvla_hs.py`.** See section 3.6.
4. **Env reuse.** The authors reuse one env per worker. Fresh-env obs0 is equal on 15/15 cases [R§7]; later-step
   equality was not tested. B3's replay covers the fresh-env path only, so this residual is disclosed.
5. **Overlay PNG bytes** cannot be verified against the authors' machine [R§3]. Disclosed.
6. **The authors' 300-trial protocol is not in the code** (`n_trajs=100`). Setting A uses its own predeclared 64
   cases. Nothing here depends on that question.
7. **Unsupported tasks** (WidowX, Apple Drawer) are refused. Adding them needs the binarized gripper and
   `bridge_orig`, the 3rd_view camera, and the sub-task advance (`parallel_inference.py:236-240`). That would be a
   later, separately reviewed change.

## 10. Review edits S-1..S-8 and advisories (critic-simpler-ext; pointers)

| edit | where | evidence |
|---|---|---|
| S-1 no-mask generate (blocking) | 3.6 `_generate`; S3 | `policies/openvla_simpler_hs.py` `_generate`; `tests/test_openvla_simpler_server.py` `TestEagerNoMaskDecode` + AST test; passes in both envs (`simpler_ext/build/results/tests_server_{openvla-authors-numerics,openvla-fallback}.log`) |
| S-2 cuDNN flags | 3.6 `__init__`, `_check_model`; S6 | `__init__` sets them; `_check_model` asserts; S6 calls the authors' `reset(instruction, seed=s)`; perception-service owner informed through the build report |
| S-3 full-vocabulary decode vs the authors | S3b | `simpler_ext/build/checks/decode_vs_authors.py`; 32,064 ids x 7 dims + 20,000 mixed rows, 0 mismatches in both envs (`simpler_ext/build/results/decode_vs_authors_*.json`) |
| S-4 pre-cast float64 log; B7 rule | 3.6 `_log_action`; 7 B7, S6 | `test_s4_precast_float64_log`; B7 rule = SETTING Addendum 2 E-2 |
| S-5 honest specs; eager class | 3.6 `get_action_spec`, `get_observation_params`, `_check_model`; 3.1 | `test_s5_specs_and_eager_assertion`, `test_action_spec_matches_benchmark_spec`, `test_specs_and_metric_keys` |
| S-6 per-launch isolation; NVIDIA ICD | 3.5; 7 B2 | `SimplerSeededBenchmark._check_render_isolation`; `TestRenderIsolation` (pass + 7 refusals); config templates |
| S-7 merge rules | Base paragraph; 8 | as written above |
| S-8 evidence statements | 3.1, 3.8, 5 | `benchmarks/__init__.py` docstring + spec 16.11; `real_env_smoke_worktree.json` |

Advisories: A-1 (benchmark-only GPU checks B2 -> B1 -> B3 first, cross-launch B3) adopted in the GPU plan; A-2 success
read from `info["success"]` (falls back to `terminated` only if the key is absent); A-3 `get_metric_keys` adds
`final_success`; A-4 both hash definitions in the manifest; A-5 manual results record (section 8.6); A-6 S6 in its own
process with smoke/B4 frames only; A-7 `SimplerDecodeShort`; A-8 INTER_AREA version sensitivity: optional, listed in
the GPU plan as a CPU side check (opencv 4.10.0.84 in both server envs vs 4.12.0.88 in the image; authors' version
unknown); A-9 `test_a9_task_config_is_ignored`; A-10 left to the diff reviewer (S1 byte-identity + regress, or a 2-case
native-equivalence check of one existing method on the worktree).
