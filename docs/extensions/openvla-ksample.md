# EXT openvla-ksample (proposal, 2026-11-02): cheap K seeded samples and per-token statistics on the OpenVLA server

> **Status: NOT MERGED - branch not merged.** Built and GPU-checked in worktree `hs2-ext-openvla-ksample`
> (branch `ext2/openvla-ksample`, which has no commits beyond main 785515f: the 11 changed files are uncommitted edits in
> that worktree as of 2026-11-03); never merged into `main`, and `harnessstudio/policies/openvla_hs.py` on `main` does not
> contain it. No method on `main` may rely on it. SCALE++ closed without using it (`runs/scalepp-20261102/REPORT.md`).
> Committed as a record in the scaleagent-20261103 housekeeping task.


Author: rmh_coder (agent ref filled by the supervisor), run `scalepp-20261102`, task `ext-ksample-proposal`.
Status: **reviewed (approve with changes C1-C9, `runs/scalepp-20261102/reviews/ext-ksample-review.json`); built in the worktree `hs2-ext-openvla-ksample` (branch `ext2/openvla-ksample`, NOT merged). Section 9 records the amendments and supersedes any conflicting text above.** Originally: proposal only, EVOLUTION.md Loop 2 step 1. Nothing is built yet: no worktree, no edits under
`harnessstudio/harnessstudio`, no GPU use. It stops here for an independent `rmh_critic` review. Branch name once
approved: `ext2/openvla-ksample` (worktree `../hs2-ext-openvla-ksample`).

## 1. Need

- **Mechanisms that need K samples per decision.** The SCALE++ queue holds two kinds:
  - the literature delta's top mechanism, `cyclevla_mbr` (minimum-Bayes-risk / medoid selection over K sampled
    actions, possibly SCALE-sampled; `runs/scalepp-20261102/literature/delta-20261102.json`);
  - several uncertainty-gated ideas. These need per-token confidence of the native greedy decision (to gate) and of
    each sample (to score), over OpenVLA's 256 action bins.
- **What the server does today** (`harnessstudio/policies/openvla_hs.py`, `HarnessOpenVLAServer.sample_chunks`):
  - **Every seed repeats the whole computation.** Each seed re-runs image preprocessing, both ViTs, the projector, the
    ~280-token LLM prefix and 6 cached decode steps.
    - Sampler off: `OpenVLAModelServer.predict` once per seed.
    - Sampler on: `_decode` once per seed.
  - **No confidence values.** It returns no per-token statistics. The `u`/`tau` extras of the SCALE path are not
    readouts, so the client filters them out before methods see them.
- **Cost today.** K samples cost about K times a native decision (Section 5: c_pol = 2.6 / 4.2 / 6.6 for K = 3 / 5 / 8).
- **Budget.** The R9 timeline (`runs/scalepp-20261102/control/timeline-r9.json`) treats c <= 2 as practical. c = 4
  pushes the latest open-full to 01:43Z, which it marks "infeasible in practice".
- **Why existing slots cannot do this:**
  - Methods only see `ctx.policy.sample(k, sampler, readouts)`.
  - Sharing a prefix across seeds, batching decode streams and computing token statistics all happen inside the
    policy server.
  - A method must not monkeypatch the framework or the server.
  - The `logits_readout` capability (full logits) is declared but implemented by no server. It would also move
    32000 floats x 7 tokens x K per step over the websocket, for statistics that cost almost nothing on the GPU.

## 2. Design (opt-in; every request without the new options runs today's code, unchanged)

### 2.1 Protocol and capabilities

- **New capability names** (`core/capabilities.py`, tier T2/T3):
  - `shared_prefix_sampling`: "K seeded samples of one decision share one image+prompt prefix (KV cache computed
    once)".
  - `token_stats_readout`: "per-token statistics of the decoded action tokens are returned as extras".
- **New readout `token_stats`.** `core/method.py` maps `"token_stats" -> "token_stats_readout"` in the existing
  readout-capability check.
- **New proposer params**, validated at registration:
  - `streams`: `"per_seed"` (default, today's behaviour) | `"shared_prefix"` (needs `shared_prefix_sampling`).
  - `batched`: bool, default true. Only with `streams: shared_prefix`.
  - `anchor`: `"sampler"` (default) | `"native"`. What sample 0 is in the candidate arm; see 2.2.
  - `state_follows`: `"stream0"` (default, today's behaviour) | `"executed"`. Needs `decoding_hooks`; see 2.3.
- **Request field `options`** (`policy_ext/protocol.py`). `make_request(raw_obs, seeds, sampler, readouts,
  options=None)` adds `spec["options"] = {"streams", "batched", "anchor", "commit_index"}` only when options are
  given.
  - The protocol version stays `v: 1`. The key is optional and absent from every existing request, so their bytes
    are unchanged.
- **Server side** (`policy_ext/mixin.py`):
  - If `spec["options"]` is present, the server must declare `shared_prefix_sampling`; otherwise the mixin raises.
    A server can never silently ignore the options.
  - The call then goes to a new method, `sample_chunks_ext(obs, seeds, sampler, readouts, options, ctx)`.
  - Without options the mixin calls `sample_chunks` exactly as today. Every existing server (pi05_openpi, gr00t,
    robodojo_pi05, fake) is untouched and bitwise unchanged.
- **Client side:**
  - `context/policy.py`: `PolicyHandle.sample(..., options=None)` forwards the options.
  - `context/episode.py`: `_PolicyProxy.sample(k, ..., streams=None, batched=True, anchor="sampler",
    commit_index=None)` builds them. The shadow (null) handle gets the same options, so the null pays the same
    compute.
  - `components/decision_loop.py` passes the proposer's `streams`/`batched`/`anchor` in every arm. It passes
    `commit_index` only in the candidate arm with `state_follows: executed` (2.3).
  - The sampler stays off in baseline/null (`arms.sampler_effective`).

### 2.2 Server: `HarnessOpenVLAServer.sample_chunks_ext` (`policies/openvla_hs.py`)

The existing `predict`, `sample_chunks`, `_decode`, `_vision_temperature` and `_actions` stay **byte-identical**. A
unit test pins their AST hashes. The new path is new functions next to them.

Notation: K = `len(seeds)`; `inputs = self._inputs(obs)`, computed once per request, exactly as today.

1. **Sample 0 (the anchor)** is bitwise identical to sample 0 of the same request without options:
   - sampler off (baseline, null, greedy methods): `OpenVLAModelServer.predict` (native greedy `generate`);
   - sampler on, `anchor: "sampler"`: the same op sequence as `_decode(decoder, state, inputs, seeds[0])`, i.e.
     today's per-seed SCALE path for seed 0. It runs in the new `_prefix` + `_continue_b1` functions. A GPU test
     proves bitwise equality with `_decode` (Section 7, check E2).
   - sampler on, `anchor: "native"`: native `predict`. Selecting index 0 then executes the native greedy action.
     This is the "greedy is one of the MBR candidates" setting.
2. **Shared prefix.** One multimodal forward at batch 1:
   - same `input_ids` (+ empty token 29871), `attention_mask`, `pixel_values`, `use_cache=True`;
   - inside `_vision_temperature(gamma)`, with gamma = `decoder.attention_temperature(state)` computed once from the
     episode state (sampler on), or the stock attention (sampler off).
   - It returns the first-token logits and `past_key_values`.
   - With `anchor: "sampler"` this prefix is also the anchor's prefix (it is the same computation), so the anchor
     costs only its 6 decode steps.
   - With a native anchor, or sampler off, the anchor's own prefix is inside `generate` and the shared prefix is a
     second forward.
3. **Streams 1..K-1.** Each gets `torch.Generator(device).manual_seed(seeds[i])`, the same generator construction
   as today.
   - **Token 1** is selected from the prefix logits, which are bitwise identical for every stream.
   - **Tokens 2..7**, `batched: true`:
     - The prefix KV cache is expanded to batch B = K-1 (`expand(...).contiguous()` per layer; about 150 MB per
       stream).
     - Each step is one forward of `self._model.language_model(input_ids=[B,1], attention_mask=None,
       position_ids=None, past_key_values=..., use_cache=True, return_dict=True)`. These are the arguments
       Prismatic's own cached-generation branch passes. That branch asserts batch size 1
       (`modeling_prismatic.py:326`), so the batched step bypasses only the wrapper and its assert.
     - Per stream, in stream order: `decoder.select_token(logits[i], gen_i)` with sampler on, or argmax with sampler
       off.
   - **Tokens 2..7**, `batched: false` ("sequential"): stream i decodes alone at batch 1 from a clone of the prefix
     cache. Every sample is then the per-seed path's op sequence. A GPU test shows bitwise equality, sample by
     sample, with today's per-seed samples. This mode is for equivalence and for methods that need strict stream
     isolation, not for speed.
   - **Prefix cache safety.** transformers 4.40 Llama converts legacy tuples to a `DynamicCache` and grows it with
     `torch.cat` (no in-place write). The build still clones the prefix cache per stream or batch and asserts the
     prefix tensors are unchanged after the streams (test E6).
4. **Actions:** `self._actions(tokens)` per stream, unchanged.
5. **Extras per sample:** `{"seed", "tokens", "gamma", "u", "tau"}` as today. With readout `token_stats` (2.4) the
   statistics are added.

### 2.3 SCALE's per-episode state with K streams

Today all seeds of a request start from the same state (isolated streams). The state then advances with seed 0's
token infos: `self._states[key] = decoder.next_state(state, first_infos)`. With K = 1 that is the official loop,
because the executed sample is seed 0.

With K > 1 and a selector, the executed sample can be any index. The official SCALE has no K > 1 rule, so this is
a declared design choice:

- **`state_follows: "stream0"` (default).** Exactly today's rule. Every stream starts from the same state S_t and
  gets gamma(S_t) on the shared prefix. Afterwards `S_{t+1} = next_state(S_t, infos of stream 0)`.
  - It is bitwise today's behaviour.
  - It equals "follows the executed sample" whenever the selector picks 0.
- **`state_follows: "executed"` (opt-in).** The history follows the executed action, as in the official
  single-sample loop.
  1. The server keeps, per (session, episode, sampler key), a *pending* list holding the token infos of all K
     streams of the last request. The state itself is not advanced yet.
  2. The next sample request of that episode carries `options.commit_index` = the index the decision loop executed
     last time (`arms.executed_index`; in the candidate arm, the selector's choice).
  3. Before computing gamma, the server commits: `S_{t+1} = next_state(S_t, pending[commit_index])`.
  4. Then it computes gamma(S_{t+1}) and decodes the new request.
  - **Missing or out-of-range `commit_index`** while a pending list exists is a protocol error. The server raises
    and never falls back to stream 0 silently.
  - **The first request of an episode** has nothing pending.
  - **The last decision's pending list** is dropped at EPISODE_END, like today's state.
  - **The gate may change K between decisions.** The commit refers to the previous request's K.
  - **Baseline and null** run with the sampler off, so there is no decoder state and nothing to commit.
  - The derived null therefore stays the native policy.
- In both modes, every stream of one request sees the same gamma, because the vision-attention temperature is a
  property of the shared prefix. This is also true today: every seed recomputes the prefix with the same gamma.

### 2.4 Per-token statistics (readout `token_stats`)

**What is computed.** For every decoded token of every returned sample, on the GPU in float32, from that step's
logits restricted to the 256 action bins `[vocab_size-256, vocab_size)` (the same mask SCALE uses):
- `logprob`: log-softmax of the chosen token;
- `entropy`: in nats;
- `margin_p`: top-1 minus top-2 probability;
- `margin_logit`: top-1 minus top-2 logit;
- `greedy_agree`: chosen == argmax.

The statistics describe the model's untempered distribution at that step. For sampler streams this is the
distribution under the request's gamma prefix. SCALE's tau/u are also returned when the sampler is on.

**What is returned.** Companion-key convention of `velocity` / `velocity_timesteps`:
- `extras["token_stats"]`: float32 `[7, 4]`, columns `(logprob, entropy, margin_p, margin_logit)`;
- `extras["token_stats_tokens"]`: int32 `[7]`;
- `extras["token_stats_greedy_agree"]`: bool `[7]`;
- `extras["token_stats_u"]`, `extras["token_stats_tau"]`: float32 `[7]`, sampler streams only.

**The native greedy decision** (sample 0 with the sampler off, or with `anchor: native`):
- It must stay bitwise native, so its statistics come from a recording `LogitsProcessor`.
- The processor stores the step scores and **returns them unchanged**.
- It is passed to the native `predict_action(..., logits_processor=LogitsProcessorList([rec]))` through a copy of
  vla-eval's `predict` that differs only by that kwarg. The copy is unit-tested against the vla-eval source; GPU
  check E4 proves the tokens equal.
- The checkpoint's `generation_config.json` declares no processors or warpers, and `do_sample=False`, so the
  recorder is the only processor and argmax sees the same tensor.
- Without the readout, the plain `predict` runs, as today.

**Leakage.** The statistics are functions of public inputs and frozen weights only. Seeds remain filtered out of
the extras by `PolicyHandle.sample` (only requested readouts reach methods).

### 2.5 Fake server and profiles

- `policy_ext/fake.py` gets a `sample_chunks_ext` with the same semantics (anchor = sample 0 of today's request,
  deterministic streams, synthetic `token_stats`). Tier 1 of methods using the extension then runs on CPU.
- `profiles/openvla-libero-hs.yaml` adds `shared_prefix_sampling` and `token_stats_readout` to its capabilities.
  No other profile changes.

## 3. Compatibility claim

- Every existing request carries no `options` and no `token_stats` readout, so it runs today's code:
  - **identity:** native `predict`;
  - **SCALE:** per-seed `_decode`;
  - **GR00T / pi0 servers:** untouched files.
- `isolated_streams` stays true for `streams: per_seed` and for `shared_prefix` with `batched: false`.
- **Exception for `batched: true`.** Sample i (i >= 1) depends on its seed and on the batch size K. Batched GEMMs pick
  different kernels than batch-1 GEMMs, so the logits of tokens 2..7 can differ in the last bits from the per-seed
  path. A token near a decision boundary can then change. The anchor (sample 0) is unaffected.
  - The capability text and `docs/writing-a-method.md` state this.
  - A method that varies K with a gate and needs sample i to be invariant to K must use `batched: false`.
  - CRN across arms is unaffected: candidate requests have the same shape in every case, and the null runs with the
    sampler off.

## 4. Freeze and hash impact

- **Framework hash.** The HarnessStudio package hash changes. These files change:
  - `core/capabilities.py`, `core/method.py`;
  - `policy_ext/{protocol,mixin,fake}.py`;
  - `context/{policy,episode}.py`, `components/decision_loop.py`;
  - `policies/openvla_hs.py`, `profiles/openvla-libero-hs.yaml`;
  - docs and tests.
- **Campaigns.** Every v2 Campaign freeze hashes the whole package. Merge only while no v2 Campaign idea is frozen
  and non-terminal, checked with `python -m robometaharness status` for every v2 run. Per the supervisor this holds
  now.
- **Method fingerprints.** Methods hash their own configs, so their fingerprints (`identity`, `scale`, ...) are
  unchanged.
- **OpenVLA policy JSON.** `runs/scalepp-20261102/ext/openvla-policy/measure_probe.py` binds the probe measurement
  to the sha256 of `openvla_hs.py`, and `--fill` refuses after a change. **Order:** merge this extension before the
  evaluator's probe measurement, or measure again after the merge. The native path is untouched, so the probe hash
  should not change. The re-measurement proves it.
- **Development map comparability.** The baseline (identity) and SCALE records the evaluator produced before the
  merge stay comparable only if E1 and E2 show bitwise-identical actions after the merge on the same cases, server
  JSON and pinned cuBLAS mode. If either fails, those records are reported as pre-extension and rerun. They are
  never mixed silently.
- **Running processes.** EVOLUTION step 6 applies: merge only when `harnessstudio/scripts/regress.py --inflight` is
  empty. The SCALE++ development screens import this package and the profile, so the evaluator must schedule a
  quiet window.

## 5. Cost model and c estimates (no GPU; from measured latencies)

**Measured** (development scene t0/init0/seed7, single stream per server):
- server round trip per native call: 0.293-0.319 s (`accounting.policy_seconds / steps`;
  `runs/scalepp-20261102/eval/determinism/g{4,5,6,7,7-ldlp}/baseline/records.jsonl`);
- episode wall per step about 0.36 s;
- SCALE per step about 0.31 s vs 0.30 s native (`harnessstudio/docs/results/scale.md`).

**Decomposition** of one native call T0 = O + P + D:
- **O, about 0.07 s:** per-request CPU and transport (msgpack, JPEG round trip, Lanczos resize, centre crop,
  processor), `generate` bookkeeping.
- **P, about 0.07 s:** DINOv2-L + SigLIP-SO400M at 224 px (about 0.37 TFLOP) and the Llama-2-7B prefix over about
  280 tokens (about 3.8 TFLOP). This yields token 1.
- **D, about 0.16 s:** 6 cached decode steps at about 27 ms each. Bandwidth-bound: 13.5 GB of bf16 weights at
  768 GB/s gives a 17.6 ms lower bound, and HF eager runs at about 25-30 ms.

**Other terms:**
- **B-stream batched decode.** A step costs `t_tok * (1 + beta * (B - 1))` with beta about 0.05 (range 0.02-0.10).
  The weights are read once; each stream adds about 150 MB of KV reads plus small compute.
- **Token selection.** About s = 1 ms per token per stream (SCALE's `select_token`: masked softmax, top-k, a few
  `.item()` syncs; range 0.5-2 ms).

**Ratios.** c_pol = server time / T0. c_wall = (0.06 + server time) / 0.36, the episode-wall ratio the timeline
formulas use.

| K | today, per seed: O + K(P+D) + 7Ks | shared prefix, sequential: O + P + K·D + 7Ks | **shared prefix, batched, anchor = sampler: O + P + D + D(1+β(K-2)) + 7Ks** | same, anchor = native (+P) | batched without a bitwise anchor (not proposed) |
|---|---|---|---|---|---|
| 3 | c_pol 2.6 (c_wall 2.3) | 2.1 (2.0) | **1.63 (1.53)** | 1.86 (1.72) | 1.12 (1.10) |
| 5 | 4.2 (3.7) | 3.3 (2.9) | **1.73 (1.61)** | 1.96 (1.80) | 1.22 (1.19) |
| 8 | 6.6 (5.6) | 4.9 (4.3) | **1.88 (1.73)** | 2.11 (1.93) | 1.37 (1.31) |

**Sensitivity of the proposed mode** (batched, anchor = sampler):
- optimistic (β 0.02, s 0.5 ms): c_pol 1.58 / 1.62 / 1.69 for K = 3 / 5 / 8;
- pessimistic (β 0.10, s 2 ms): c_pol 1.73 / 1.93 / 2.23;
- prefix-heavier split (P 0.10 s, D 0.13 s, β 0.05, s 1 ms): c_pol 1.53 / 1.62 / 1.75.

**Per-token statistics** add batched reductions over 256 bins and one host transfer per request: under 1 ms per
token, i.e. c_pol < +0.03.

**The null arm.** It runs with the sampler off and the same options:
- native `predict` (O+P+D) for index 0;
- a second shared prefix (P);
- K-1 batched argmax streams.

So the null pays the candidate's cost plus about P (+0.23 in c_pol). That is conservative for the overhead control
and disclosed.

**Measurement.** The R9 planning value of c must be measured on the worktree build by check E7 before any candidate
schedule relies on it. These numbers are planning estimates only.

## 6. Tests (CPU, in the worktree; part of `regress.py`)

- **Protocol and mixin:**
  - requests without options are byte-identical to today's (golden msgpack bytes);
  - options round-trip;
  - a server without `shared_prefix_sampling` refuses options;
  - `v` stays 1.
- **Registration:**
  - `token_stats` needs `token_stats_readout`;
  - `streams: shared_prefix` needs `shared_prefix_sampling`;
  - `state_follows: executed` needs `decoding_hooks` and `streams: shared_prefix`;
  - unknown values are refused.
- **Decision loop:**
  - options reach baseline, null and candidate requests;
  - the sampler stays off in baseline/null;
  - `commit_index` is sent only in the candidate arm with `state_follows: executed` and equals the previous
    `executed_index`; the null's executed index stays 0.
- **Server logic, model-free** (a stub model with deterministic logits and a stub KV cache):
  - anchor = today's sample 0;
  - streams use their own generators in stream order;
  - stream0 vs executed state commits against a fake decoder (`next_state` call sequence), and protocol errors;
  - `token_stats` against a float64 numpy reference on synthetic logits (atol 1e-5).
- **Source pins:** AST hashes of `predict`, `sample_chunks`, `_decode`, `_vision_temperature` and `_actions` equal
  the pre-extension source. The `predict` copy used for the recorder differs from vla-eval's
  `OpenVLAModelServer.predict` only by the `logits_processor` kwarg (AST diff).
- **Fake server:** `sample_chunks_ext`; tier 1 of a toy method using `streams: shared_prefix` + `token_stats`.
- **Regression gate:** `regress.py` ok, with tier 0/1 of every installed method unchanged.

## 7. GPU equivalence plan (sole evaluator, one leased process-free GPU, pinned OpenVLA JSON serving mode)

All checks use the worktree server started from the policy JSON's interpreter, env and args. Script:
`runs/scalepp-20261102/ext/ksample/gpu_equiv.py`, written at build time and reviewed with the diff.

- **E1 native bitwise.**
  - One identity (baseline) episode on development scene libero_10 t0/init0/seed7. All 520 action and observation
    hashes must equal the evaluator's pinned-mode record (`eval/determinism/g7-ldlp`).
  - Plus `hs verify identity --tier 2` on 2 development cases (null vs baseline M).
- **E2 SCALE per-seed bitwise.**
  - One SCALE episode on a development scene already in the dev map, with equal action hashes.
  - Plus, on 50 fixed development observations: `sample_chunks_ext(streams=shared_prefix, batched=false)` equals
    `sample_chunks` per seed, bitwise in tokens, actions, u, tau and gamma, for K = 8, SCALE and greedy hooks.
- **E3 anchor.** With `batched: true`, sample 0 equals today's sample 0 bitwise, on the same observations.
- **E4 native statistics.** `predict` with the recorder gives tokens and actions bitwise equal to plain `predict`.
  The recorded statistics match a float64 recomputation within 1e-5.
- **E5 batched streams.**
  - Token agreement with sequential mode is reported per position, expected >= 0.95 on these observations.
  - Same per-dimension action means within sampling error.
  - Two server processes give bitwise-identical batched outputs.
- **E6 cache safety.** The prefix KV tensors' sha256 is unchanged after decoding all streams.
- **E7 timing.** Server time per decision for K = 1/3/5/8 in each mode, measured with the CUDA allocator warm.
  Report c_pol and peak memory. This replaces Section 5's estimates.
- **E8 tier 2.** `hs verify scale --tier 2` and one new toy method (`shared_prefix` + `token_stats`): null vs
  baseline M.

## 8. Risks

- **Batched numerics** (Section 3): streams 1..K-1 are not the per-seed samples. They are new, deterministic
  samples. Methods must not claim "the same samples as SCALE's per-seed path" unless they use `batched: false`.
- **Bypassing Prismatic's batch-1 assert.** The batched step calls `language_model` directly. A future remote-code
  revision could change the wrapper. The policy JSON pins the remote code (revision 47a0ec7f, verify_files); the
  build adds a source check of the wrapper branch.
- **transformers cache semantics** (legacy tuple vs DynamicCache, in-place growth): check E6, plus defensive clones.
- **`state_follows: executed`** is a new protocol interaction: a lost or stale `commit_index` changes SCALE's
  gamma. It is strict (it raises) and tested, and stays opt-in.
- **Recorder `LogitsProcessor`.** It could alter `generate`'s path. Check E4; only used with the `token_stats`
  readout.
- **Latency estimates may be off.** Section 5's P/D split is inferred, not profiled. If E7 shows c > 2 for K = 8,
  methods use K <= 5 or a gate.
- **Memory.** K = 8 adds about 1.2 GB of KV plus activations on a 48 GB A6000 next to the 15 GB model: fine. Two
  client streams per GPU still serialize on the server's inference lock.
- **Schedule.**
  - Build, CPU tests and regress: about 2.5-3 h coder. GPU checks: about 45 min evaluator.
  - The R9 latest freeze for c = 2 is 2026-11-03T01:14Z.
  - The merge needs a window with no running `hs` process and must precede the policy-JSON probe measurement
    (Section 4).
- **Scope.** Only OpenVLA. pi0-FAST and GR00T servers are untouched, so the generic protocol fields are refused by
  them.

## 9. Amendments after review C1–C9 (2026-11-02; they supersede conflicting text above)

### C1. Baseline and K = 1

- **The baseline sends nothing new.** The baseline arm never sends options or readouts. In the framework
  (`context/episode.py:_native_chunk`) it is a seed-0 sample with no readouts, and the decision loop is not used.
  The extension does not change that path. The text in 2.1 saying options go "in every arm" is wrong: the decision
  loop sends `streams`/`batched`/`anchor` only in the null and candidate arms.
- **K = 1 requests with options** (e.g. a gate choosing K = 1) run today's single-sample computation with no extra
  prefix:
  - sampler off: native `predict` (the recorder copy when `token_stats` is requested);
  - sampler on: one stream whose op sequence is `_decode`'s (prefix + batch-1 continuation).
- **The recorder path is reached only in the null and candidate arms**, as sample 0. New check **E1b**:
  - a toy method (`shared_prefix` + `token_stats`, K = 3, PickFirst; `ksample_toys:greedy_stats_k3`) runs one full
    null episode and one full candidate episode on development scene libero_10 t0/init0/seed7;
  - both episodes' 520 actions and observations must equal the pinned-mode identity baseline record bitwise.

### C2. `anchor: native` with stateful samplers is refused

- **Registration** (`core.method._validate_stream_options`) builds the sampler and refuses `anchor: native` when
  `init_state()` is not None (SCALE).
- **The server** refuses it again at request time.
- **Stateless decoders** may use it. Their state is not advanced by a native sample: under `stream0` there are no
  infos; under `executed` the pending entry for a native sample is `None` and commits leave the state unchanged.
- **The capability text says** that the native anchor uses stock vision attention while streams 1..K-1 use the
  sampler's prefix (e.g. SCALE's gamma).
- Tests cover both levels: part 1 (registration) and part 2 (server).

### C3. Commits are bound to a pending request

Implemented as `policy_ext/pending.py:PendingStates`, shared by the real and the fake server.

- **Response.** Every sampler request with `state_follows: executed` returns
  `{"harnessstudio": {"v": 1, "pending_id": "pN"}}`, which `protocol.response_meta` exposes.
- **The next sampler request** for the same (session, episode, sampler) must carry one of:
  - `options.commit = {"pending_id": "pN", "index": i}`: the state advances with sample i's infos;
  - `options.commit = null`: explicitly not executed from this sampler; the entry is dropped and the state is not
    advanced (documented rule).
- **Errors that raise:**
  - a missing `commit` while an entry is pending;
  - a commit with nothing pending;
  - a wrong id (stale or interleaved);
  - an out-of-range index (K changed);
  - a duplicate or retried commit.
  - A failed commit leaves the entry in place for a correct retry.
- **EPISODE_START / END** drop the episode's entries (`_drop_states` extended).
- **The decision loop** stores `{pending_id, executed_index}` on the episode context after each candidate
  decision and sends it with the next sampler request.
- **Tests:** part 1 `test_pending_commit_rules` covers gated interleaving, K changes, retries and EPISODE_END; the
  fake-stack test checks the commit sequence per episode. Part 2 checks the SCALE state against `next_state` of the
  committed sample.

### C4. One lock over the whole request; patch restoration (E10)

- **The lock.** `sample_chunks_ext` holds `_infer_lock` for the whole request: native anchor, prefix, anchor,
  streams, statistics and the vision-patch restore.
- **CPU test.** Part 2's stub model asserts the lock is held on every model call. It caught the tests' own unlocked
  calls during development.
- **New gate E10:**
  - on at least 10 predeclared development observations whose SCALE state gives gamma != 1, a native predict right
    after a shared-prefix SCALE request equals a native predict before it, bitwise;
  - every ViT attention `scale` and `forward` is restored.
  - Also covered on CPU (part 2).

### C5. E5 is a gate

E5 is a gate, and any failure blocks the merge:
- **(i)** token 1 of every stream equals the per-seed path's token 1, bitwise;
- **(ii)** sampler-off batched argmax streams agree with native greedy tokens at >= 0.99 per position, and every
  disagreement is a near-tie (top-2 logit gap < 0.05);
- **(iii)** max |logit| difference batched vs sequential per step (up to the first token divergence) <= 0.5;
- **(iv)** two server processes give bitwise-identical batched outputs (`batched_digest`).

**Sabotage controls on the CPU stub** show the gates catch real bugs:
- reversed stream rows fail (i);
- a +1.0 logit shift in the batched step fails (ii) and (iii).

Evidence: `runs/scalepp-20261102/ext/ksample/gpu/cpu_sabotage*_result.json`.

### C6. Build evidence

In `runs/scalepp-20261102/ext/ksample/`:
- **Method fingerprints:** before (live main 785515f) and after (worktree) for all 64 installed methods, equal in
  fingerprint and card JSON. Files: `fingerprints-{before,after}.json`.
- **Golden msgpack bytes** of option-less requests and responses equal; timestamp pinned. Files:
  `golden-bytes-{before,after}.json`, also a unit test.
- **Source pins:** the existing server functions' source sha256 are unchanged (unit test). `_drop_states` is
  extended on purpose.
- **pi0-FAST `default_equiv`:** the integration under `robometaharness/` is unchanged by this extension. The capture
  is rerun after any merge.
- **regress.py on the worktree:** `ok`.

### C7. Merge and probe order

1. Build, then run E1–E10 against the WORKTREE package on a leased free GPU. The dev map may keep running.
2. Merge into `main` only after:
   - the post-build diff review;
   - with no frozen non-terminal v2 Campaign idea;
   - with `regress.py --inflight` empty. The dev-map `hs eval` processes and servers must be stopped or finished.
3. Then run `measure_probe` and `--fill` on the merged `openvla_hs.py`, then spec and freeze.
4. If the merge cannot land before the planned probe time, the supervisor chooses explicitly:
   - measure on the pre-merge file and run the first Campaign without the extension, or
   - wait.
5. Never hand-edit a filled JSON without review.
6. Pre-merge dev-map records carry the `openvla_hs.py` hash `e7c6a4c3` (the live file). They count as comparable only
   if E1/E2 are bitwise.

### C8. Predeclared observation set

Declared before any GPU run: `runs/scalepp-20261102/ext/ksample/obs_set/` (`obs_set.npz` sha256 `21409571...`,
`obs_set.json`).
- **Source:** decoded frames of the pinned-mode identity episode on DEVELOPMENT scene libero_10 t0/init0/seed7
  (`eval/determinism/g7-ldlp`, MP4 sha256 in the manifest).
- **Content:** two blocks of 30 consecutive frames, steps 40–69 and 300–329, fed in order inside one policy episode
  each, so SCALE's state evolves.
- **Requirement:** at least 10 requests with gamma != 1 per block, checked by the gate.
- **Use:** E2c, E3, E4, E5, E9, E10.

### C9. E7 cost per arm, and the gated-decision cost

- **E7 reports per-decision time separately** for:
  - the candidate (SCALE sampler, `shared_prefix`, batched true/false, with and without `token_stats`);
  - the null (sampler off, same options: native predict + second prefix + K-1 argmax streams);
  - at K = 1/3/5/8, with today's per-seed SCALE and native predict as references;
  - plus peak memory.
- **R9 planning** uses the measured CANDIDATE cost. The null's cost (about +P) is reported as the overhead control's,
  conservative.
- **A gated decision** that first asks for the current step's native statistics costs T0 (a K = 1 sampler-off
  `token_stats` request, whose native action is executed when the gate does not fire). When the gate fires it
  costs T0 plus the ext request (c_ext·T0). For a gate firing on a fraction f of decisions, c_gated = 1 + f·c_ext,
  with c_ext the measured E7 candidate cost, never assumed.

### Build summary

Worktree `hs2-ext-openvla-ksample`, branch `ext2/openvla-ksample`, not merged, not committed. The diff is exported
with hashes in `runs/scalepp-20261102/ext/ksample/`.

- **Framework files:**
  - `core/capabilities.py`, `core/method.py`;
  - `policy_ext/{protocol,mixin,fake,pending*}.py`;
  - `context/{policy,episode}.py`;
  - `components/{decision_loop,streams*}.py`;
  - `policies/openvla_hs.py`;
  - `profiles/{openvla-libero-hs,fake-t2}.yaml`.
  - `*` marks new files.
- **Tests:**
  - `tests/test_openvla_ksample.py` (part 1, HarnessStudio venv, 11 tests);
  - `tests/test_openvla_ksample_server.py` (part 2, real server class on a stub model under the OpenVLA env's torch,
    `python -m unittest`, 8 tests);
  - CPU tier 0/1 of three toy methods.
- **GPU checks E1–E10:** commands in `runs/scalepp-20261102/ext/ksample/gpu/run_checks.md`.
