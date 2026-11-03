# EXT openvla-contrast (proposal, 2026-11-03): per-token logits of native OpenVLA decodes, for policy contrastive decoding

> **Status (2026-11-03): MERGED into `main`** (run pcd-port-20261103 task conv-merge; branch head `adc271a`, the commit the
> GPU checks (`runs/pcd-port-20261103/reports/gpu_phase1.md` 2.1: G1-G5, G7, G8 and the library gate PASS; G6 and G9
> measured) and the scored runs used). Merge identity: the adc271a package equals main at the merge commit `4f93662`;
> `harnessstudio/benchmarks/__init__.py`'s docstring was later changed by EXT simpler-seeded's S-8a, accepted as a
> docstring-only exception (`runs/pcd-port-20261103/reports/critic-merge.review.json`). History below:
> **APPROVED WITH EDITS, BUILT.** The `rmh_critic` review
> (`runs/pcd-port-20261103/reports/critic-ext-proposal.review.json`, verdict approve_with_edits, rulings: design L;
> alternative C rejected; the server file lives only in the worktree package until merge; the SIMPLER seeded protocol
> is a SEPARATE later extension) required edits R1-R9. They are applied below, each marked **[R#]**, and section 12
> lists every required edit and advisory with a pointer. Section 12 supersedes any conflicting text above.
> Built in worktree `../hs2-ext-openvla-contrast`, branch `ext2/openvla-contrast` from `main` `e4ae8b8` (committed);
> build report `runs/pcd-port-20261103/reports/ext_build.md`. GPU checks G1-G9 and the diff review are pending.

- **Author:** converter task `conv-ext-proposal` in run `pcd-port-20261103`. The agent ref is in the supervisor
  ledger `runs/pcd-port-20261103/control/research/supervision`.
- **Inputs:**
  - framework audit `runs/pcd-port-20261103/reports/framework_feasibility.md` (sha256 `5bcb4f35...`);
  - literature audit `runs/pcd-port-20261103/reports/pcd_audit.md` (sha256 `4ffa776e...`, sections 3, 4 and 9);
  - every file cited below, hashed in `runs/pcd-port-20261103/ext/proposal_source_hashes.sha256`.
- **Paper and code:** PCD, arXiv:2505.13255 (v3-v6 share the method text). Code `Koorye/PCD@31b027e`, which has no
  licence. The Apache-2.0 journal repo `Eres-lcy/PCD@00c4ac0` contains the same OpenVLA logic.

**Short version.**
- **Revised design.** The audit showed that PCD's released code does **not** decode in lockstep. It runs two
  ordinary, independent greedy decodes, then combines their per-position logits after the fact. The feasibility
  audit sketched a lockstep `contrast` request field with a `select_token_contrast` hook (its section 4); this
  proposal replaces that sketch.
- **Smallest faithful change.** The framework has to return **the per-position logits of the native greedy decode**
  (readout `logits`) from a new OpenVLA server subclass. A `null: shadow` controller can then:
  1. send the real frame and the counterfactual frame as two ordinary seeded requests. The counterfactual is a
     public-key override, which `ctx.policy.sample(obs=...)` already allows;
  2. apply the pluggable PCD formula in the method;
  3. execute the combined tokens through a bin-to-action table that the server returns.
- **Files that stay unchanged:** the protocol, the mixin, `ctx`, `core/capabilities.py`, `core/method.py` and
  `openvla_hs.py` (byte-identical).
- Section 9 records the feasibility audit's server-side contrast design as the rejected alternative C.

## 1. Mechanism to support (code-faithful, `pcd_audit.md` section 3)

For each environment step:
1. **Counterfactual frame.** A frozen perception op builds ô from the public frame o and the instruction. On the
   faithful path, per `pcd_audit.md` section 2:
   - Grounding DINO-base with box/text thresholds 0.25/0.3;
   - SAM2.1-L camera-predictor tracking, with no re-detection after the first hit;
   - Big-LaMa inpainting of `dilate(mask,5x5) & ~robot`;
   - all of this runs every step.

   Perception is method-side, a separate frozen service. **It is not part of this extension.**
2. **Stream A.** The native greedy 7-token decode on o gives per-position scores `s[t]` in R^32064, t = 0..6. Each
   position is conditioned on **A's own greedy prefix**.
3. **Stream B.** The same native greedy decode on ô gives `c[t]`, conditioned on **B's own greedy prefix**.
   - The two decodes have separate prefills and separate KV caches.
   - The combined tokens are never fed back into either decode.
4. **Combine.** For t = 0..5, `f[t] = (1+alpha)*s[t] - alpha*c[t]` with alpha = 0.8. For the gripper, `f[6] = s[6]`
   (not contrasted).
   - The token is `argmax_v f[t][v]` over the **full** 32064-entry logit vector: no plausibility cutoff, no
     temperature, no sampling (`openvla_contrast.py:53-58`).
   - The scores are **raw logits**, because greedy decoding with no logits processors returns them unchanged.
     Shifting each row by a per-position constant does not change the argmax, so raw logits and log-softmax give
     the same token here.
5. **Fallbacks [R1b].** (`openvla_contrast.py:42-58`)
   - If A has fewer than 7 tokens, A's scores become `zeros(7, 32064)`. The contrast is **still applied** when B has
     7 rows: dims 0-5 then take `argmax(-0.8*c)` = `argmin(c)`. Only the gripper (and every dim when B also has
     fewer than 7) takes `argmax(0)` = token 0.
   - If B has fewer than 7 tokens, the contrast is skipped.
   - Token 0 maps to bin **254** (the top bin), not bin 0 (`pcd_audit.md:70` says bin 0; that is wrong).
6. **De-tokenization [R1a].** `bin = clip(vocab_size - token - 1, 0, bin_centers.shape[0] - 1)` with **254** as the
   upper limit: `bin_centers` holds the 255 midpoints of `linspace(-1, 1, 256)` (`modeling_prismatic.py:500-501,504`,
   `openvla_hs.py:241-243`). Then q01/q99 un-normalization and the server's gripper mapping. Tokens
   `[0, 31743, 31744, 31745, 31746, 31999, 32000, 32063]` map to native bins `[254, 254, 254, 254, 253, 0, 0, 0]`.

On our stack these facts hold. Hashes are in `ext/proposal_source_hashes.sha256`.
- **Native decode.** vla-eval's native `predict` is exactly that greedy decode: `predict_action(do_sample=False)`
  -> `generate(max_new_tokens=7)` (`vla-eval-v0.8.0/.../model_servers/openvla.py:118-135`; remote code
  `modeling_prismatic.py:506-536` at openvla-7b `47a0ec7f`, which passes `**kwargs` to `generate` at `:518`).
- **Greedy step.** transformers 4.40.1 `_greedy_search` computes `next_token_logits = outputs.logits[:, -1, :]`,
  then `logits_processor(...)`, then `torch.argmax` (`generation/utils.py:2504-2530`).
- **Float32 logits.** Llama casts logits to float32 (`models/llama/modeling_llama.py:1228`, `logits = logits.float()`).
- **No default processors.** The checkpoints' `generation_config.json` declares only bos/eos/pad, so generate adds
  no default processors and the scores equal the raw float32 logits.
- **Vocabulary sizes.** The logit vector has `config.text_config.vocab_size` = 32064 entries. `model.vocab_size` is
  32000 (`modeling_prismatic.py:504`).

## 2. Why the existing slots, ctx and profiles cannot express it

The method side already works:
- a `null: shadow` controller can issue several `ctx.policy.sample` calls per step;
- it can override the public key `images` (`context/episode.py:267-279`, `_policy_input`);
- it can call a method-local perception service (the `roboharness` pattern, `methods/roboharness/hs_roboharness/controller.py:101,119`).

What it cannot get is the **per-position logits of a decode**:
- **(G1) No server returns logits.** `HarnessOpenVLAServer.sample_chunks` refuses every readout
  (`policies/openvla_hs.py:127-128`).
  - Sampler-off requests return only the native action (`:130-135`).
  - Sampler requests return `tokens`/`u`/`tau` extras (`:146-147`), but the client drops every extra that was not
    requested as a readout (`context/policy.py:108-113`).
  - The capability `logits_readout` and the readout name `logits` are declared (`core/capabilities.py:20`,
    `core/method.py:200`), but no server or fake implements them.
- **(G2) A sampler cannot stand in.** A server-side decoder hook sees one stream's logits and has no model access
  (`openvla_hs.py:161-164, 200-202`). It cannot see a second conditioning.
  - The sampler path is also not the native op sequence: it is a hand-written KV loop, not `generate`, so its
    argmax tokens are not guaranteed bitwise native.
- **(G3) No tier-1 coverage.** The fake server implements only the `velocity` readout (`policy_ext/fake.py:99-102`),
  so a logits-consuming method cannot pass tier 1.
  - Tier 0 requires a *registered* profile that offers the method's capabilities (`verify/tier0.py:26-28`).
  - Tier 1 picks its fake profile from `fake-t0/t1/t2` only (`verify/tier1.py:67-71`), and none of these offers
    `logits_readout`.
- **(G4) No bin-to-action map for the method.** De-tokenization needs `bin_centers` and the checkpoint's q01/q99 for
  the server's `unnorm_key` (`openvla_hs.py:238-251`). That is server-held checkpoint state. A method that copied it
  would duplicate checkpoint knowledge outside the pinned server and could drift.

Rejected workarounds:
- **Recovering logits by repeated forced-prefix requests** needs a protocol field anyway and costs 7x the
  latency.
- **Hiding logits in a sampler extra** abuses a declared readout.
- **Monkeypatching** is forbidden.
- **A policy server inside the method package** is ruled out: the server is the frozen-policy boundary and belongs
  to the framework's reviewed `policies/`, and tier 0 needs a registered profile in any case.

## 3. Smallest change (design L: native logits readout)

All other framework files stay as they are, in particular:
- `policy_ext/protocol.py` and `policy_ext/mixin.py`;
- `context/*`;
- `core/*`;
- `components/*`;
- `policies/openvla_hs.py`, which stays **byte-identical**: Run X's pinned policy JSON names it as its server
  script (`robometaharness/integrations/harnessstudio_v2/policies/openvla-libero10.json:16`).

### 3.1 New file `harnessstudio/policies/openvla_logits_hs.py`

```python
class HarnessOpenVLALogitsServer(HarnessOpenVLAServer):
    harness_capabilities = HarnessOpenVLAServer.harness_capabilities | {"logits_readout"}

    def sample_chunks(self, obs, seeds, sampler, readouts, ctx):
        if "logits" not in readouts:                       # [R2] every non-logits request: parent code, unchanged
            return super().sample_chunks(obs, seeds, sampler, readouts, ctx)   # (incl. its own readout refusal)
        if tuple(readouts) != ("logits",):
            raise ValueError("the 'logits' readout cannot be combined with other readouts")
        if sampler:
            raise ValueError("the 'logits' readout is defined for the native greedy decode only (sampler None)")
        with self._infer_lock:
            out = []
            for seed in seeds:                             # greedy: identical per seed, compute paid per seed
                action, rows, tokens = self._recorded(obs)  # rows float32 [T, 32064]; tokens = GENERATED ids
                # [R3] self-check inside _recorded: argmax(rows) == generated tokens at EVERY position (a
                # StoppingCriteria recorder sees input_ids after each append), and _actions(tokens) == native action
                # when T == get_action_dim(unnorm_key) [A4]; any mismatch raises RecordedDecodeMismatch
                out.append((action[None, :], {"seed": int(seed),
                            "logits": rows, "logits_tokens": tokens.astype(np.int32),
                            "logits_bin_actions": self._bin_table(),          # float32 [256, 7]
                            "logits_token_end": int(self._model.vocab_size),  # 32000
                            "logits_n_bins": int(self._model.config.n_action_bins)}))
            return out
```

**`_predict_recorded(obs)`** is a copy of vla-eval's `OpenVLAModelServer.predict` (`openvla.py:118-135`) with two
differences only:
- it passes `logits_processor=LogitsProcessorList([rec])`, where `rec` is a recorder;
- it returns `(action, rec.rows())`.

The recorder (`transformers.LogitsProcessor`):
- stores `scores[0].detach().clone()` on the device at each call;
- **returns `scores` unchanged** (the same tensor object);
- stacks the rows and copies them to the host once after `generate` (one device-to-host transfer of about 0.9 MB).

This is the ksample recorder pattern (`docs/extensions/openvla-ksample.md`, 2.4).

**Bin table.** `_bin_table()` is `self._actions(np.tile(tok[:, None], (1, action_dim)))` with
`tok = vocab_size - 1 - arange(n_bins)`:
- it applies the server's own de-tokenization, un-normalization and gripper map (`openvla_hs.py:238-251`)
  element-wise to every bin;
- it is computed once per server and cached.

Because `_actions` is element-wise, `table[clip(vocab_size - t_d - 1, 0, 255), d]` equals `_actions(t)[d]` bitwise
for every token vector t, including out-of-range tokens (unit test U5).

**Readout extras** follow the companion-key convention (`context/policy.py:110-111`). Every name starts with
`logits`, and every extra is a function of the public inputs and the frozen weights only:

| key | dtype / shape | meaning |
|---|---|---|
| `logits` | float32 `[T, 32064]` | raw scores of each generated position (T = 7 unless EOS came early) |
| `logits_tokens` | int32 `[T]` | greedy tokens = native tokens |
| `logits_bin_actions` | float32 `[256, 7]` | executable action value of each bin and dimension |
| `logits_token_end` | int | `model.vocab_size` (32000), for `bin = clip(end - token - 1, 0, n_bins - 1)`; exact only because every table row is produced by `_actions` (row 255 = token 31744 equals row 254 by construction) [R1a] |
| `logits_n_bins` | int | 256 |

**Request routing:**
- **Requests without readouts** take the parent code: the baseline arm (`context/episode.py:_native_chunk`), the
  shadow null's executed native decisions, SCALE and every other existing method.
- **Native vla-eval requests** with no `harnessstudio` key also take the parent code (`policy_ext/mixin.py:28-31`).
- **Logits together with a sampler** is refused. Logits of the SCALE/hook path would be a separate extension.

**Launch.** Use the same YAML server args and pinned serving mode as `openvla_hs.py` (cuBLASLt pin,
P-20261103-01), with only the `script:` path changed. The startup probe must reproduce `23d4ad8ab4182245`: the
native path is the same code.

### 3.2 New profile `harnessstudio/profiles/openvla-libero-logits.yaml`

```yaml
# harnessstudio.policies.openvla_logits_hs (HarnessOpenVLAServer + the native-decode 'logits' readout).
capabilities: [act, chunk, seeded_sampling, isolated_streams, decoding_hooks, logits_readout]
chunking: harness
execute_horizon: 1
```

`openvla-libero-hs.yaml` is unchanged. It must not advertise `logits_readout`, because its server lacks it.

### 3.3 Fake path (tier 1)

- **`policy_ext/fake.py`, `FakePolicyServer.sample_chunks`:** when `"logits" in readouts`, also return:
  - deterministic synthetic `logits` float32 `[action_dim, 320]`, from `rng([seed, digest(obs), 2])`;
  - `logits_tokens` = argmax;
  - a constant `logits_bin_actions` float32 `[256, action_dim]`, from `rng([3])`;
  - `logits_token_end` = 320 and `logits_n_bins` = 256.

  The returned chunk stays `self.chunk(obs, seed)`, so **a readout never changes actions** (test U6).
  The seeded fake's `harness_capabilities` gains `logits_readout`. Nothing reads it today; the change keeps the
  declaration honest.
- **New `profiles/fake-t3.yaml`:** the `fake-t2` capabilities plus `logits_readout`.
- **`verify/tier1.py:68`:** the selection list becomes `("fake-t0", "fake-t1", "fake-t2", "fake-t3")`.
  - Methods satisfied by t0-t2 keep their profile, because the loop returns the first match.
  - Test U7 proves this for every installed method: profile chosen before == after.

### 3.4 Docs and tests

- `docs/writing-a-method.md` gets a section, "Per-token logits of the native OpenVLA decode (`logits` readout;
  contrastive decoding recipe)". It covers the extras table, the bin-table rule and the shadow-null cost.
- The spec status section names the first server implementing `logits_readout`.
- `docs/extensions/README.md` gets a backlog line.
- Tests are in section 6.

### 3.5 File list (the whole diff)

| file | change |
|---|---|
| `harnessstudio/policies/openvla_logits_hs.py` | NEW |
| `harnessstudio/profiles/openvla-libero-logits.yaml` | NEW |
| `harnessstudio/profiles/fake-t3.yaml` | NEW |
| `harnessstudio/policy_ext/fake.py` | `logits` readout branch, plus the capability name |
| `harnessstudio/verify/tier1.py` | one tuple entry |
| `tests/test_openvla_logits.py` | NEW (part 1, HarnessStudio venv) |
| `tests/test_openvla_logits_server.py` | NEW (part 2, OpenVLA env torch, stub model) |
| `tests/toys/logits_contrast_toy.py` | NEW toy `null: shadow` controller, loaded as `module:attr` |
| `docs/writing-a-method.md`, `docs/extensions/README.md`, spec status, this file | docs |

## 4. How a method uses it (PCD; a separate method package after the merge, not part of this extension)

```python
# controller, one decision per step (execute_horizon 1), null: shadow
cf = await perception.counterfactual(frame, ctx.instruction, episode_token)   # method-local frozen service; tool_calls += 1
key = next(iter(obs["images"]))                                                 # the image OpenVLA reads (openvla.py:88-89)
[real] = await ctx.policy.sample(1, readouts=["logits"])
[cfp]  = await ctx.policy.sample(1, obs={"images": {**obs["images"], key: cf}}, readouts=["logits"])
tokens, stats = combine(real.extras["logits"], [cfp.extras["logits"]], **frozen_params)   # pluggable, code-faithful default
action = table_lookup(real.extras, tokens)                                      # bitwise == server _actions(tokens)
ctx.record("pcd", **stats); await ctx.execute(action[None, :])
```

- **The default combine is `pcd_code`, the code-faithful mode.** It applies `(1+alpha)*s - alpha*c` to the six
  motion dimensions (`contrast_dims = [0..5]`), keeps the gripper un-contrasted, takes a full-vocabulary float32
  argmax, and applies the code's fallbacks (section 1.5).
  - In float32, `a*s - b*c` evaluates identically in numpy and torch eager: each is a separate IEEE-rounded
    element-wise op, so no FMA is involved.
  - CPU parity against the authors' torch expression must show 0 differences.
- **Optional formulas** are named, frozen, method-side functions disclosed in `paper.yaml`, never a default:
  - `paper_eq2_logsoftmax`, which gives the same argmax as `pcd_code`;
  - `kde_keep_threshold` (only Octo/pi0 use it).

  This extension has no lockstep mode. A lockstep dual-condition decode is a different mechanism
  (`pcd_audit.md` section 4, item 1), and it would need the alternative-C server hook.
- **More than one counterfactual** costs one more request each: `combine` takes a list. PCD uses one.
- **Activation statistics** are computed in the method from both logit tensors: per step, the number of dimensions
  whose token changed, the skip flag when B has fewer than 7 tokens, the detection-failure flag, and logit margins.
  They are recorded with `ctx.record`, which closes feasibility gap G3.
- **Detection failure.** The method still sends the second request with ô = o, as the code does (its cost and
  behaviour are both kept), and it counts those steps.

## 5. Null-arm semantics and cost accounting

**The PCD method is a controller with `null: shadow`** (`core/method.py:186-188`). Per step:

| arm | perception | requests on the method's session | executed action | GPU decodes/step |
|---|---|---|---|---|
| baseline | none | none (framework: 1 native request, no readouts) | native greedy | 1 |
| candidate | yes (tool_calls +1) | 2 `logits` requests (o, ô) | `table[argmax f]` | 2 |
| null | yes, on its own tracker session | the same 2 `logits` requests on the **shadow** session (`context/episode.py:133-137`) | native: `execute()` substitutes `_native_action()` (`episode.py:216-217, 300-303`), a plain no-readout request on the primary session, i.e. the baseline code path | 3 |

- **Paying the cost.** The null pays everything the candidate pays: perception, both recorded decodes and the
  readout transfer. On top of that it pays one native decode, the framework's shadow replay (the same accepted cost
  as P-20261101-19). The overhead control is therefore conservative and disclosed.
- **Bitwise native actions in the null.** They come from the parent `sample_chunks` path. This is proved on the fake
  stack (tier 1) and on the GPU (check G4).
- **Accounting fields** (`context/episode.py:85-97`):
  - `method_policy_samples` = 2 per step, in both candidate and null;
  - `shadow_policy_samples` = 2 per step in the null;
  - `native_policy_samples` = 1 per step in baseline and null;
  - `tool_calls`: perception calls, counted by the method;
  - `policy_seconds` / `method_seconds` as usual.
- **Budget.** `budgets.policy_calls` >= 2 x horizon (libero_goal 300 -> 600 + margin), and `tool_calls` >= horizon.
- **CRN.** The candidate's two requests per step advance `ctx._decision` twice (P-20261101-19). The OpenVLA
  sampler-off path ignores seeds (`openvla_hs.py:132`), so this cannot change an action. It is disclosed.
- **Planning cost.** From measured T0 = 0.29-0.32 s per native request (`openvla-ksample.md` section 5):
  - candidate c_pol ≈ 2.05 (2 decodes + about 1.8 MB/step of float32 logits over localhost msgpack + one PNG
    encode of ô);
  - null c_pol ≈ 3.05;
  - perception comes on top. The author-reported total is 0.86 -> 1.77 s/step (paper Table 5).
  - These are planning numbers. GPU check G6 measures the real ones.

## 6. Compatibility claim and tests

**Claim.**
- **identity / native stays bitwise.** No request without the `logits` readout reaches new code, and
  `openvla_hs.py` is byte-identical.
- **Every installed method's tier 0/1 is unchanged.** No core, protocol, ctx or component file changes; the fake
  changes only under the new readout; the tier-1 profile choice is unchanged.
- **Existing parity scripts still pass:** scale, scale_gtok, greedy_escape, fork30 and gpp use `openvla_hs.py` or
  fake-t2, both untouched.
- **Method fingerprints are equal before and after** (ksample C6 practice).

**CPU unit tests**, run in the worktree as part of `regress.py`:
- **U1 Source pins.**
  - `openvla_hs.py` sha256 == `e7c6a4c302fece60...`.
  - `git diff --stat main` lists only the files in 3.5.
  - The AST hashes of `HarnessOpenVLAServer.{predict, sample_chunks, _decode, _inputs, _actions}` are unchanged.
- **U2 Routing** (part 2, stub model).
  - A no-readout request calls the parent `sample_chunks` with identical arguments (spy) and returns identical
    bytes.
  - A native observation goes to `predict`.
  - `logits` plus a sampler is refused, and so is an unknown readout.
  - The `_infer_lock` is held on every model call.
- **U3 Predict copy.** The AST diff between `_predict_recorded` and vla-eval's `OpenVLAModelServer.predict` is only
  the `logits_processor` kwarg and the return.
- **U4 Recorder** (stub `generate` that runs the processor list, then argmax).
  - The rows equal the tensors argmax saw.
  - The recorder returns its input object unchanged (identity check).
  - The T < 7 path (EOS) returns ragged rows without a self-check error.
- **U5 Bin table.** For 10^4 random token vectors, including tokens 0, 31743, 31999, 32000 and 32063:
  `table[clip(end - t - 1, 0, 255), d]` == `_actions(t)` bitwise.
- **U6 Fake.**
  - The `logits` extras are deterministic in (obs, seed) and absent without the readout.
  - Chunks are bitwise identical with and without the readout.
  - The client filter passes `logits_*` companions only when `logits` is requested.
- **U7 Profiles.**
  - `default_profile(config)` before == after for every installed method (the result is stored in
    `ext/profile-choice-{before,after}.json`).
  - `openvla-libero-logits` passes `capabilities.check`.
  - Tier 0 accepts a method requiring `logits_readout`.
- **U8 Toy method.** `tests/toys/logits_contrast_toy.py:toy` is a `null: shadow` controller that sends 2 `logits`
  requests per step, the second with an `images` override.
  - Tier 0/1 pass on `fake-t3`.
  - null == baseline bitwise.
  - The accounting counts are as in section 5.
- **U9 Sabotage, proving the checks bite (P-20261031-21).** Each of these must make its test fail:
  - a recorder that adds +1 to one logit (fails U4 and the server self-check);
  - an off-by-one bin table (fails U5);
  - a fake whose readout perturbs the chunk (fails U6);
  - a subclass that routes no-readout requests to `_predict_recorded` (fails U2/U3).

  Evidence goes to `runs/pcd-port-20261103/ext/sabotage/`.

**Method-side parity (later, in the PCD package, not a gate for this extension):**
- `combine(pcd_code)` against the authors' expression, re-derived (no licence on Koorye) or taken from Eres-lcy
  (Apache-2.0), on randomized 32064-wide float32 rows: 0 token differences, and bitwise-equal `f`.
- alpha = 0 gives exactly `argmax(s)`.

## 7. CPU regression gate and testing without touching the shared venv

Run X processes import `harnessstudio/.venv`. **Never `pip install -e` / `uv pip install` the worktree or the toy
into it, and never `uv run` / `uv sync` the OpenVLA env.**

1. **Main gate.** Use the live script with the shared interpreter, read-only, through a PYTHONPATH override (what
   `regress.py:env_for` already does, `scripts/regress.py:63-68`):
   ```bash
   cd /home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio
   PYTHONDONTWRITEBYTECODE=1 python3 scripts/regress.py --source ../hs2-ext-openvla-contrast \
     --python /home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio/.venv/bin/python \
     --out ../hs2-ext-openvla-contrast/regress.json        # must report "ok": true
   ```
   - The worktree and its `methods/*` come first on `sys.path`, so they shadow the editable installs.
   - The venv's site-packages are only read.
   - `PYTHONDONTWRITEBYTECODE=1` keeps `.pyc` files out of shared directories. Pass `--python` explicitly, so a
     stray worktree `.venv` can never be picked up (`regress.py:28-37`).
   - The toy method is not an entry point. As built it is `tests/toys/hs_ext_logits_toy.py`, loaded as
     `hs_ext_logits_toy:toy` with `tests/toys` on `sys.path` (module paths work: `core/registry.py:18-22`); its
     tier 0/1 runs inside pytest. No `tests/__init__.py` was added (A11 deviation, section 12), so the existing
     test collection is unchanged.
2. **Part-2 server tests.** Run with the pinned OpenVLA interpreter, read-only:
   ```bash
   # [R6] no GPU visible, scratch offline HF caches, server built with __new__ + stub _model/_processor (never __init__)
   cd ../hs2-ext-openvla-contrast && env -u TRANSFORMERS_CACHE -u HF_HUB_CACHE CUDA_VISIBLE_DEVICES= \
     PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     HF_HOME=$RUN/ext/build/hf-scratch HF_MODULES_CACHE=$RUN/ext/build/hf-scratch/modules PYTHONPATH=$PWD nice -n 19 \
     /home/sejain/.cache/uv/environments-v2/openvla-hs-73218eaab6c13068/bin/python -B -m unittest tests.test_openvla_logits_server -v
   ```
   - Before and after, hash the env's `pyvenv.cfg`, its `.pth` files, the interpreter binary, the `verify_files` of
     the policy JSON (`openvla-libero10.json`), its `env_manifest.dists_sha256` (recomputed with
     `server.dists_digest`) and its `frozen_sources` / `frozen_source_trees` [R6]; plus the live checkout and the
     shared venv [R5b]. Script `runs/pcd-port-20261103/ext/integrity.py`; evidence
     `runs/pcd-port-20261103/ext/build/integrity-{before,after}.json`; they must be equal.
   - The stub model needs no weights and no GPU.
3. **Before/after evidence** goes to `runs/pcd-port-20261103/ext/`:
   - method fingerprints and card JSON for all installed methods;
   - tier-1 profile choice;
   - `regress.py --inflight` snapshots.
4. **Fallback,** only if some test needs an installed entry point: a throwaway venv under
   `runs/pcd-port-20261103/ext/venv`. Create it with `uv venv`, then install non-editable copies of vla-eval, the
   worktree and the methods into it. Pass it with `--python`. Never use the shared venv.
5. **Commit inside the worktree branch** (`ext2/openvla-contrast`) with the message naming the EXT ID. ksample left
   its edits uncommitted, which weakened provenance.

## 8. GPU checks (sole evaluator; [R7] only after Run X closes, or after a GPU handoff that the supervising session logs in `runs/scaleagent-shared/COORDINATION.md`, on a leased process-free GPU among 4-7)

These checks need one leased, process-free GPU among 4-7. They run against the **worktree** package:
- the server is launched with the pinned OpenVLA interpreter, `PYTHONPATH=<worktree>`, and the policy JSON's env and
  args with only `script` changed;
- the harness uses the shared `hs` interpreter with `PYTHONPATH=<worktree>`.

The predeclared observation set is `runs/scalepp-20261102/ext/ksample/obs_set/obs_set.npz` (sha256 `21409571...`):
development frames of libero_10 t0/init0/seed7, inside the init 0-9 allowed by Run X. For libero_goal, the
evaluator predeclares an extra development set before running.

| id | check | pass rule |
|---|---|---|
| G1 | identity baseline, one development episode (libero_10 t0/init0/seed7) on the new server; startup probe | all action/obs hashes equal the pinned-mode record (`runs/scalepp-20261102/eval/determinism/g7-ldlp`); probe `23d4ad8ab4182245` |
| G2 | recorder on 60 obs-set frames: `sample(readouts=[logits])` vs a plain `sample` | actions bitwise equal; `logits_tokens` = native tokens; table action = native action; no self-check errors |
| G3 | the same frame twice, and an ô = o copy | logits bitwise equal across requests (determinism) |
| G3b [R4] | a fixed visibly different ô (centre 96x96 grey) on every obs-set frame; one alpha 0.8 toy development episode | logits differ from o's on every frame; action == a plain native request on ô; the toy episode reports changed dims > 0 |
| G4 | the toy (and later PCD) `--arms null`, one development episode | bitwise equal to baseline (class M); shadow logits requests counted |
| G5 | PCD with alpha = 0 as the candidate, one development episode | bitwise equal to baseline (`1*s - 0*c == s` exactly) |
| G6 | timing and memory | c_pol of candidate and null, logits bytes/step, peak memory, perception latency; replaces the section 5 estimates |
| G7 | `hs verify identity --tier 2` on the new profile; tier 2 of the toy | null vs baseline M |
| G8 | two server processes on the obs set | bitwise-identical `logits` digests |
| G9 [R9] | ô = o with alpha 0.8 on the obs set; gap histograms | measurement: argmax flips of `(1+a)*s - a*s` vs `s`, top1-top2 gaps of s and f (no pass/fail) |

Scored PCD runs (subset, validation, Campaign) start only after the merge. A Campaign freeze hashes the merged
package, not a worktree. A pinned policy JSON for `openvla_logits_hs.py` (`frozen_sources` = both server files) is
an edit under `robometaharness/integrations/harnessstudio_v2/policies/`, so COORDINATION rule 2 applies to it too.
A run-local JSON path is the alternative if the spec accepts one; that is open point 4.

## 9. Alternative C, considered and not proposed: a server-side contrast request

This is the feasibility audit's section 4 design, corrected for independent prefixes:
- a `contrast` request field with public-key overrides;
- `ctx.policy.sample(contrast=...)`;
- a capability `contrastive_decoding`;
- a server that runs native recorded decodes for o and each ô and calls a decoder hook
  `combine_contrast(real_rows, [cf_rows])` (lockstep reserved and refused);
- a `contrast_stats` readout.

Its advantages are one round trip and about 7 KB instead of about 1.8 MB per step.

It is not proposed because it touches six more framework files that running Run X processes import:
- `core/capabilities.py`, `core/method.py`;
- `policy_ext/protocol.py`, `policy_ext/mixin.py`;
- `context/policy.py`, `context/episode.py`.

It also has these drawbacks:
- it overlaps four of the files edited by the unmerged ksample branch;
- it puts the formula in a server-imported `hs_` module, which needs the method on the server's PYTHONPATH and
  hashes paper logic into the policy process;
- it adds a protocol surface that needs golden-bytes tests;
- its performance benefit is negligible against 2 x 0.3 s decodes plus perception.

If G6 shows the transfer cost is material (more than 5% of the step), alternative C can be proposed later as a new
extension on top of this one.

## 10. Merge plan (EVOLUTION step 6; `runs/scaleagent-shared/COORDINATION.md` rule 2)

1. Critic review of this proposal, then build, CPU tests and `regress.py` ok, all in the worktree.
2. The evaluator runs G1-G8 against the worktree. This needs no merge.
3. An independent diff review covers the diff, `regress.json`, the sabotage evidence and the GPU results.
4. **Merge into `main` only when all of the following hold:**
   - **(a) No frozen non-terminal Campaign idea in any run.** Check with `python -m robometaharness status` for
     Run X `runs/scaleagent-codex-20261103` and every other v2 run, read-only, and in Run X's PROGRESS.md.
   - **(b) `scripts/regress.py --inflight` is empty.**
   - **(c) No reviewer objection is open.**
   - **(d) A log line is appended to `runs/scaleagent-shared/COORDINATION.md`** before merging (time from
     `date -u`, run, "merge ext2/openvla-contrast", commit).

   **[R8]** The unmerged ksample branch edits much more than `fake.py` and the profile list: also
   `policies/openvla_hs.py` (this subclass's parent), `profiles/openvla-libero-hs.yaml`, `profiles/fake-t2.yaml`,
   `policy_ext/{protocol,mixin}.py` and `context/{policy,episode}.py` (`openvla-ksample.md` build summary). If ksample,
   or any other change to `openvla_hs.py`, lands first, this extension needs a **new review, new U1/U3 pins and a
   re-run of G1-G9**, not only a rebase.
5. After merging:
   - `regress.py` on `main`;
   - an `openvla-libero-logits` startup probe;
   - a FRAMEWORK.md note through `skill_evolve.py`;
   - a README backlog line, "done".

   The PCD method package then uses the new hook as an ordinary method.
6. **Nothing touches Run X** (its processes, leases, files or tmux session `scaleagentx`). While its evaluations run,
   nothing in the live checkout changes.

## 11. Risks and open points

1. **Recorder side effects on `generate`.** In 4.40.1 a non-empty processor list is applied as one call on the same
   tensor. G2 is the gate.
2. **Wire size** is 1.8 MB/step. msgpack has `max_size=None` on both ends (`vla_eval/model_servers/serve.py:327`,
   `connection.py:202`), and float32 arrays are encoded lossless as raw bytes (`protocol/numpy_codec.py:47-53`). The
   cost is measured in G6.
3. **Numeric stack [R9].** Measured by G9 (section 12), not only disclosed. The method card discloses:
   transformers 4.40.1, torch 2.10.0, numpy 2.4.2 (ours) vs 4.44.2 / 2.3.1 (authors); SDPA Llama attention vs
   upstream eager (`openvla_model.py:45`); the cuBLASLt 12.6 pin; LANCZOS + jpeg_roundtrip + center_crop vs cv2
   INTER_AREA; the OpenVLA prompt template vs the raw instruction. It claims **no token-level replication** of the
   authors' decode.
4. **Pinned policy JSON for the new script.** Location: the integration directory, after the merge, or a run-local
   path. The supervisor decides.
5. **Prompt.** The authors' SIMPLER wrapper uses the raw instruction. Our LIBERO server uses the OpenVLA template
   (`openvla_hs.py:173`), identically in every arm. This is a method-card disclosure, not an extension issue.
6. **Scope.** OpenVLA only. The other servers, `openvla_hs.py` and its profile do not change. A SIMPLER contract and
   base `openvla-7b` weights are separate tasks (feasibility section 2).

## 12. Amendments after review R1-R9 and advisories A1-A12 (2026-11-03; they supersede conflicting text above)

Review: `runs/pcd-port-20261103/reports/critic-ext-proposal.review.json` (sha256 `4a60a6a3...`), verdict
approve_with_edits. Build: worktree `hs2-ext-openvla-contrast`, branch `ext2/openvla-contrast`, commits on top of
`main` `e4ae8b8`; build report `runs/pcd-port-20261103/reports/ext_build.md`.

| item | what changed | where (code / tests / doc) |
|---|---|---|
| **R1** decode spec | native clip at **254** (255 centres); table row 255 == row 254 by construction; index rule exact only because rows come from `_actions`; A-short: contrast still applied when B has 7 rows (dims 0-5 -> argmin c); token 0 -> bin 254; U5 adds 31744/31745/31746 | section 1 items 5-6, extras table; `openvla_logits_hs.py` module docstring; `tests/test_openvla_logits.py::test_bin_table_equals_actions_bitwise`, `::test_boundary_tokens_map_like_native_clip`; toy `pcd_code_combine` (A-short zeros, B-short skip) |
| **R2** routing | `if "logits" not in readouts: return super().sample_chunks(...)`: every non-logits request (incl. other readouts and the parent's byte-identical refusal) runs the parent; `logits` + other readout or + sampler refused | `HarnessOpenVLALogitsServer.sample_chunks`; `::test_requests_without_logits_run_the_parent_unchanged` (spy, readouts `()`, `velocity`, `tokens`, `logit`), `::test_parent_refusal_message_is_unchanged`, `::test_logits_refusals`, `::test_sabotaged_routing_is_caught`; part 2 `test_no_readout_and_other_readouts_take_the_parent` |
| **R3** self-check | token level at **every** position: a `StoppingCriteria` recorder sees `input_ids` after each append (all T generated ids, the gripper token included; stronger than the reviewer's input_ids[0,-1]-at-calls-1..T-1 which misses the last token); `argmax(rows) == generated` exactly, and `_actions(tokens) == native action` when T == action dim; float32 enforced; mismatch raises `RecordedDecodeMismatch` | `_recorders`, `_recorded`, `_self_check`, `_rows_numpy`; part 2 `TestSelfCheckSabotage`: gripper row perturbed to a same-sign token (action check blind) -> caught; rows shifted to position t+1 -> caught; bf16 rows -> refused |
| **R4** override positive control | G3b (fixed visibly different ô: logits differ on every frame; action == plain native request on ô) + one alpha 0.8 toy development episode with changed-dim count > 0 (G4b); tier 1: the fake's received sha256 contains the fingerprint of the override observation | `ext/gpu/gpu_checks.py` (G3b); `::test_override_reaches_the_policy_and_changes_tokens` (SpyFake, received-hash assertion, changed_dims > 0); part 2 `test_override_positive_control` |
| **R5** regress prerequisites | untracked asset `methods/tube_steer_vfb/hs_tube_steer_vfb/assets/visual_memory.npz` copied sha-checked, uncommitted (lesson 20261102T170428Z); 'before' regress on the pristine worktree at `e4ae8b8`; after == before per method; live checkout + shared venv integrity before/after | `ext/build/{copied-untracked-assets.sha256, live-untracked-methods.txt, regress-before.json, regress-after.json, integrity-before.json, integrity-after.json}`; `ext/integrity.py` |
| **R6** part-2 hardening | `CUDA_VISIBLE_DEVICES=` (asserted by the test), scratch offline `HF_HOME`/`HF_MODULES_CACHE` with `TRANSFORMERS_CACHE`/`HF_HUB_CACHE` unset (asserted), server via `__new__` + stub `_model`/`_processor`; integrity covers the policy JSON's `env_manifest.dists_sha256` (recomputed), `verify_files`, `frozen_sources`, `frozen_source_trees` | section 7 item 2 command; part 2 `TestEnvironment`; `ext/integrity.py` |
| **R7** GPU authority | GPU checks only after Run X closes or after a COORDINATION-logged GPU handoff, on a leased process-free GPU in 4-7; G1 on GPU 7 against `g7-ldlp`, or a fresh same-session `openvla_hs.py` identity reference on the same GPU (`start_server.py --live`) | section 8 heading; `ext_build.md` GPU plan |
| **R8** ksample overlap | ksample also edits `openvla_hs.py`, `openvla-libero-hs.yaml`, `fake-t2.yaml`, `protocol.py`, `mixin.py`, `policy.py`, `episode.py`: landing first => new review, new U1/U3 pins, re-run G1-G9 | section 10 item 4 |
| **R9** ties | G9 (ô = o, alpha 0.8: argmax flips of `(1+a)*s - a*s` vs `s` on the 6 contrasted dims; top1-top2 gap histograms of s and f); combine evaluated exactly as `(1 + alpha) * s - alpha * c` (Python-double alpha, float32 arrays, that order; `s + alpha*(s-c)` is not bitwise equal); card disclosure list; no token-level replication claim | section 11.3; `ext/gpu/gpu_checks.py` (G9); toy `pcd_code_combine` |

Advisories:
- **A1** cost per arm: decodes per step baseline 1, candidate 2, null 3. PCD's overhead is reported from the
  candidate's measured cost only. `accounting.policy_seconds` counts native decodes only, so G6 also reports wall
  seconds per step per arm and in-process seconds per plain/logits request.
- **A2** toy budget `policy_calls` 1100 (libero_10 520 x 2 + margin). PCD: >= 2 x horizon.
- **A3** fake logits depend on the observation only (`fake.fake_logits`; `::test_fake_logits_readout`).
- **A4** uses `self._model.get_action_dim(self.unnorm_key)`.
- **A5** method side (perception token from the method, per-arm tracker sessions, never short-circuit ô = o):
  recorded for the PCD method package; not part of this extension.
- **A6** run-local policy JSON requirements recorded for the method task (per-checkpoint probe, expected_libraries,
  frozen_sources incl. both server files and vla-eval openvla.py, merged profile path).
- **A7** optional unpinned cuBLASLt 12.8 tie proxy: listed as optional G9b in the GPU plan.
- **A8** codec round trip test `::test_logits_round_trip_through_vla_eval_codec`; wire bytes in G6.
- **A9** regress under `nice -n 19 ionice -c3`.
- **A10** the supervising session checks Run X's freeze/terminal status from idea-status metadata only; PCD agents
  do not open `runs/scaleagent-codex-20261103`.
- **A11** part 2 asserts recorded rows are float32 and bitwise equal to the stub forward's
  `outputs.logits[:, -1, :]` (`test_native_action_unchanged_and_rows_exact`). Deviation: no `tests/__init__.py`
  (it would change how pytest imports every existing test module); the toy is loaded from `tests/toys` on
  `sys.path` instead.
- **A12** no fidelity claim; a LIBERO PCD method stays `adapted`.

Merge note: the live checkout holds this file untracked while the branch adds it. Before merging, the merger checks
that both copies have the same sha256 and moves the untracked live copy aside (never deletes it), otherwise
`git merge` refuses to overwrite it.

GPU checks G1-G9 (exact commands, expected bitwise results, GPU and time): `runs/pcd-port-20261103/reports/ext_build.md`
section "GPU check plan"; tooling in `runs/pcd-port-20261103/ext/gpu/`.
