"""Controller, boundary detector and re-home law tests with a scripted fake context and a kinematic OSC stub (no network)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

import hs_vlm_rehome  # noqa: E402
from hs_vlm_rehome import judge  # noqa: E402
from hs_vlm_rehome.controller import VLMRehome  # noqa: E402
from hs_vlm_rehome.rehome import (BoundaryDetector, geodesic_deg, home_reached, logmap, rehome_action, rotmat,  # noqa: E402
                                  tilt_deg)

RUN = Path("/home/sejain/research/cvpr27/RoboMetaHarness-claude/runs/rtcfpp-20261101")
AMEAN = np.asarray(json.loads((PKG / "hs_vlm_rehome/assets/pi0_fast_libero_norm_stats.json").read_text())
                   .get("norm_stats", {}).get("actions", {}).get("mean")
                   or json.loads((PKG / "hs_vlm_rehome/assets/pi0_fast_libero_norm_stats.json").read_text())["actions"]["mean"])


def params(**kw):
    p = dict(hs_vlm_rehome.config().controller.params)
    p.update(kw)
    return p


# ---------------------------------------------------------------------------------------------------------------------
# geometry and law
def test_rotation_math_roundtrip_and_angles_above_pi():
    rng = np.random.default_rng(0)
    for _ in range(500):
        aa = rng.normal(size=3)
        aa *= rng.uniform(0, 3.13) / np.linalg.norm(aa)
        assert np.allclose(logmap(rotmat(aa)), aa, atol=1e-7)
    aa = np.array([3.5, 0.0, 0.0])                       # robosuite axis-angle can exceed pi (quat_no_antipodal)
    assert np.allclose(rotmat(aa), rotmat(aa / 3.5 * (3.5 - 2 * np.pi)), atol=1e-12)
    assert geodesic_deg(aa, aa / 3.5 * (3.5 - 2 * np.pi)) < 1e-6
    near = np.array([np.pi - 1e-6, 0, 0])
    assert abs(np.linalg.norm(logmap(rotmat(near))) - (np.pi - 1e-6)) < 1e-4
    assert tilt_deg([np.pi, 0, 0]) == pytest.approx(0.0, abs=1e-6)        # gripper straight down
    assert tilt_deg([np.pi / 2, 0, 0]) == pytest.approx(90.0, abs=1e-6)


class OSCStub:
    """Kinematic stand-in for robosuite OSC_POSE with control_delta: each step moves a fraction beta of the commanded goal
    offset (0.05 m * a_pos, world rotation exp(0.5 * a_rot)); beta < 1 models tracking lag."""

    def __init__(self, p, aa, beta_p=0.25, beta_r=0.128):
        self.p, self.R, self.bp, self.br = np.array(p, float), rotmat(aa), beta_p, beta_r

    def step(self, a):
        self.p = self.p + self.bp * 0.05 * np.asarray(a[:3])
        self.R = rotmat(self.br * 0.5 * np.asarray(a[3:6])) @ self.R

    @property
    def aa(self):
        return logmap(self.R)


@pytest.mark.parametrize("beta_p,beta_r", [(0.25, 0.128), (1.0, 1.0), (0.5, 0.3)])
def test_rehome_law_converges_monotonically_within_60_steps(beta_p, beta_r):
    p0, aa0 = np.array([-0.2, 0.0, 1.18]), np.array([np.pi, 0.0, 0.0])
    # 0.30 m away (and below z0: lift phase first) and 120 deg off
    k = np.array([0.3, 1.0, 0.2]) / np.linalg.norm([0.3, 1.0, 0.2])
    start_R = rotmat(k * np.radians(120)) @ rotmat(aa0)
    stub = OSCStub(p0 + np.array([0.2, -0.2, -0.1]), logmap(start_R), beta_p, beta_r)
    assert np.linalg.norm(stub.p - p0) == pytest.approx(0.3, abs=1e-9)
    pe, re_, steps = [], [], 0
    while not home_reached(stub.p, stub.aa, p0, aa0, 0.02, 10.0) and steps < 60:
        a, _ = rehome_action(stub.p, stub.aa, p0, aa0, z_first=True, kp=1.0, kr=1.0, pos_scale=0.05, rot_scale=0.5,
                             gripper_open=-1.0, lift_tol=0.02)
        assert np.all(np.abs(a[:6]) <= 1.0) and a[6] == -1.0
        stub.step(a)
        steps += 1
        pe.append(np.linalg.norm(stub.p - p0))
        re_.append(geodesic_deg(aa0, stub.aa))
    assert home_reached(stub.p, stub.aa, p0, aa0, 0.02, 10.0), (steps, pe[-1], re_[-1])
    assert steps <= 60
    assert all(b <= a + 1e-9 for a, b in zip(re_, re_[1:]))                    # rotation error monotone
    zphase = [i for i in range(len(pe))]
    assert pe[-1] < pe[0]


def test_lift_first_holds_xy_until_z0():
    p0, aa0 = np.array([0.0, 0.0, 1.0]), np.array([np.pi, 0, 0])
    a, lifting = rehome_action([0.2, 0.1, 0.9], aa0, p0, aa0, z_first=True, kp=1, kr=1, pos_scale=0.05, rot_scale=0.5,
                               gripper_open=-1.0, lift_tol=0.02)
    assert lifting and a[0] == 0 and a[1] == 0 and a[2] == pytest.approx(1.0)
    a, lifting = rehome_action([0.2, 0.1, 1.1], aa0, p0, aa0, z_first=True, kp=1, kr=1, pos_scale=0.05, rot_scale=0.5,
                               gripper_open=-1.0, lift_tol=0.02)
    assert not lifting and a[0] == -1.0 and a[1] == -1.0 and a[2] == -1.0
    a, _ = rehome_action([0.01, 0.0, 1.0], aa0, p0, aa0, z_first=True, kp=1, kr=1, pos_scale=0.05, rot_scale=0.5,
                         gripper_open=-1.0, lift_tol=0.02)
    assert a[0] == pytest.approx(-0.2)          # deadbeat in nominal OSC units inside the clip


# ---------------------------------------------------------------------------------------------------------------------
# boundary detector vs the r1-domain custody file
def test_boundary_detector_reproduces_boundary_custody():
    if not (RUN / "ideation/scripts-domain/boundary_custody.json").exists():
        pytest.skip("run data not present")
    import glob
    ref = {(b["src"], b["task"], b["init"]): b["step"] for b in
           json.load(open(RUN / "ideation/scripts-domain/boundary_custody.json"))["boundary"]}
    eps = []
    for f in sorted(glob.glob(str(RUN / "eval/dev-baseline/shards/identity-t*-i*/baseline/vla_eval/episodes/*/*.jsonl"))):
        shard = f.split("/shards/")[1].split("/")[0]
        t = int(shard.split("-t")[1].split("-")[0]); i0 = int(shard.split("-i")[1].split("-")[0])
        ep = int(Path(f).name.split("_ep")[1][:4])
        steps = [json.loads(x) for x in open(f)]
        act = np.array([s["action"] for s in steps]); st = np.array([s["state"] for s in steps])
        ds = np.arange(0, len(steps), 5); S = st[np.maximum(ds - 1, 0)]
        zc = [bool(np.all(np.abs(act[k, :6] - (AMEAN[:6] + S[j, :6])) < 1e-3)) for j, k in enumerate(ds)]
        eps.append((("dev", t, i0 + ep), S, np.sign(act[ds, 6]), zc, ds))
    d = np.load(RUN / "analysis/s2_compact.npz")
    for e in json.load(open(RUN / "analysis/s2_index.json")):
        k = e["key"]; act = d[k + "_act"]; ds = d[k + "_dstep"]
        eps.append((("s2", e["task"], e["init"]), d[k + "_state"].astype(float),
                    np.sign(act[np.minimum(ds, len(act) - 1), 6]), d[k + "_dfail"].astype(bool), ds))
    n_match = 0
    for key, S, g, zc, ds in eps:
        det = BoundaryDetector(4, 0.004, 1)
        b = None
        for j in range(len(S)):
            if det.update(float(g[j]), float(S[j, 6] - S[j, 7]), bool(zc[j])):
                b = int(ds[j])
                break
        assert b == ref.get(key), key
        n_match += 1
    assert n_match == 280 and len(ref) == sum(1 for k in ref)


def test_boundary_detector_second_boundary_and_cap():
    det = BoundaryDetector(4, 0.004, 2)
    seq = [(1, .02, False)] * 4 + [(-1, .08, False)] + [(1, .02, False)] * 3 + [(-1, .02, True)] + [(1, .02, False)] \
        + [(-1, .08, False)] + [(1, .02, False)] * 5 + [(-1, .08, False)]
    hits = [i for i, (g, w, z) in enumerate(seq) if det.update(g, w, z)]
    assert hits == [4, 10] and det.count == 2     # zero-chunk open does not end the hold; third boundary capped


# ---------------------------------------------------------------------------------------------------------------------
# fake context
class _P:
    def __init__(self, a):
        self.actions = a


class FakeCtx:
    """Scripted episode: decision j returns chunk_fn(j, state); OSC stub dynamics; width 0.02 while the last gripper
    command closed (holding) else 0.08."""

    def __init__(self, chunk_fn, replies=None, horizon=400, beta_p=0.5, beta_r=0.3, llm_delay=None, llm_exc=None,
                 instruction="put the bowl on the plate and the mug in the basket"):
        self.chunk_fn, self.replies, self.h = chunk_fn, list(replies or []), horizon
        self.stub = OSCStub([-0.2, 0.0, 1.18], [np.pi, 0.0, 0.0], beta_p, beta_r)
        self.width, self.steps, self._stopped, self.events, self.actions = 0.08, 0, None, [], []
        self.decision = 0
        self.llm_calls, self.llm_delay, self.llm_exc = 0, llm_delay, llm_exc
        self.instruction = instruction
        ctx = self

        class Policy:
            async def sample(self, k=1, obs=None, instruction=None):
                assert k == 1 and obs is None and instruction is None
                j = ctx.decision
                ctx.decision += 1
                return [_P(np.asarray(ctx.chunk_fn(j, ctx.state()), dtype=np.float32))]
        self.policy = Policy()

        async def llm(messages, operator=None, max_tokens=None):
            assert operator == "judge" and max_tokens == 64
            assert messages[1]["content"][0]["text"] == "Task: " + ctx.instruction
            ctx.llm_calls += 1
            if ctx.llm_exc is not None:
                raise ctx.llm_exc
            return ctx.replies[(ctx.llm_calls - 1) % len(ctx.replies)]
        self.tools = {"llm": llm}

    def state(self):
        w = self.width
        return np.concatenate([self.stub.p, self.stub.aa, [w / 2, -w / 2]])

    def observe(self):
        img = np.full((256, 256, 3), self.steps % 251, dtype=np.uint8)
        return {"states": self.state().astype(np.float32), "images": {"agentview": img}}

    @property
    def ended(self):
        return self._stopped is not None or self.steps >= self.h

    async def execute(self, actions):
        n = 0
        for a in np.asarray(actions):
            if self.ended:
                break
            self.actions.append(np.array(a, dtype=np.float64))
            self.stub.step(a)
            self.width = 0.02 if a[6] > 0 else 0.08
            self.steps += 1
            n += 1
        return n

    def record(self, event, **fields):
        self.events.append({"name": event, **fields})

    def stop(self, reason):
        self._stopped = reason

    def ev(self, name):
        return [e for e in self.events if e["name"] == name]


def two_stage(j, s):
    """4 close decisions (grasp), release at decision 4 (boundary at step 20), then open with a tilting motion."""
    c = np.zeros((10, 7))
    if j < 4:
        c[:, 6] = 1.0
    else:
        c[:, 6] = -1.0
        c[:, 3] = 0.6                     # tilt away after the release
        c[:, 2] = 0.3
    return c


def run(ctrl, ctx):
    asyncio.run(ctrl.run(ctx))
    return ctx


def test_vlm_flag_triggers_rehome_to_step0_pose():
    ctx = run(VLMRehome(**params(models_check=False)),
              FakeCtx(two_stage, replies=['{"state": "normal"}', '{"state": "abnormal"}'], horizon=300))
    b = ctx.ev("vpr_boundary")
    assert [x["step"] for x in b] == [20]
    checks = [e for e in ctx.ev("vpr_check") if e["checked"]]
    assert [c["step"] for c in checks] == [30, 50] and [c["verdict"] for c in checks] == ["normal", "abnormal"]
    assert checks[0]["act"] is False and checks[1]["act"] is True
    start, end = ctx.ev("vpr_rehome_start"), ctx.ev("vpr_rehome_end")
    assert len(start) == len(end) == 1 and end[0]["reached"] and end[0]["steps"] <= 60
    assert ctx.llm_calls == 2
    home = ctx.ev("vpr_home")[0]
    assert np.allclose(home["pos"], [-0.2, 0.0, 1.18], atol=1e-6)
    # re-home actions open the gripper; afterwards native decisions resume (no further checks for boundary 0)
    s_start = 50
    assert all(a[6] == -1.0 for a in ctx.actions[s_start:s_start + end[0]["steps"]])
    assert all(c["step"] < 50 or not c["checked"] for c in ctx.ev("vpr_check")[2:])
    assert checks[1]["png_sha256"] == judge.sha256(judge.image_png(np.full((256, 256, 3), 50 % 251, np.uint8)))


def native_actions(chunk_fn, horizon=300):
    ctx = run(VLMRehome(**params(models_check=False, max_calls=0)), FakeCtx(chunk_fn, replies=["x"], horizon=horizon))
    assert ctx.llm_calls == 0 and not ctx.ev("vpr_rehome_start")
    return ctx.actions


@pytest.mark.parametrize("replies,exc", [(['{"state": "normal"}'], None), (['{"state": "unsure"}'], None),
                                         (["Based on the visual evidence: abnormal"], None), (["x"], TimeoutError("timed out")),
                                         (["x"], RuntimeError("LLM provider returned HTTP 503 for /chat/completions"))])
def test_abstentions_stay_bitwise_native(replies, exc):
    ref = native_actions(two_stage)
    ctx = run(VLMRehome(**params(models_check=False)), FakeCtx(two_stage, replies=replies, llm_exc=exc, horizon=300))
    assert ctx.llm_calls == 4 and not ctx.ev("vpr_rehome_start")          # +10/30/50/80, all abstain
    assert len(ctx.actions) == len(ref) and all(np.array_equal(a, b) for a, b in zip(ctx.actions, ref))
    v = [c["verdict"] for c in ctx.ev("vpr_check") if c["checked"]]
    if exc is not None:
        assert set(v) == {"timeout" if isinstance(exc, TimeoutError) else "http_error"}
    elif "Based" in replies[0]:
        assert set(v) == {"invalid_json"}


def test_reply_after_60s_is_a_timeout_abstention(monkeypatch):
    import hs_vlm_rehome.controller as C
    import types
    clock = iter(np.arange(0, 100000, 61.0))             # each monotonic() read advances 61 s: every call lasts 61 s
    monkeypatch.setattr(C, "time", types.SimpleNamespace(monotonic=lambda: float(next(clock))))
    ctx = run(VLMRehome(**params(models_check=False)), FakeCtx(two_stage, replies=['{"state": "abnormal"}'], horizon=300))
    v = [c["verdict"] for c in ctx.ev("vpr_check") if c["checked"]]
    assert v and set(v) == {"timeout"} and not ctx.ev("vpr_rehome_start")


def test_tiltgate_ignores_verdict_and_uses_tilt():
    ctx = run(VLMRehome(**params(models_check=False, gate="tilt")),
              FakeCtx(two_stage, replies=['{"state": "abnormal"}'], horizon=300))
    checks = [c for c in ctx.ev("vpr_check") if c["checked"]]
    assert all(c["verdict"] == "abnormal" for c in checks)          # paid and recorded
    acted = [c for c in checks if c["act"]]
    assert acted and all(c["tilt_deg"] > 55 for c in acted) and all(c["tilt_deg"] <= 55 for c in checks if not c["act"])
    assert ctx.ev("vpr_rehome_start")
    # a never-tilting trajectory with 'abnormal' replies: tiltgate never re-homes, vlm does
    def flat(j, s):
        c = two_stage(j, s)
        c[:, 3] = 0.0
        return c
    t = run(VLMRehome(**params(models_check=False, gate="tilt")), FakeCtx(flat, replies=['{"state": "abnormal"}'], horizon=300))
    v = run(VLMRehome(**params(models_check=False)), FakeCtx(flat, replies=['{"state": "abnormal"}'], horizon=300))
    assert not t.ev("vpr_rehome_start") and v.ev("vpr_rehome_start") and t.llm_calls == 4


def test_forced_rehomes_at_first_eligible_check_without_calls():
    ctx = run(VLMRehome(**dict(hs_vlm_rehome.config_forced().controller.params)),
              FakeCtx(two_stage, replies=['{"state": "normal"}'], horizon=300))
    assert ctx.llm_calls == 0
    st = ctx.ev("vpr_rehome_start")
    assert len(st) == 1 and st[0]["step"] == 30


def test_budgets_calls_and_rehomes_capped():
    def many(j, s):          # a boundary every 6 decisions (4 close, open, open)
        c = np.zeros((10, 7))
        c[:, 6] = 1.0 if j % 6 < 4 else -1.0
        return c
    ctx = run(VLMRehome(**params(models_check=False, max_boundaries=10)),
              FakeCtx(many, replies=['{"state": "normal"}'], horizon=520))
    assert ctx.llm_calls <= 8
    ctx2 = run(VLMRehome(**params(models_check=False, max_boundaries=10)),
               FakeCtx(many, replies=['{"state": "abnormal"}'], horizon=520))
    assert len(ctx2.ev("vpr_rehome_start")) <= 2
    # default card: at most 2 boundaries
    ctx3 = run(VLMRehome(**params(models_check=False)), FakeCtx(many, replies=['{"state": "normal"}'], horizon=520))
    assert len(ctx3.ev("vpr_boundary")) == 2


def test_new_grasp_stops_checks_and_holding_is_never_checked():
    def regrasp(j, s):
        c = np.zeros((10, 7))
        c[:, 6] = 1.0 if (j < 4 or j >= 6) else -1.0      # release at decision 4, close again from decision 6 (step 30)
        return c
    ctx = run(VLMRehome(**params(models_check=False)), FakeCtx(regrasp, replies=['{"state": "abnormal"}'], horizon=200))
    checked = [c for c in ctx.ev("vpr_check") if c["checked"]]
    # step 30: last command open -> checked; afterwards holding (close + width 0.02) -> not checked, then stopped
    assert [c["step"] for c in checked] == [30] and ctx.llm_calls == 1


def test_identical_frames_use_the_episode_cache():
    class StillCtx(FakeCtx):
        def observe(self):
            o = super().observe()
            o["images"]["agentview"] = np.zeros((256, 256, 3), np.uint8)
            return o
    ctx = run(VLMRehome(**params(models_check=False)), StillCtx(two_stage, replies=['{"state": "normal"}'], horizon=300))
    checks = [c for c in ctx.ev("vpr_check") if c["checked"]]
    assert len(checks) == 4 and ctx.llm_calls == 1 and sum(bool(c["cached"]) for c in checks) == 3


def test_models_check_stops_on_missing_pin(monkeypatch):
    import hs_vlm_rehome.controller as C

    class Op:
        base_url, model, api_key_env = judge.BASE_URL, "qwen3-vl", "NRP_API_KEY_TEST_UNSET"

    class Tool:
        _ops = {"judge": Op()}

        async def __call__(self, *a, **k):
            return '{"state": "normal"}'

    class Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"data": self.data}).encode()

    for listed, stops in (([{"id": "qwen3-vl", "root": "Qwen/Qwen3.8-Flash-Next-FP8"}], False), ([{"id": "other"}], True)):
        monkeypatch.setattr(C.urllib.request, "urlopen", lambda req, timeout=None, d=listed: Resp(d))
        ctx = FakeCtx(two_stage, replies=['{"state": "normal"}'], horizon=60)
        ctx.tools = {"llm": Tool()}
        run(VLMRehome(**params()), ctx)
        assert (ctx._stopped is not None) == stops
        m = ctx.ev("vpr_models")[0]
        assert m["listed"] == (not stops)
        if not stops:
            assert m["entry"]["root"] == "Qwen/Qwen3.8-Flash-Next-FP8" and ctx.steps == 60


def test_card_params_and_controller_guards():
    cfgs = [hs_vlm_rehome.config(), hs_vlm_rehome.config_tiltgate(), hs_vlm_rehome.config_forced()]
    assert [c.name for c in cfgs] == ["vlm-rehome", "vlm-rehome-tiltgate", "vlm-rehome-forced"]
    for c in cfgs:
        assert c.null == "shadow" and c.trainable_parameters == 0 and c.budgets.policy_calls == 110
        assert c.controller.params["judge_sha256"] == judge.sha256((PKG / "hs_vlm_rehome/judge.py").read_bytes())
    a, t = cfgs[0].controller.params, cfgs[1].controller.params
    assert {k: v for k, v in a.items() if k != "gate"} == {k: v for k, v in t.items() if k != "gate"}
    assert cfgs[0].operators == cfgs[1].operators and cfgs[0].operators[0].extra_body == judge.PIN["extra_body"]
    assert cfgs[0].budgets.llm_calls == 8 and cfgs[0].budgets.max_tokens == 64 and cfgs[2].budgets.llm_calls == 0
    with pytest.raises(RuntimeError):
        VLMRehome(**params(judge_sha256="0" * 64))
    with pytest.raises(ValueError):
        VLMRehome(**params(vlm_timeout_s=90))
    with pytest.raises(ValueError):
        VLMRehome(**params(gate="random"))
