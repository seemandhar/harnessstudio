"""Controller slot (milestone 2): registration, shadow-mode null, candidate control, budgets, LLM operator."""
import dataclasses
import json
import os
import threading
import unittest
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from harnessstudio.core.arms import Arm
from harnessstudio.core.controller import png_bytes
from harnessstudio.core.errors import MethodContractError, RegistrationError
from harnessstudio.core.llm import LLMBudgetExhausted, OpenAICompatibleOperator, credential_names
from harnessstudio.core.method import Fidelity, MethodConfig, ModuleRef, validate
from harnessstudio.core.capabilities import caps
from harnessstudio.core.pipeline import run_episode
from harnessstudio.methods.llm_subtask_planner import parse_plan, pick_camera
from harnessstudio.verify import fixtures as fx
from harnessstudio.verify.tier1 import _policy_spec, _spec


def fake_llm(responses, max_calls=8):
    return {"provider": "fake", "responses": responses, "max_calls": max_calls, "max_tokens": 64}


class Scripted:
    """Test controller: one act per entry of `script` ((prompt, steps) pairs), optional LLM call first."""
    def __init__(self, min_act_steps, script, llm=None, ask=0):
        self.script, self.ask = script, ask

    def run(self, tools):
        for _ in range(self.ask):
            tools.llm([{"role": "user", "content": "plan"}])
        for prompt, steps in self.script:
            tools.act(prompt, steps)


def method(script, *, min_act_steps=4, llm=None, ask=0):
    params = {"min_act_steps": min_act_steps, "script": script, "ask": ask}
    if llm is not None:
        params["llm"] = llm
    return validate(MethodConfig(name="scripted", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                                 substitutions=(), required=caps("action_chunk", "seeded_sampling"),
                                 slots={"controller": ModuleRef("tests.test_controller:Scripted", params)}))


def run(m, arm, baseline=None):
    env, rows = fx.SyntheticEnv(), []
    record = run_episode(m, arm, _spec("c1"), env.observation(), env, fx.SyntheticClient(),
                         _policy_spec(caps("action_chunk", "seeded_sampling")), trace=rows.append,
                         baseline_core_traces=baseline)
    return record, rows


def dispatch(rows):
    return [r for r in rows if r["event"] == "action_dispatch"]


class RegistrationTest(unittest.TestCase):
    def test_controller_slot_registers_and_needs_min_act_steps(self):
        self.assertEqual(method([]).name, "scripted")
        with self.assertRaises(RegistrationError):
            validate(dataclasses.replace(method([]), slots={"controller": ModuleRef("tests.test_controller:Scripted", {})}))

    def test_llm_params_are_validated_and_astra_is_refused(self):
        good = {"provider": "openai_compatible", "base_url": "nrp", "model": "qwen3-vl", "api_key_env": "NRP_API_KEY",
                "max_calls": 4, "max_tokens": 32}
        method([], llm=good)
        for bad in (dict(good, model="gpt-6-astra"), dict(good, base_url="http://insecure"),
                    dict(good, max_calls=0), dict(good, api_key_env="not a name"), dict(good, provider="other")):
            with self.assertRaises(RegistrationError, msg=bad):
                method([], llm=bad)

    def test_credential_names_come_from_the_method(self):
        llm = {"provider": "openai_compatible", "base_url": "nrp", "model": "m", "api_key_env": "NRP_API_KEY",
               "max_calls": 1, "max_tokens": 1}
        self.assertEqual(credential_names(method([], llm=llm).to_json()), ["NRP_API_KEY"])
        self.assertEqual(credential_names(method([]).to_json()), [])


class ArmTest(unittest.TestCase):
    SCRIPT = [("first reach the cube", 6), ("then lift it", 20)]

    def test_null_is_shadow_mode_and_matches_baseline(self):
        m = method(self.SCRIPT, llm=fake_llm(["x"]), ask=2)
        base, brows = run(m, Arm.BASELINE)
        null, nrows = run(m, Arm.NULL, baseline=base["core_traces"])
        key = lambda rows: [(r["input"], r["action"], r["noise_seed"], r["execute_steps"]) for r in dispatch(rows)]
        self.assertEqual(key(brows), key(nrows))
        self.assertIsNone(null["divergence"])
        self.assertEqual(null["controller"]["llm"]["calls"], 2)  # null pays the controller's compute
        self.assertNotIn("controller", base)
        self.assertTrue(all(r["controller_prompt"] is None for r in dispatch(nrows)))
        self.assertEqual(null["instruction_used"], fx.INSTRUCTION)

    def test_candidate_applies_prompt_and_step_cap(self):
        record, rows = run(method(self.SCRIPT), Arm.CANDIDATE)
        d = dispatch(rows)
        self.assertEqual(d[0]["selected_k"], 6)           # act("first...", 6) -> one decision capped at 6
        self.assertIsNotNone(d[0]["controller_prompt"])
        self.assertEqual([r["selected_k"] for r in d[1:3]], [16, 4])  # 20 steps = 16 + 4 (>= min_act_steps)
        self.assertTrue(all(r["controller_prompt"] is None for r in d[3:]))  # native after run() returns
        self.assertEqual(record["steps"], 64)
        self.assertLessEqual(record["policy_calls"], record["max_policy_calls"])
        self.assertEqual(record["controller"]["acts"], 2)

    def test_step_cap_never_goes_below_min_act_steps(self):
        _, rows = run(method([("p", 18)], min_act_steps=8), Arm.CANDIDATE)
        self.assertEqual([r["selected_k"] for r in dispatch(rows)[:2]], [16, 8])  # leftover 2 rounds up to 8

    def test_llm_budget_exhaustion_ends_control_and_episode_finishes(self):
        record, _ = run(method([("p", 8)], llm=fake_llm(["x"], max_calls=1), ask=3), Arm.CANDIDATE)
        self.assertEqual(record["controller"]["stopped"], "llm_budget")
        self.assertEqual(record["controller"]["acts"], 0)
        self.assertEqual(record["steps"], 64)

    def test_adapter_without_instruction_swap_refuses_prompts(self):
        spec = dataclasses.replace(_spec("c1"), with_instruction=None)
        env = fx.SyntheticEnv()
        with self.assertRaises(MethodContractError):
            run_episode(method([("p", 8)]), Arm.CANDIDATE, spec, env.observation(), env, fx.SyntheticClient(),
                        _policy_spec(caps("action_chunk", "seeded_sampling")), trace=lambda r: None)


class PlannerTest(unittest.TestCase):
    def test_parse_plan(self):
        self.assertEqual(parse_plan('<think>hm</think> ["a", "b", "c"]', 2), ["a", "b"])
        self.assertEqual(parse_plan("1. open the drawer\n2) take the cup", 4), ["open the drawer", "take the cup"])

    def test_pick_camera(self):
        self.assertEqual(pick_camera(fx.PUBLIC_KEYS), "video.robot0_agentview_left")
        self.assertEqual(pick_camera(["observation/wrist_image", "observation/image", "prompt"]), "observation/image")

    def test_png_bytes_is_a_valid_png(self):
        import numpy as np
        data = png_bytes(np.full((1, 2, 3, 3), 7, dtype=np.uint8))
        self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
        idat = data[data.index(b"IDAT") + 4:data.index(b"IEND") - 8]
        self.assertEqual(zlib.decompress(idat), (b"\x00" + bytes([7] * 9)) * 2)


class _Provider(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *args):
        pass

    def _reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.seen.append(("GET", self.path, self.headers.get("Authorization")))
        self._reply({"data": [{"id": "pinned-model"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen.append(("POST", self.path, body["model"], body["max_tokens"]))
        self._reply({"model": body["model"], "choices": [{"message": {"content": "yes"}}],
                     "usage": {"prompt_tokens": 5, "completion_tokens": 1}})


class OperatorTest(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _Provider)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        _Provider.seen.clear()
        OpenAICompatibleOperator._verified.clear()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def params(self, model="pinned-model"):
        return {"provider": "openai_compatible", "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                "model": model, "api_key_env": "HS_TEST_API_KEY", "max_calls": 2, "max_tokens": 16}

    def test_pinned_model_budget_and_accounting(self):
        op = OpenAICompatibleOperator(self.params())
        with mock.patch.dict(os.environ, {"HS_TEST_API_KEY": "secret-value"}):
            self.assertEqual(op([{"role": "user", "content": "hi"}], max_tokens=99), "yes")
            op([{"role": "user", "content": "again"}])
            with self.assertRaises(LLMBudgetExhausted):
                op([{"role": "user", "content": "third"}])
        self.assertEqual(_Provider.seen[0], ("GET", "/v1/models", "Bearer secret-value"))
        self.assertEqual(_Provider.seen[1], ("POST", "/v1/chat/completions", "pinned-model", 16))
        self.assertEqual(len([s for s in _Provider.seen if s[0] == "GET"]), 1)
        summary = op.summary()
        self.assertEqual((summary["calls"], summary["prompt_tokens"]), (2, 10))
        self.assertNotIn("secret-value", json.dumps(op.log) + json.dumps(summary))

    def test_unlisted_model_has_no_fallback(self):
        op = OpenAICompatibleOperator(self.params(model="other-model"))
        with mock.patch.dict(os.environ, {"HS_TEST_API_KEY": "k"}):
            with self.assertRaisesRegex(RuntimeError, "no fallback"):
                op([{"role": "user", "content": "hi"}])

    def test_missing_key_is_an_infrastructure_error(self):
        op = OpenAICompatibleOperator(self.params())
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "HS_TEST_API_KEY is not set"):
                op([{"role": "user", "content": "hi"}])


if __name__ == "__main__":
    unittest.main()
