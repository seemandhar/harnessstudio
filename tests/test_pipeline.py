import unittest

import tests  # noqa: F401
from harnessstudio.core.arms import Arm
from harnessstudio.core.capabilities import caps, unified_names
from harnessstudio.core.errors import MethodContractError
from harnessstudio.core.method import Fidelity, MethodConfig, ModuleRef
from harnessstudio.core.pipeline import EpisodeSpec, run_episode
from harnessstudio.verify import fixtures as fx

FULL = caps("action_chunk", "seeded_sampling", "isolated_streams", "proprio", "variable_execute")


def spec(case="case-a", horizon=64):
    return EpisodeSpec(case_id=case, horizon=horizon, chunk_len=fx.CHUNK, action_dim=fx.ACTION_DIM,
                       control_hz=20.0, benchmark="synthetic", public_keys=fx.PUBLIC_KEYS,
                       noise_namespace="fixture-ns", instruction_of=fx.instruction_of,
                       with_instruction=fx.with_instruction, input_hashes=fx.input_hashes,
                       action_hashes=fx.action_hashes, action_layout=fx.ACTION_LAYOUT,
                       proprio_eef=fx.PROPRIO_EEF)


def policy_spec():
    from harnessstudio.runtime import PolicySpec
    return PolicySpec("synthetic", "fixture-checkpoint", unified_names(FULL))


def method(slots=None):
    return MethodConfig(name="t", version="1", paper="none", fidelity=Fidelity.FAITHFUL,
                        substitutions=(), required=caps("action_chunk"), slots=slots or {})


def run(m, arm, env=None, baseline=None, case="case-a"):
    env = env or fx.SyntheticEnv()
    rows = []
    record = run_episode(m, arm, spec(case), env.observation(), env, fx.SyntheticClient(), policy_spec(),
                         trace=rows.append, baseline_core_traces=baseline)
    return record, [r for r in rows if r["event"] == "action_dispatch"]


FIELDS = ("input", "action", "noise_seed", "execute_steps", "selected_k")
CUT = {"executor": ModuleRef("harnessstudio.verify.fixtures:CutEverywhereExecutor", {"min_k": 1, "per_step": True})}


class PipelineTest(unittest.TestCase):
    def test_baseline_runs_to_horizon_with_native_chunks(self):
        record, rows = run(method(), Arm.BASELINE)
        self.assertEqual(record["steps"], 64)
        self.assertEqual([r["execute_steps"] for r in rows], [16, 16, 16, 16])
        self.assertEqual(record["policy_calls"], 4)

    def test_null_equals_baseline_and_pays_k(self):
        pool = {"proposer": ModuleRef("harnessstudio.methods.medoid_bon:SeededPool", {"num_samples": 4}),
                "selector": ModuleRef("harnessstudio.methods.medoid_bon:MBRSelect")}
        base, brows = run(method(pool), Arm.BASELINE)
        null, nrows = run(method(pool), Arm.NULL, baseline=base["core_traces"])
        self.assertEqual([[r[f] for f in FIELDS] for r in brows], [[r[f] for f in FIELDS] for r in nrows])
        self.assertEqual(null["policy_calls"], 16)
        self.assertIsNone(null["divergence"])

    def test_render_noise_switches_null_to_unguarded(self):
        base, _ = run(method(), Arm.BASELINE)
        null, _ = run(method(), Arm.NULL, env=fx.SyntheticEnv(render_noise_steps=frozenset({16})),
                      baseline=base["core_traces"])
        self.assertEqual(null["divergence"]["reason"], "input")
        self.assertEqual(null["steps"], 64)

    def test_candidate_cut_and_budget(self):
        record, rows = run(method(CUT), Arm.CANDIDATE)
        self.assertEqual(record["steps"], 64)
        self.assertTrue(all(r["selected_k"] == 16 and r["planned_k"] == 16 for r in rows))
        self.assertLessEqual(record["policy_calls"], 64)
        self.assertEqual(record["cuts"][0], 1)
        self.assertTrue(all(k == 1 for k in record["executed_k"]))

    def test_null_records_but_ignores_cuts(self):
        base, _ = run(method(CUT), Arm.BASELINE)
        null, _ = run(method(CUT), Arm.NULL, baseline=base["core_traces"])
        self.assertEqual(null["policy_calls"], 4)
        self.assertTrue(all(c == 1 for c in null["cuts"]))

    def test_executor_contract_violation(self):
        ex = {"executor": ModuleRef("harnessstudio.verify.fixtures:BelowMinKExecutor",
                                    {"min_k": 4, "per_step": False})}
        with self.assertRaises(MethodContractError):
            run(method(ex), Arm.CANDIDATE)

    def test_one_frame_per_decision_and_row_fields(self):
        frames, rows = [], []
        env = fx.SyntheticEnv()
        record = run_episode(method(CUT), Arm.CANDIDATE, spec(), env.observation(), env, fx.SyntheticClient(),
                             policy_spec(), trace=rows.append, before_decision=frames.append)
        dispatch = [r for r in rows if r["event"] == "action_dispatch"]
        self.assertEqual(len(frames), record["decisions"])
        self.assertEqual(len(dispatch), record["decisions"])
        self.assertEqual(dispatch[0]["noise_stream"], "primary")
        self.assertEqual(len(dispatch[0]["extras_sha256"]), 64)

    def test_episode_op_only_changes_candidate(self):
        op = {"episode_op": ModuleRef("harnessstudio.verify.fixtures:UpperCaseOp")}
        cand, _ = run(method(op), Arm.CANDIDATE)
        null, _ = run(method(op), Arm.NULL)
        self.assertEqual(cand["instruction_used"], fx.INSTRUCTION.upper())
        self.assertEqual(null["instruction_used"], fx.INSTRUCTION)
        self.assertEqual(null["instruction_rewrite"], fx.INSTRUCTION.upper())
