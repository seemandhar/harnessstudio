"""subtask-sequencer: deterministic instruction splitting, stall-based advancement, arm behaviour."""
import unittest

from harnessstudio.core.arms import Arm
from harnessstudio.core.capabilities import caps
from harnessstudio.core.pipeline import run_episode
from harnessstudio.core.registry import get
from harnessstudio.methods.subtask_sequencer import SubtaskSequencer, split_steps
from harnessstudio.verify import fixtures as fx
from harnessstudio.verify.tier1 import _policy_spec, _spec


class SplitTest(unittest.TestCase):
    def test_sequenced_sentences_and_clauses(self):
        self.assertEqual(split_steps("Pick the kettle from the counter and place it on the tray. Then pick the mug "
                                     "from the cabinet and place it on the tray. Then close the cabinet doors."),
                         ["Pick the kettle from the counter and place it on the tray.",
                          "Pick the mug from the cabinet and place it on the tray.", "Close the cabinet doors."])
        self.assertEqual(split_steps("Open the microwave, place the bowl with waffle inside the microwave, then close "
                                     "the microwave door and turn it on."),
                         ["Open the microwave.", "Place the bowl with waffle inside the microwave.",
                          "Close the microwave door and turn it on."])

    def test_keeps_pick_and_place_and_back_references_together(self):
        for text in ("Grab a lemon wedge from the fridge and one ice cube from the ice bowl, and put them in the glass.",
                     "Place the appropriate cutting tool for cutting the avocado skin on the cutting board.",
                     "Move the plastic bottles to the plastics group, and the glass bottles to the glass group."):
            self.assertEqual(split_steps(text), [text])

    def test_non_imperative_lead_is_not_split(self):
        steps = split_steps("From the pastries on the counter, select a croissant and place it on the board. "
                            "Then retrieve a jar of jam from the cabinet.")
        self.assertEqual(steps[0], "From the pastries on the counter, select a croissant and place it on the board.")
        self.assertEqual(len(steps), 2)


class StallTest(unittest.TestCase):
    def test_needs_dwell_and_a_full_still_window(self):
        s = SubtaskSequencer(16, min_dwell_steps=96, stall_window_steps=48, stall_tolerance_m=0.01)
        moving = [(i * 16, (0.01 * i, 0, 0)) for i in range(8)]
        still = [(i * 16, (0.5, 0, 0)) for i in range(8)]
        self.assertFalse(s.stalled(moving))
        self.assertTrue(s.stalled(still))
        self.assertFalse(s.stalled(still[:5]))  # 64 steps < min dwell


class ArmTest(unittest.TestCase):
    def test_null_is_baseline_and_candidate_prompts_steps(self):
        method, _ = get("subtask-sequencer")
        spec = _spec("c1").__class__(**{**_spec("c1").__dict__, "instruction_of": lambda o: "Open the drawer. Then close the drawer."})
        out = {}
        for arm in (Arm.BASELINE, Arm.NULL, Arm.CANDIDATE):
            env, rows = fx.SyntheticEnv(), []
            record = run_episode(method, arm, spec, env.observation(), env, fx.SyntheticClient(),
                                 _policy_spec(caps("action_chunk", "seeded_sampling", "proprio")), trace=rows.append)
            out[arm] = (record, [r for r in rows if r["event"] == "action_dispatch"])
        key = lambda rows: [(r["action"], r["noise_seed"], r["execute_steps"]) for r in rows]
        self.assertEqual(key(out[Arm.BASELINE][1]), key(out[Arm.NULL][1]))
        self.assertTrue(all(r["controller_prompt"] is None for r in out[Arm.NULL][1]))
        self.assertTrue(all(r["controller_prompt"] is not None for r in out[Arm.CANDIDATE][1]))
        self.assertEqual(out[Arm.CANDIDATE][0]["policy_calls"], out[Arm.BASELINE][0]["policy_calls"])


if __name__ == "__main__":
    unittest.main()
