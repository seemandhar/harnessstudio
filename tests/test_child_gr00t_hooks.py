import unittest

try:
    import torch
except ImportError:  # only meaningful under the GR00T env
    torch = None


def make_policy():
    class Head(torch.nn.Module):
        action_horizon, num_inference_timesteps = 16, 4

        def __init__(self):
            super().__init__()
            self.action_encoder = torch.nn.Linear(32, 32)
            self.action_decoder = torch.nn.Linear(32, 32)

        def sample(self):
            x = torch.randn(1, 16, 32)
            for _ in range(4):
                v = self.action_decoder(self.action_encoder(x))[:, -16:]
                x = x + 0.25 * v
            return x

    class Policy:
        def __init__(self):
            self.model = type("M", (), {})()
            self.model.action_head = Head()

        def get_action(self, obs):
            return {"x": self.model.action_head.sample()}
    torch.manual_seed(0)
    return Policy()


@unittest.skipIf(torch is None, "needs torch (run under the GR00T env)")
class HookTest(unittest.TestCase):
    def test_seeded_sample_reproducible_and_noop_sampler_identical(self):
        from harnessstudio.policies import gr00t_n15 as g
        p = make_policy()
        a = g.seeded_sample(p, {}, 123, samplers=[])
        b = g.seeded_sample(p, {}, 123, samplers=[g.NoOpSampler()])
        c = g.seeded_sample(p, {}, 123, samplers=[])
        self.assertTrue(torch.equal(a["x"], b["x"]))
        self.assertTrue(torch.equal(a["x"], c["x"]))

    def test_clone_replacement_is_identical(self):
        from harnessstudio.policies import gr00t_n15 as g

        class Clone:
            identity = {}
            def on_step(self, step, x_t, v_t, ctx):
                return v_t.clone()
            def extras(self):
                return {}
        p = make_policy()
        self.assertTrue(torch.equal(g.seeded_sample(p, {}, 9, samplers=[])["x"],
                                    g.seeded_sample(p, {}, 9, samplers=[Clone()], effective=True)["x"]))

    def test_replacing_sampler_changes_output_only_when_effective(self):
        from harnessstudio.policies import gr00t_n15 as g

        class Zero:
            identity = {}
            def on_step(self, step, x_t, v_t, ctx):
                return torch.zeros_like(v_t)
            def extras(self):
                return {}
        p = make_policy()
        native = g.seeded_sample(p, {}, 7, samplers=[])
        discarded = g.seeded_sample(p, {}, 7, samplers=[Zero()], effective=False)
        replaced = g.seeded_sample(p, {}, 7, samplers=[Zero()], effective=True)
        self.assertTrue(torch.equal(native["x"], discarded["x"]))
        self.assertFalse(torch.equal(native["x"], replaced["x"]))

    def test_velocity_readout_shape(self):
        from harnessstudio.methods.chunktrust_ahs.modules import VelocityReadout
        from harnessstudio.policies import gr00t_n15 as g
        r = VelocityReadout()
        g.seeded_sample(make_policy(), {}, 1, samplers=[r])
        v = r.extras()["velocity"]
        self.assertEqual(v.shape, (4, 16, 32))
        self.assertEqual(str(v.dtype), "float32")
