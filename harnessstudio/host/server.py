"""Policy-server host (spec §5.2). Runs in the GR00T env; serves get_action_seeded and propose.
Server-side modules are built fresh per request and hold no state across requests (§4.7)."""
from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from ..core.method import ModuleRef
from ..core.slots import DecisionCtx
from . import rpc


class Endpoints:
    def __init__(self, policy, *, method_slots: dict, manifest_sha256: str, log, hashes):
        self.policy, self.manifest_sha256, self.log, self.hashes = policy, manifest_sha256, log, hashes
        ref = method_slots.get("sampler")
        self.sampler_ref = ModuleRef(ref["factory"], ref.get("params", {})) if ref else None
        self.calls = 0

    def _log(self, row):
        self.log.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        self.log.flush()

    def get_action_seeded(self, payload):
        from ..policies import gr00t_n15 as g
        observation, seed = rpc.parse_seeded(payload)
        self.calls += 1
        started = time.monotonic()
        result = g.seeded_sample(self.policy, observation, seed, samplers=[])
        self._log({"call": self.calls, "endpoint": "get_action_seeded", "noise_seed": seed,
                   "input": self.hashes(observation), "action": self.hashes(result),
                   "wall_seconds": time.monotonic() - started})
        return result

    def propose(self, payload):
        from ..policies import gr00t_n15 as g
        req = rpc.parse_request(payload, manifest_sha256=self.manifest_sha256)
        self.calls += 1
        started = time.monotonic()
        natives, extras = [], {}
        # instruction_sha256 is "" in M1: no milestone-1 Sampler reads it.
        ctx = DecisionCtx(req.decision_index, req.step_index, self.hashes(req.observation)["__aggregate__"], "", 0, 0)
        for index, seed in enumerate(req.seeds):
            samplers = [self.sampler_ref.build()] if self.sampler_ref and "sampler" in req.want else []
            natives.append(g.seeded_sample(self.policy, req.observation, seed, samplers=samplers,
                                           effective=(req.arm == "candidate"), ctx=ctx))
            if index == 0 and samplers:
                extras["sampler"] = samplers[0].extras()
        rpc.check_extras_size(extras)
        self._log({"call": self.calls, "endpoint": "propose", "arm": req.arm, "seeds": req.seeds,
                   "input": self.hashes(req.observation), "actions": [self.hashes(n) for n in natives],
                   "wall_seconds": time.monotonic() - started})
        return {"natives": natives, "extras": extras,
                "cost": {"policy_calls": len(req.seeds), "gpu_seconds": time.monotonic() - started,
                         "denoise_steps": g.DENOISING_STEPS * len(req.seeds)}}


def serve(manifest: dict, port: int, receipt) -> None:
    import torch
    from gr00t.eval.robot import RobotInferenceServer
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy
    from harnessstudio.bench import groot_native as gw
    from ..policies import gr00t_n15 as g
    receipt = Path(receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data = DATA_CONFIG_MAP["panda_omron"]
    policy = Gr00tPolicy(model_path=str(manifest["checkpoint_path"]), modality_config=data.modality_config(),
                         modality_transform=data.transform(), embodiment_tag="new_embodiment",
                         denoising_steps=g.DENOISING_STEPS)
    policy.model.eval()
    policy.model.requires_grad_(False)
    if any(p.requires_grad for p in policy.model.parameters()):
        raise RuntimeError("Model is not frozen")
    method = manifest["method_manifest"]
    log = (receipt.parent / "server-calls.jsonl").open("x", buffering=1)
    ep = Endpoints(policy, method_slots=method["method"]["slots"], manifest_sha256=method["sha256"], log=log,
                   hashes=gw.hashes)
    server = RobotInferenceServer(policy, host="127.0.0.1", port=port)

    def wrap(fn):
        def endpoint(payload):
            try:
                gw.public_copy(payload["observation"])  # 9-key schema check, without mutating input
                return fn(payload)
            except BaseException:
                ep._log({"status": "error", "error": traceback.format_exc()})
                raise
        return endpoint
    server.register_endpoint("get_action_seeded", wrap(ep.get_action_seeded))
    server.register_endpoint("propose", wrap(ep.propose))
    gw._write(receipt, {"status": "ready", "pid": os.getpid(), "port": port, "host": "127.0.0.1",
                        "checkpoint_identity": manifest["checkpoint_identity"],
                        "method_manifest_sha256": method["sha256"], "denoising_steps": g.DENOISING_STEPS,
                        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "frozen_weights": True,
                        "load_seconds": time.monotonic() - started,
                        "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2 ** 20})
    try:
        server.run()
    finally:
        log.close()
