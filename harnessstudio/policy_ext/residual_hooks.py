"""Request-scoped decoder-layer residual hooks (EXT openvla-residual-hook, capability ``residual_hooks``).

A server-side *mixer* rewrites the output of every decoder layer of a transformer stack during one request:

    mixer.reset()                    called at layer 0 of every forward pass of the stack
    mixer.step(h_in, h_out) -> h     called once per layer, in order; h replaces the layer's hidden-state output
    mixer.correction_start_layer     optional int (1-based): the first layer the mixer may change (for accounting)
    mixer.activity() -> dict         optional, called by the server once after the decode (logging only, e.g.
                                     STAR's per-layer edit ratio; EXT openvla-residual-hook review critic-ext-2 P9)

`residual_hooks(layers, mixer)` attaches it with PyTorch forward pre-hooks / forward hooks (no module is modified:
the hooks are removed in ``finally``, also when the decode raises), so a request without a mixer runs the untouched
module graph. This is the hook-based form of the STAR authors' layer wrapper (PKU-VaLuE-Lab/STAR@0eaa15f
src/openpi/inference/star/runtime.py:33-53): the pre-hook stores h^{k-1} (first positional argument or the
``hidden_states`` keyword, runtime.py:38), layer 0 resets the mixer (runtime.py:44-45), the forward hook replaces
element 0 of the output tuple (runtime.py:46-51). A layer called without a hidden state runs unchanged, without a
step (runtime.py:39-40).

`ForwardRecord` counts what the hooks saw (forwards, per-forward sequence length, layer calls, calls at or after
the mixer's correction start, the mixer's step context at each forward start) so a server can assert the request
shape at runtime (review critic-ext-1 R4). Callers must hold the server's inference lock for the whole `with`
block: hooks are module state (R3).
"""
from __future__ import annotations

import contextlib
from typing import Any, Sequence


class ForwardRecord:
    def __init__(self) -> None:
        self.forwards: list[dict] = []

    def start(self, seq_len: int, ctx: Any) -> None:
        self.forwards.append({"seq_len": int(seq_len), "layer_calls": 0, "mixed_calls": 0, "ctx": ctx})

    def summary(self) -> dict:
        return {"forwards": len(self.forwards), "seq_lens": [f["seq_len"] for f in self.forwards],
                "layer_calls": [f["layer_calls"] for f in self.forwards],
                "mixed_calls": [f["mixed_calls"] for f in self.forwards]}


def _ctx_of(mixer) -> Any:
    get = getattr(mixer, "step_context", None)
    if callable(get):
        return get()
    ctx = getattr(mixer, "_step_ctx", None)
    return dict(ctx) if isinstance(ctx, dict) else None


@contextlib.contextmanager
def residual_hooks(layers: Sequence[Any], mixer, record: ForwardRecord | None = None):
    handles = []
    stash: dict[int, Any] = {}
    start = int(getattr(mixer, "correction_start_layer", 1) or 1)

    def make_pre(idx):
        def pre(module, args, kwargs):
            hidden_in = args[0] if args else kwargs.get("hidden_states")
            if hidden_in is None:
                return None
            if idx == 0:
                mixer.reset()
                if record is not None:
                    record.start(hidden_in.shape[1] if getattr(hidden_in, "ndim", 0) >= 2 else -1, _ctx_of(mixer))
            stash[idx] = hidden_in
            return None
        return pre

    def make_post(idx):
        def post(module, args, kwargs, out):
            hidden_in = stash.pop(idx, None)
            if hidden_in is None:
                return None
            hidden_out = out[0] if isinstance(out, tuple) else out
            hidden_out = mixer.step(hidden_in, hidden_out)
            if record is not None and record.forwards:
                record.forwards[-1]["layer_calls"] += 1
                if idx + 1 >= start:
                    record.forwards[-1]["mixed_calls"] += 1
            if isinstance(out, tuple):
                return (hidden_out, *out[1:])
            return hidden_out
        return post

    try:
        for idx, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(make_pre(idx), with_kwargs=True))
            handles.append(layer.register_forward_hook(make_post(idx), with_kwargs=True))
        yield mixer
    finally:
        for h in handles:
            h.remove()
        stash.clear()
