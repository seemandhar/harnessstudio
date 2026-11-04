"""Attach a StarLayerMixer to a stack of transformer decoder layers with PyTorch hooks.

Equivalent of the authors' monkeypatch (STAR@0eaa15f src/openpi/inference/star/runtime.py:33-53), expressed with
`register_forward_pre_hook` / `register_forward_hook` so the module objects are never modified and the hooks are
removed in `finally` (STAR-off requests then run the untouched graph: bitwise native).

Per forward pass of the stack: the pre-hook of layer 0 resets the mixer (runtime.py:44-45); for every layer the
pre-hook stores h^{k-1} (the layer's first positional argument or `hidden_states` kwarg, runtime.py:38) and the
forward hook replaces element 0 of the output tuple (or the tensor) with mixer.step(h^{k-1}, h^k)
(runtime.py:46-51).
"""
from __future__ import annotations

import contextlib
from typing import Any, Sequence


@contextlib.contextmanager
def star_hooks(layers: Sequence[Any], mixer):
    """Context manager: mixer applied to `layers` (e.g. OpenVLA `model.language_model.model.layers`)."""
    handles = []
    stash: dict[int, Any] = {}

    def make_pre(idx):
        def pre(module, args, kwargs):
            hidden_in = args[0] if args else kwargs.get("hidden_states")
            if hidden_in is None:          # runtime.py:39-40: no hidden state -> original forward, no step
                return None
            if idx == 0:
                mixer.reset()
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
