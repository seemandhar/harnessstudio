"""Tier-1 contract check for samplers that expose guide() (Kinetix RTC extension, EC3).

Numpy only (no JAX): an AST scan of the sampler's source for forbidden side-effect names, then
guide() on synthetic per-rollout arrays across the (d, s) grid: shape/dtype/finite, bitwise
determinism, and no mutation of its inputs. A pass is necessary, not sufficient: every traced
guide() module also needs an independent source review and the child-env jaxpr callback scan.
"""
from __future__ import annotations

import ast
import importlib
import inspect

FORBIDDEN_NAMES = frozenset({"callback", "io_callback", "pure_callback", "debug", "print", "open", "os", "sys",
                             "time", "random", "subprocess", "socket", "requests", "urllib", "input", "exec",
                             "eval", "globals", "setattr", "__import__"})

ALLOWED_IMPORTS = frozenset({"__future__", "numpy", "math", "typing"})


def _ast_scan(cls) -> dict:
    source = inspect.getsource(inspect.getmodule(cls))
    tree = ast.parse(source)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad.append(f"name {node.id} line {node.lineno}")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            bad.append(f"attribute .{node.attr} line {node.lineno}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.append(f"global/nonlocal line {node.lineno}")
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    bad.append(f"import {a.name} line {node.lineno}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            internal = node.level > 0 and mod.split(".")[0] == "core"   # config-building only (method/capabilities)
            if not internal and mod.split(".")[0] not in ALLOWED_IMPORTS:
                bad.append(f"from {'.' * node.level}{mod} import line {node.lineno}")
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name != "__init__":
            for node in ast.walk(fn):
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        for sub in ast.walk(t):
                            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self":
                                bad.append(f"assigns self.{sub.attr} in {fn.name} line {node.lineno}")
    if bad:
        raise AssertionError("guide() module is not side-effect free: " + "; ".join(sorted(set(bad))))
    return {"source_lines": len(source.splitlines())}


def check_guide_contract(ref) -> dict:
    import numpy as np
    module, _, attr = ref.factory.partition(":")
    cls = getattr(importlib.import_module(module), attr)
    scan = _ast_scan(cls)
    sampler = ref.build()
    H, A = 8, 6
    rng = np.random.default_rng(0)
    calls = 0
    for d in range(0, 5):
        for s in range(max(1, d), H - d + 1):
            for t in (0.0, 0.2, 0.4, 0.6, 0.8):
                x = rng.standard_normal((H, A)).astype(np.float32)
                v = rng.standard_normal((H, A)).astype(np.float32)
                prev = rng.standard_normal((H, A)).astype(np.float32)
                tt = np.float32(t)
                x1 = (x + (1 - tt) * v).astype(np.float32)

                def vjp(cot, _t=tt):  # synthetic linear pullback
                    return (cot * np.float32(1.0 - 0.5 * float(_t))).astype(np.float32)

                inputs = [a.copy() for a in (x, v, x1, prev)]
                with np.errstate(divide="ignore", invalid="ignore"):
                    out1 = sampler.guide(np, x, tt, v, x1, vjp, prev, d, s, H)
                    out2 = sampler.guide(np, x, tt, v, x1, vjp, prev, d, s, H)
                calls += 2
                out1, out2 = np.asarray(out1), np.asarray(out2)
                if out1.shape != (H, A) or out1.dtype != np.float32:
                    raise AssertionError(f"guide returned {out1.shape} {out1.dtype}, expected ({H}, {A}) float32")
                if not np.all(np.isfinite(out1)):
                    raise AssertionError(f"guide returned non-finite values at d={d} s={s} t={t}")
                if out1.tobytes() != out2.tobytes():
                    raise AssertionError("guide is not deterministic")
                if any(a.tobytes() != b.tobytes() for a, b in zip(inputs, (x, v, x1, prev))):
                    raise AssertionError("guide mutated its inputs")
    if sampler.extras() != {}:
        raise AssertionError("a guide() sampler must return empty extras")
    return {"calls": calls, **scan, "identity": getattr(sampler, "identity", None)}
