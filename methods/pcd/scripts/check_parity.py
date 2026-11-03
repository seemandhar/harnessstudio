"""Combine parity: hs_pcd.combine vs the authors' OpenVLAContrastInference.predict_action (Koorye/PCD @ 31b027e,
contrast_policies/openvla_contrast.py), loaded BY SOURCE (read before running) with transport-only stubs.

    <venv with torch>/bin/python -I -B check_parity.py --out DIR [--cases N]       # authors (torch) vs ours (numpy)
    <any numpy venv>/bin/python -I -B check_parity.py --out DIR --ours-only         # digests only (cross-venv check)

Stubs (transport only; none touches the combine):
  S1 `simpler_env.policies.openvla.openvla_model.OpenVLAInference`: an empty base class (the module imports it at :6);
  S2 `transforms3d.euler.euler2axangle`: raises if called (used only by postprocess_actions, never reached);
  S3 instance attributes: `vla.get_action_dim` -> 7; `_forward_scores` returns the case's prepared stream-A then
     stream-B score tensors (it replaces the two vla.generate calls, :106-124); `_decode_actions` returns the token
     ids it receives (it replaces de-tokenization, which the method does through the server's bin table).
  The authors' code (:39-59: fallbacks, combine, argmax) runs unchanged; its prints are captured and counted.
Cases: >= 10^4 randomized at width 32064 (float32 normal / bf16-quantized / forced ties / action-vocab dominance /
counterfactual == real / stream A < 7 rows / stream B 6 rows / B < 6 rows / both short / other alphas).
Pass: 0 token differences and 0 final-score bitwise differences. Sabotage controls must FAIL (prove the checker
bites): the algebraically equal s + a*(s - c), and a combine without the stream-A zero fallback.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

UPSTREAM = Path("/home/sejain/research/cvpr27/PCD-upstream/repo")
UPSTREAM_FILE = UPSTREAM / "contrast_policies/openvla_contrast.py"
HERE = Path(__file__).resolve().parent
COMBINE_FILE = HERE.parent / "hs_pcd" / "combine.py"
V, AD, ALPHA = 32064, 7, 0.8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_authors():
    # S1 / S2: import-time stubs for the two heavy imports of openvla_contrast.py (:3, :6).
    pkgs = ["simpler_env", "simpler_env.policies", "simpler_env.policies.openvla", "transforms3d"]
    for p in pkgs:
        sys.modules.setdefault(p, types.ModuleType(p))
    om = types.ModuleType("simpler_env.policies.openvla.openvla_model")

    class OpenVLAInference:  # noqa: D401 - empty transport stub
        pass
    om.OpenVLAInference = OpenVLAInference
    sys.modules["simpler_env.policies.openvla.openvla_model"] = om
    eul = types.ModuleType("transforms3d.euler")

    def euler2axangle(*a, **k):
        raise AssertionError("postprocess_actions must not run in the combine parity")
    eul.euler2axangle = euler2axangle
    sys.modules["transforms3d.euler"] = eul
    return load_by_path("authors_openvla_contrast", UPSTREAM_FILE)


def authors_combine(mod, torch, s: np.ndarray, c: np.ndarray, alpha: float):
    obj = mod.OpenVLAContrastInference.__new__(mod.OpenVLAContrastInference)
    obj.alpha = alpha
    obj.vla = types.SimpleNamespace(get_action_dim=lambda key=None: AD)
    streams = [torch.from_numpy(np.array(s, copy=True)), torch.from_numpy(np.array(c, copy=True))]
    obj._forward_scores = lambda inputs, unnorm_key, **kw: streams.pop(0)       # S3
    obj._decode_actions = lambda ids, unnorm_key: ids                             # S3
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ids, aux = obj.predict_action(None, None, unnorm_key=None, do_sample=False)
    return ids.numpy().astype(np.int64), aux["final_logits"].numpy(), buf.getvalue().count("*** Warning")


def bf16(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def cases(n: int, seed: int = 20261103):
    """Deterministic case stream: (category, s [Ta, V], c [Tb, V], alpha)."""
    rng = np.random.default_rng(seed)
    plan = [("normal", 0.50), ("bf16", 0.10), ("ties", 0.08), ("action_vocab", 0.06), ("cf_equals_real", 0.05),
            ("a_short", 0.05), ("b_six", 0.03), ("b_short", 0.03), ("both_short", 0.03), ("other_alpha", 0.07)]
    counts = {k: max(1, int(round(f * n))) for k, f in plan}
    counts["normal"] += n - sum(counts.values())
    for cat, k in counts.items():
        for _ in range(k):
            scale = float(rng.uniform(0.5, 20.0))
            off = float(rng.uniform(-10, 10))
            s = (rng.standard_normal((AD, V), dtype=np.float32) * np.float32(scale) + np.float32(off)).astype(np.float32)
            c = (rng.standard_normal((AD, V), dtype=np.float32) * np.float32(scale) + np.float32(off)).astype(np.float32)
            alpha = ALPHA
            if cat == "bf16":
                s, c = bf16(s), bf16(c)
            elif cat == "ties":
                for row in range(AD):
                    i, j = rng.choice(V, 2, replace=False)
                    top = np.float32(max(s[row].max(), c[row].max()) + 5)
                    s[row, i] = s[row, j] = top
                    c[row, i] = c[row, j] = np.float32(off)
            elif cat == "action_vocab":
                s[:, V - 64 - 256: V - 64] += np.float32(8.0)
                c[:, V - 64 - 256: V - 64] += np.float32(8.0)
                c[:, : V - 64 - 256] -= np.float32(4.0)              # contrast pushes mass outside the action bins
            elif cat == "cf_equals_real":
                c = s.copy()
            elif cat == "a_short":
                s = s[: int(rng.integers(1, AD))]
            elif cat == "b_six":
                c = c[: AD - 1]
            elif cat == "b_short":
                c = c[: int(rng.integers(1, AD - 1))]
            elif cat == "both_short":
                s, c = s[: int(rng.integers(1, AD))], c[: int(rng.integers(1, AD))]
            elif cat == "other_alpha":
                alpha = float(rng.choice([0.0, 0.2, 0.5, 1.0, 2.0]))
            yield cat, s, c, alpha


def sabotage_plus(s, c, alpha):
    s = np.asarray(s, np.float32)
    if s.shape[0] < AD or c.shape[0] != AD:
        return None
    f = s.copy()
    f[:-1] = s[:-1] + alpha * (s[:-1] - c[:-1])
    return f.argmax(-1), f


def sabotage_no_a_fallback(s, c, alpha):
    if s.shape[0] >= AD:
        return None
    f = np.zeros((AD, V), np.float32)
    f[: s.shape[0]] = s                                   # pads instead of the authors' zero replacement
    if c.shape[0] == AD:
        f[:-1] = (1 + alpha) * f[:-1] - alpha * c[:-1]
    return f.argmax(-1), f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", type=int, default=12000)
    ap.add_argument("--ours-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ours = load_by_path("hs_pcd_combine", COMBINE_FILE)
    t0 = time.time()
    res = {"schema": "pcd-combine-parity/1", "upstream_file": str(UPSTREAM_FILE),
           "upstream_sha256": sha256(UPSTREAM_FILE) if UPSTREAM_FILE.exists() else None,
           "combine_file": str(COMBINE_FILE), "combine_sha256": sha256(COMBINE_FILE), "numpy": np.__version__,
           "python": sys.version.split()[0], "cases": a.cases, "ours_only": a.ours_only}
    d_in, d_tok, d_fin = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    per_cat: dict = {}
    if not a.ours_only:
        import torch
        torch.set_num_threads(4)
        res["torch"] = torch.__version__
        authors = load_authors()
        sab = {"plus_form": [0, 0], "no_a_fallback": [0, 0]}         # [cases applied, cases with a difference]
    for cat, s, c, alpha in cases(a.cases):
        d_in.update(s.tobytes()); d_in.update(c.tobytes()); d_in.update(np.float64(alpha).tobytes())
        tok, info, fin = ours.pcd_code_final(s, c, alpha, AD, V)
        d_tok.update(tok.astype(np.int64).tobytes()); d_fin.update(fin.tobytes())
        st = per_cat.setdefault(cat, {"n": 0, "token_diff_cases": 0, "final_bitwise_diff_cases": 0, "warnings": 0,
                                      "contrasted": 0, "a_fallback": 0})
        st["n"] += 1
        st["contrasted"] += int(info["contrasted"])
        st["a_fallback"] += int(info["a_fallback"])
        if a.ours_only:
            continue
        a_tok, a_fin, warns = authors_combine(authors, torch, s, c, alpha)
        st["warnings"] += warns
        st["token_diff_cases"] += int(not np.array_equal(a_tok, tok))
        st["final_bitwise_diff_cases"] += int(a_fin.dtype != fin.dtype or a_fin.shape != fin.shape
                                              or a_fin.tobytes() != fin.tobytes())
        for key, fn in (("plus_form", sabotage_plus), ("no_a_fallback", sabotage_no_a_fallback)):
            out = fn(s, c, alpha)
            if out is not None:
                sab[key][0] += 1
                sab[key][1] += int(out[1].tobytes() != a_fin.tobytes())
    res["per_category"] = per_cat
    res["digests"] = {"inputs": d_in.hexdigest(), "ours_tokens": d_tok.hexdigest(), "ours_final": d_fin.hexdigest()}
    if not a.ours_only:
        res["token_diff_cases"] = sum(v["token_diff_cases"] for v in per_cat.values())
        res["final_bitwise_diff_cases"] = sum(v["final_bitwise_diff_cases"] for v in per_cat.values())
        res["sabotage"] = {k: {"cases": v[0], "cases_with_final_difference": v[1]} for k, v in sab.items()}
        res["passed"] = (res["token_diff_cases"] == 0 and res["final_bitwise_diff_cases"] == 0
                         and all(v[0] > 0 and v[1] > 0 for v in sab.values()))
    res["seconds"] = round(time.time() - t0, 1)
    name = "parity_ours_only.json" if a.ours_only else "parity_result.json"
    (a.out / name).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res.get(k) for k in ("passed", "token_diff_cases", "final_bitwise_diff_cases", "seconds")}
                     | {"digests": res["digests"]}))
    return 0 if (a.ours_only or res["passed"]) else 1


if __name__ == "__main__":
    sys.exit(main())
