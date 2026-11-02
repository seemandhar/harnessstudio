"""U5+Z composition glue (rtcfpp-20261101, reviews/a5-decision.json tz_composite_ruling.frozen_spec; proposal review
reviews/u5z-proposal-review.json C1-C9, amendment A6.Z.2). This file is the ONLY new logic of the unit; both components
are the frozen, installed packages, imported read-only after their files are sha256-verified.

Precedence (frozen, no other variant admissible this session, C9):
  every decision:
    1. U5 (hs_tube_steer TubeSteerActionOp, frozen) is called FIRST on the NATIVE chunk and the decision-time obs.
    2. If U5 has not armed in this episode after that call (U5.first_arm is None), Z (tilt-guard v1.2 'z', or the
       control 'zsign' = tilt-guard v1.1 'zsign') is called on the SAME native chunk and obs, exactly as in its own
       method, and Z's output is executed. Z's per-episode state (prev_chunk, zero_run, latched gripper, decision
       counter, signctrl seed) is therefore fed on every pre-arming decision exactly as in v1.2 / v1.1.
    3. Otherwise (U5's first arming decision and every later decision, including after a U5 disarm) U5's output is
       executed, U5's own zero-chunk handling applies unchanged, and Z is never called or executed again.
  A zero chunk AT U5's arming decision therefore belongs to U5.
No scalar is added: every component parameter is the frozen card's params block (checked for equality at construction).
Public inputs only (obs['states'], obs['task_description'], the native chunk); no RNG beyond the control's public coins;
no simulator state, case id, seed or arm. Policy inference ratio 1.0.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from harnessstudio.core.card import load_card

# --------------------------------------------------------------------------------------------------------------------
# Frozen component identities (manifests: runs/rtcfpp-20261101/implementation/{tube-steer,tilt-guard-v12,tilt-guard-v11}/
# manifest.json). Every listed file of the installed package must match before anything of it is imported.
NORM_SHA = "4ece6c43f425c3769b85819a7a4bb025d214dc263a13a6dbeb02cd36dc2cee5e"
COMPONENTS: dict = {
    "tube-steer": {
        "package": "hs_tube_steer", "distribution": "hs-tube-steer==1.0.0",
        "factory": "hs_tube_steer.action_op:TubeSteerActionOp", "card": "paper.yaml",
        "manifest": "runs/rtcfpp-20261101/implementation/tube-steer/manifest.json",
        "manifest_sha256": "a36e974e19583d5e883dc9d8ca31b76d7cbc00f57b6e29c90ccb7d2409eb2175",
        "files": {
            "__init__.py": "2f2cb0bd98d06506a7629c656cfeaa27659150c7fa9663dd961f3cfdf976ffca",
            "action_op.py": "6d3b1e1ce17956b2934a6572ef091702dbf5327b5b3977f154a3a2542e25e7a5",
            "paper.yaml": "b02f5e59af89b510d442ec1f6a7ad4708203158ae4bfd7cc2f0edb8a3d6cb35f",
            "assets/tube_memory.npz": "fcf8338306294cace8979ede718d2ede7d8b287198144fbf07e21604a5677f0a",
            "assets/tube_memory.json": "00d2e87bc024ffed09fc04953e389f1bbded1f26c3ba3a4fe2be25d39fb2d180",
            "assets/pi0_fast_libero_norm_stats.json": NORM_SHA,
        },
    },
    "tilt-guard-v12-z": {
        "package": "hs_tilt_guard_v12", "distribution": "hs-tilt-guard-v12==1.2.0",
        "factory": "hs_tilt_guard_v12.composite:TiltGuardV12ActionOp", "card": "paper_z.yaml",
        "manifest": "runs/rtcfpp-20261101/implementation/tilt-guard-v12/manifest.json",
        "manifest_sha256": "b79c35dba36ec75d10aa612b68dc14190469303d193275b65977a3939ae978af",
        "files": {
            "__init__.py": "14f4d87de1dbf9396c9726366e44745e9dc973fc56b8abab7dc8dee3ff20f4dd",
            "action_op.py": "b4348f2ff28649ec20cde6e2350bc5f960e4b8fbef1100e1861a16aeec17471b",
            "composite.py": "cdd6cad1594080ce762ece3f7ac7953b0b64d3146c0a33b91fe62be9d0a43d8c",
            "geometry.py": "d8f7bbb5c0dad83d42843054a9ef7ab927c2335a3e1131c247b2b75b5e5624c0",
            "paper_z.yaml": "e798eca953b9a3b71f956d6570f7b0d46858ef75256b20a6d9416ea14e64e228",
            "assets/pi0_fast_libero_norm_stats.json": NORM_SHA,
        },
    },
    "tilt-guard-v11-zsign": {
        "package": "hs_tilt_guard_v11", "distribution": "hs-tilt-guard-v11==1.1.0",
        "factory": "hs_tilt_guard_v11.action_op:TiltGuardActionOp", "card": "paper_zsign.yaml",
        "manifest": "runs/rtcfpp-20261101/implementation/tilt-guard-v11/manifest.json",
        "manifest_sha256": "f917c3f0d398df847655e36752b97bf827ad6235ed9b0c8f8a573979382fa2df",
        "files": {
            "__init__.py": "a391d2251bea67c3d6a787483d48284f59812e39b5d81eb18932f73873d988b7",
            "action_op.py": "b4348f2ff28649ec20cde6e2350bc5f960e4b8fbef1100e1861a16aeec17471b",
            "geometry.py": "d8f7bbb5c0dad83d42843054a9ef7ab927c2335a3e1131c247b2b75b5e5624c0",
            "paper_zsign.yaml": "8be57c4d25f96c1b0d195c8c3f9ceee9c80f8dd41930db05324434d860418868",
            "assets/pi0_fast_libero_norm_stats.json": NORM_SHA,
        },
    },
}
# variant -> (U5 component, Z component). The -tv variants are NOT defined (U5 = TV only via A5.3(5); see README).
VARIANTS: dict = {
    "u5z-t": ("tube-steer", "tilt-guard-v12-z"),
    "u5zsign-t": ("tube-steer", "tilt-guard-v11-zsign"),
}
# C5: the control's parameters equal paper_z v1.2's except these keys (zero; the inert guard fields of v1.1 cards).
ZSIGN_DIFF_KEYS = ("zero", "guard_factory", "guard_params")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_dir(package: str) -> Path:
    """Directory of an installed top-level package, found WITHOUT executing it."""
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(f"frozen component package {package!r} is not installed")
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def verify_component(name: str, components: Optional[Mapping] = None) -> Path:
    """sha256-verify every pinned file of a frozen component before any of its code is imported; return its dir."""
    comp = (COMPONENTS if components is None else components)[name]
    root = package_dir(comp["package"])
    bad = {}
    for rel, want in comp["files"].items():
        p = root / rel
        got = _sha(p) if p.is_file() else "missing"
        if got != want:
            bad[rel] = got
    if bad:
        raise RuntimeError(f"frozen component {name} ({root}) differs from its pinned sha256: {bad} (refusing to start)")
    return root


def card_params(name: str, components: Optional[Mapping] = None) -> dict:
    comp = (COMPONENTS if components is None else components)[name]
    root = verify_component(name, {name: comp})
    return json.loads(json.dumps(dict(load_card(root / comp["card"])["params"]), sort_keys=True))


def check_zsign_params(zsign: Mapping[str, Any], z: Mapping[str, Any]) -> None:
    """C5: zsign (v1.1 card) == paper_z (v1.2 card) except 'zero' and the inert guard fields; guard false."""
    if zsign.get("guard") is not False or z.get("guard") is not False:
        raise RuntimeError("Z / Zsign must have guard false (hs_band_guard is never built)")
    if zsign.get("zero") != "signctrl" or z.get("zero") != "substitute":
        raise RuntimeError("Zsign must be zero='signctrl' and Z zero='substitute'")
    a = {k: v for k, v in zsign.items() if k not in ZSIGN_DIFF_KEYS}
    b = {k: v for k, v in z.items() if k not in ZSIGN_DIFF_KEYS}
    if a != b:
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        raise RuntimeError(f"Zsign parameters differ from paper_z beyond {ZSIGN_DIFF_KEYS}: {diff}")


def _build(name: str, factory: str, params: Mapping[str, Any], components: Mapping):
    comp = components[name]
    if factory != comp["factory"]:
        raise RuntimeError(f"{name}: factory {factory!r} != frozen {comp['factory']!r}")
    frozen = card_params(name, components)         # verifies the files first
    if json.loads(json.dumps(dict(params), sort_keys=True)) != frozen:
        raise RuntimeError(f"{name}: params differ from the frozen card {comp['card']} (no new scalars allowed)")
    root = package_dir(comp["package"])
    module, _, attr = factory.partition(":")
    cls = getattr(importlib.import_module(module), attr)
    # every loaded module of the component must come from the verified directory (no shadowing distribution)
    for mname, mod in list(sys.modules.items()):
        if mname == comp["package"] or mname.startswith(comp["package"] + "."):
            f = getattr(mod, "__file__", None)
            if f is not None and root not in Path(f).resolve().parents:
                raise RuntimeError(f"{mname} loaded from {f}, not from the verified {root}")
    return cls(**dict(params))


class U5ZActionOp:
    """One DecisionLoop action_op: U5 first on the native chunk; Z's output iff U5 has not armed yet (see module doc).

    Extension point (for a LATER, separate distribution only, e.g. the -tv entries if A5.3(5) makes U5 = TV): a subclass
    may set COMPONENTS / VARIANTS class attributes with additional frozen, sha-pinned components; the precedence logic
    (transform) is inherited unchanged. This package itself defines only the -t variants."""

    COMPONENTS: Mapping = COMPONENTS
    VARIANTS: Mapping = VARIANTS

    def __init__(self, variant: str, u5: Mapping[str, Any], z: Mapping[str, Any]) -> None:
        comps = self.COMPONENTS
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VARIANTS)}, got {variant!r}")
        u5_name, z_name = self.VARIANTS[variant]
        if u5.get("component") != u5_name or z.get("component") != z_name:
            raise RuntimeError(f"{variant}: components must be {u5_name} + {z_name}")
        if z_name == "tilt-guard-v11-zsign":
            check_zsign_params(dict(z["params"]), card_params("tilt-guard-v12-z", comps))
        self.variant = variant
        self.u5 = _build(u5_name, u5["factory"], u5["params"], comps)
        self.z = _build(z_name, z["factory"], z["params"], comps)
        if getattr(self.u5, "first_arm", "missing") is not None or getattr(self.u5, "mode", None) != "steer":
            raise RuntimeError("U5 must be a fresh tube-steer op in mode 'steer' exposing first_arm")
        if getattr(self.z, "use_guard", True) or getattr(self.z, "guard", "x") is not None:
            raise RuntimeError("Z must run without G (guard false)")
        # diagnostic flag only: the frozen detector of the module that defines Z's transform (verified above)
        self._is_zero = sys.modules[type(self.z).transform.__module__].is_openpi_zero_chunk
        self.q = 0
        self.arm_decision: Optional[int] = None
        self._diag: dict = {}
        self.stats = {"decisions": 0, "z_called": 0, "z_executed_substitutions": 0, "z_first_substitution": None,
                      "u5_executed_decisions": 0, "u5_first_arm": None, "native_zero_after_arming": 0}

    def diagnostics(self) -> dict:
        return dict(self._diag)

    def _native_zero(self, chunk, obs) -> Optional[bool]:
        """Diagnostic only (never changes outputs): is the NATIVE chunk the openpi zero chunk (frozen Z detector)?"""
        try:
            import numpy as np
            s = self.z._state(obs)
            c = np.asarray(chunk)
            if s is None or s.size < 6 or not np.all(np.isfinite(s[:6])) or c.ndim != 2 or c.shape[1] != self.z.action_dim:
                return None
            return bool(self._is_zero(c, s, self.z.action_mean, self.z.zero_tol))
        except Exception:  # noqa: BLE001  (diagnostics must never raise)
            return None

    def transform(self, chunk, obs, memory):
        t0 = time.perf_counter()
        q = self.q
        self.q += 1
        self.stats["decisions"] += 1
        u5_out = self.u5.transform(chunk, obs, memory)            # (1) U5 first, on the native chunk
        u5_diag = self.u5.diagnostics()
        z_diag = None
        if self.u5.first_arm is None:                               # (2) U5 not armed yet after this call -> Z
            z_out = self.z.transform(chunk, obs, memory)            #     same native chunk and obs
            z_diag = self.z.diagnostics()
            self.stats["z_called"] += 1
            out, executed = z_out, "z"
            substituted = z_diag.get("z_mode") in ("tail", "hold")
            if substituted:
                self.stats["z_executed_substitutions"] += 1
                if self.stats["z_first_substitution"] is None:
                    self.stats["z_first_substitution"] = q
        else:                                                       # (3) from U5's first arming on: U5 only
            if self.arm_decision is None:
                self.arm_decision = q
                self.stats["u5_first_arm"] = self.u5.first_arm
            out, executed = u5_out, "u5"
            substituted = False
            self.stats["u5_executed_decisions"] += 1
        native_zero = self._native_zero(chunk, obs)
        if executed == "u5" and native_zero:
            self.stats["native_zero_after_arming"] += 1
        glue = {"q": q, "executed": executed, "u5_first_arm": self.u5.first_arm, "arm_decision": self.arm_decision,
                "z_called": z_diag is not None, "z_substituted": bool(substituted), "native_zero": native_zero,
                "u5_steered": bool(u5_diag.get("steer", False)), "out_is_native": out is chunk,
                "wall_s": time.perf_counter() - t0, "stats": dict(self.stats)}
        self._diag = {"glue": glue, "u5": u5_diag, "z": z_diag}
        return out
