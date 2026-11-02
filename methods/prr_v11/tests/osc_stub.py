"""Kinematic/dynamic stub of the robosuite OSC_POSE plant used by LIBERO (test fixture, not used at runtime):
per control step the goal is reset to current + 0.05 a (m) / exp(0.5 a) R (world frame), then 25 x 2 ms substeps of
x'' = kp (goal - x) - kd x', kp 150, kd = 2 sqrt(kp) (critical damping), the same for orientation (world-frame log map)."""
from __future__ import annotations

import math

import numpy as np

from hs_prr_v11.geometry import exp, log


class OSCStub:
    def __init__(self, p0, R0, kp: float = 150.0, zeta: float = 1.0):
        self.p, self.R = np.asarray(p0, float).copy(), np.asarray(R0, float).copy()
        self.v, self.w = np.zeros(3), np.zeros(3)
        self.kp, self.kd = kp, 2 * zeta * math.sqrt(kp)

    def step(self, a):
        a = np.clip(np.asarray(a, float)[:6], -1, 1)
        gp, gR = self.p + 0.05 * a[:3], exp(0.5 * a[3:6]) @ self.R
        for _ in range(25):
            self.v += (self.kp * (gp - self.p) - self.kd * self.v) * 0.002
            self.p = self.p + self.v * 0.002
            er = log(gR @ self.R.T)
            self.w += (self.kp * er - self.kd * self.w) * 0.002
            self.R = exp(self.w * 0.002) @ self.R

    def state(self, width=0.08):
        return np.concatenate([self.p, log(self.R), [width / 2, -width / 2]])
