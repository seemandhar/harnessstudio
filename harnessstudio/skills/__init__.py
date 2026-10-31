"""Skill libraries (spec 5.4): bounded routines a controller runs through ctx.

eef_delta: analytic primitives for end-effector-delta control (LIBERO, LIBERO-Pro), plus vla_act, the
frozen policy as a skill. Skills read only public observations and act only through ctx.execute /
ctx.decide, so budgets, traces and the shadow null apply to them unchanged.
"""
