"""Runner v2.2: free each robosuite render context's GL objects in ITS OWN EGL context.

Repair `repair-render-destructor-v2-2` (repair 2 of 2, supervisor decision
control/launch/supervisor_decision_repair2_v2_2.json; diagnosis reports/render_concurrency_diagnosis.md).

Defect (class G, gross/persistent stale or garbage frames).  robosuite @5ce6643f
`utils/binding_utils.py` MjRenderContext.__del__ calls ``self.con.free()`` (mjr_freeContext)
without making ``self.gl_ctx`` current. RoboCasaGymEnv.__init__ performs an unseeded reset whose render
context becomes cyclic garbage after ``reset(seed)`` (hard reset) has created and made current a NEW
context. When Python's cyclic GC finalises the old context, MuJoCo deletes framebuffers, renderbuffers,
textures and display lists *by name in the new context* (names coincide), so later readbacks return stale
or uninitialised buffers. GC timing is process-dependent, so this looks like random render corruption.

Repair.  Rebind ``MjRenderContext.__del__`` (simulator client only, fail closed unless robosuite's
binding_utils.py and egl_context.py match the audited hashes and the original __del__ source matches):
remember the current EGL context, make the dying context current, free the MuJoCo GL objects, restore the
previous current context (or none if the dying one was current), then destroy the dying EGL context.
Rendering, cameras, image sizes, MSAA, physics, observations and the policy path are untouched; only the
destruction order/context of an object that is already being destroyed changes.

Not addressed (disclosed): class S, sporadic +-1..2-level differences in a few pixels (~1% of frames),
which occurs even in a single process on the machine; cause unknown.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

PINNED = {
    'robosuite/utils/binding_utils.py': 'a3d2526fb2e60875e2c3a87b8828b8e3b512781708c80d0c96ee6024980896c8',
    'robosuite/renderers/context/egl_context.py': '72ff1e153da85b85a35ebd0350c4693d3b89ffd5e1f13ff49813b7a3ad4c51c6',
}
ROBOSUITE_COMMIT = '5ce6643f3092639d08f7b0f90ed1c6a84f50552c'
ORIGINAL_DEL_SHA256 = None  # filled at import from the audited text below and re-checked in install()
ORIGINAL_DEL_TEXT = '''    def __del__(self):
        # free mujoco rendering context and GL rendering context
        self.con.free()
        try:
            self.gl_ctx.free()
        except Exception:
            # avoid getting OpenGL.error.GLError
            pass
        del self.con
        del self.gl_ctx
        del self.scn
        del self.cam
        del self.vopt
        del self.pert
'''
ORIGINAL_DEL_SHA256 = hashlib.sha256(ORIGINAL_DEL_TEXT.encode()).hexdigest()
PATCH_NAME = 'robosuite MjRenderContext.__del__ -> free GL objects in own EGL context'

# Evaluator-only diagnostics (never returned to the policy): one entry per finalised render context.
EVENTS = []


def _same(a, b):
    return bool(a) and bool(b) and a.address == b.address


def owned_context_del(self):
    """Replacement for MjRenderContext.__del__ (same effects, correct GL context)."""
    from OpenGL import EGL
    from robosuite.renderers.context import egl_context as ec
    gl = getattr(self, 'gl_ctx', None)
    con = getattr(self, 'con', None)
    own = getattr(gl, '_context', None)
    previous = EGL.eglGetCurrentContext()
    foreign_current = bool(previous) and not _same(previous, own)
    EVENTS.append({'foreign_context_was_current': foreign_current})
    try:
        if con is not None:
            if own:
                gl.make_current()
            con.free()
    finally:
        if foreign_current:
            EGL.eglMakeCurrent(ec.EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, previous)
        try:
            if gl is not None:
                gl.free()  # releases current binding only if the dying context is current, then destroys it
        except Exception:
            pass
    for name in ('con', 'gl_ctx', 'scn', 'cam', 'vopt', 'pert'):
        try:
            delattr(self, name)
        except AttributeError:
            pass


def install():
    """Apply the repair (idempotent) in the simulator client; fail closed on unaudited sources."""
    import robosuite
    from robosuite.utils import binding_utils as bu
    from robosuite.renderers.context import egl_context  # noqa: F401  (pinned, used by the fix)
    base = Path(robosuite.__file__).resolve().parent.parent
    hashes = {}
    for rel, want in PINNED.items():
        got = hashlib.sha256((base / rel).read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f'{rel} changed ({got}); render-context fix not audited for it')
        hashes[rel] = got
    current = bu.MjRenderContext.__dict__.get('__del__')
    if current is not owned_context_del:
        try:
            text = inspect.getsource(current)
        except (OSError, TypeError):
            text = ''
        if hashlib.sha256(text.encode()).hexdigest() != ORIGINAL_DEL_SHA256:
            raise RuntimeError('MjRenderContext.__del__ is not the audited original')
        bu.MjRenderContext.__del__ = owned_context_del
    return {'patch': PATCH_NAME, 'robosuite_commit': ROBOSUITE_COMMIT, 'sha256': hashes,
            'original_del_sha256': ORIGINAL_DEL_SHA256}
