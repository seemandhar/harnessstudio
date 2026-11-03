"""Isolated `codex exec` planner operator (Run X, scaleagent-codex-20261103, ledger task x-codex-operator).

Method-local module (no HarnessStudio framework edit). It calls the Codex CLI (ChatGPT login) as a stateless
LLM/VLM operator for an in-harness planner, with the same contract as Run C's generic planner-CLI proposal
(`runs/scaleagent-20261103/ext/claude_cli_proposal.md`, `OperatorPin(kind="planner_cli", cli={"backend": "codex"})`):

    CodexPlanner(model, effort, timeout_s, call_budget, log_path).complete(messages, max_tokens) -> (text, usage)

`messages` is the OpenAI chat shape a HarnessStudio controller sends through `ctx.tools["llm"]` (system / user /
assistant turns; content a string or a list of `text` and `image_url` data-URL parts), so an instance can stand in
for `harnessstudio.tools.llm.Operator`. `ask(prompt, images=[...])` is a convenience wrapper.

Isolation (each point unit-tested with a fake `codex`; live evidence in REPORT.md):
- every call runs in a fresh, empty scratch directory created under a root OUTSIDE any project (default
  /tmp/codexop-scratch); the root and all its ancestors must hold no AGENTS.md / CLAUDE.md / .codex / .claude /
  .agents / .git markers; the directory is deleted after the call;
- the child gets HOME=<scratch> (so no ~/.agents skills, no ~/AGENTS.md) and CODEX_HOME=<the login's codex home>
  (auth only), `--ignore-user-config` (no ~/.codex/config.toml: no MCP servers, plugins, profiles, project trust),
  `--ignore-rules`, `--ephemeral` (no session files), `--skip-git-repo-check`, `--sandbox read-only`,
  `approval_policy="never"`, `project_doc_max_bytes=0` (no AGENTS.md), `skills.include_instructions=false`,
  `skills.bundled.enabled=false`, `web_search="disabled"`, and the agent features shell/exec, apps, plugins, hooks,
  memories, image generation, browser/computer use, goals, multi-agent, code mode, view_image and daemon auto-start
  disabled;
- the child environment is an allowlist: no API key, token or parent agent-session variable reaches it;
- the CLI's own trace-safe telemetry (RUST_LOG target `codex_otel.trace_safe`, which carries no account id/email)
  is parsed on every call: model/slug, reasoning effort, sandbox and approval policy, MCP server count, image count,
  skill count and token usage must match the pin, else the call fails. Raw stderr is never stored;
- per-call wall timeout (the whole process group is SIGKILLed), per-episode call budget (charged before the call,
  failed calls count), Astra refusal at construction and on any reported id, model-drift refusal;
- one JSONL line per call (and per refused call) with timestamp, model ids reported by codex, effort, wall s,
  ttft, tokens, exit status, request/reply sha256 - never prompt text, reply text or credentials.
"""
from __future__ import annotations

import base64
import binascii
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

__all__ = ["CodexPlanner", "BudgetExceeded", "CodexCallError", "CodexIsolationError", "flatten", "build_argv",
           "parse_trace", "parse_json_reply", "MODEL_PIN", "EFFORTS", "planner_cli_pin"]

MODEL_PIN = "gpt-6.1-sol"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
FORBIDDEN_MODEL_WORDS = ("astra",)
DEFAULT_SCRATCH_ROOT = "/tmp/codexop-scratch"
# Agent features turned off for a stateless planner call (all listed by `codex features list`, codex-cli 0.166.0).
DISABLED_FEATURES = ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "image_generation", "goals",
                     "browser_use", "computer_use", "in_app_browser", "skill_search", "tool_suggest", "multi_agent",
                     "memories", "code_mode_host", "sleep_tool", "remote_plugin", "plugin_discovery_prefetch",
                     "skill_mcp_dependency_install", "view_image", "daemon_auto_start")
CONFIG_OVERRIDES = ('approval_policy="never"', "project_doc_max_bytes=0", "skills.include_instructions=false",
                    "skills.bundled.enabled=false", 'web_search="disabled"')
# Only the CLI's privacy-safe telemetry target (no account id / email) plus the skill-catalog count line.
RUST_LOG = "error,codex_otel.trace_safe=info,codex_skills_extension::step_state=debug"
ENV_ALLOWLIST = ("PATH", "USER", "LOGNAME", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "SSL_CERT_FILE",
                 "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy")
FORBIDDEN_ENV = re.compile(r"(API_KEY|_TOKEN$|SECRET|CACHE_SALT|^CLAUDECODE$|^CLAUDE_|^ANTHROPIC_|^OPENAI_|^NRP_"
                           r"|^HS_|^CODEX_(?!HOME$))")
MARKERS = ("AGENTS.md", "AGENTS.override.md", "CLAUDE.md", "CLAUDE.local.md", ".codex", ".claude", ".agents", ".git",
           ".mcp.json")
# JSONL item types that are a planner reply or harmless notices; anything else is a tool use and fails the call.
REPLY_ITEMS = ("agent_message",)
NOTICE_ITEMS = ("reasoning", "error")
_DATA_URL = re.compile(r"^data:(image/(?:png|jpeg|gif|webp));base64,(.*)$", re.DOTALL)
_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}
_RECONNECT = re.compile(r"^Reconnecting\.\.\. \d+/\d+")
_KV = re.compile(r'([A-Za-z_][\w.]*)=("(?:[^"\\]|\\.)*"|\S+)')


class BudgetExceeded(RuntimeError):
    """The per-episode call budget is spent (message contains 'budget', which harness_vla's controller checks)."""


class CodexCallError(RuntimeError):
    pass


class CodexIsolationError(CodexCallError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


def _refuse_astra(model: str) -> None:
    if any(w in str(model).lower() for w in FORBIDDEN_MODEL_WORDS):
        raise ValueError(f"CodexPlanner refuses Astra model ids ({model!r}); Astra needs the user's explicit "
                         "confirmation for that use and is not available through this operator")


def model_ok(pin: str, answered: str) -> bool:
    return bool(re.fullmatch(re.escape(pin) + r"(-\d{8})?", str(answered)))


# ----------------------------------------------------------------------------------------------------- messages
def _image_bytes(img: Any) -> tuple[bytes, str]:
    """(bytes, media type) of a path, raw PNG/JPEG bytes, a data URL or an HxWx3 uint8 array."""
    if isinstance(img, (bytes, bytearray)):
        b = bytes(img)
        return b, ("image/png" if b[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg")
    if isinstance(img, (str, os.PathLike)) and str(img).startswith("data:"):
        m = _DATA_URL.match(str(img))
        if not m:
            raise ValueError("unsupported data URL")
        return base64.b64decode(m.group(2), validate=True), m.group(1)
    if isinstance(img, (str, os.PathLike)):
        p = Path(img)
        b = p.read_bytes()
        mt = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
              ".gif": "image/gif"}.get(p.suffix.lower())
        if not mt:
            raise ValueError(f"unsupported image file type {p.suffix!r}")
        return b, mt
    try:  # numpy array (lazy import: the module itself needs only the standard library)
        from PIL import Image  # noqa: WPS433
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"cannot encode image of type {type(img).__name__}: {exc}") from None


def _parts(content: Any) -> list:
    if content is None:
        return []
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for p in content:
        kind = p.get("type")
        if kind == "text":
            out.append(("text", str(p.get("text", ""))))
        elif kind == "image_url":
            url = (p.get("image_url") or {}).get("url", "")
            m = _DATA_URL.match(url)
            if not m:
                raise ValueError("CodexPlanner accepts images only as base64 data URLs (no remote fetch)")
            try:
                out.append(("image", base64.b64decode(m.group(2), validate=True), m.group(1)))
            except binascii.Error:
                raise ValueError("image data URL is not valid base64") from None
        else:
            raise ValueError(f"unsupported content part type {kind!r}")
    return out


def flatten(messages: Sequence[Mapping[str, Any]]) -> tuple[str, list]:
    """(system text, parts of ONE user turn). Same semantics as Run C's planner_cli.flatten: `codex exec` takes one
    user turn, so an earlier chat is replayed in order with role headers (disclosed adaptation)."""
    system, turns = [], []
    for m in messages:
        role = m.get("role")
        parts = _parts(m.get("content"))
        if role in ("system", "developer"):
            if any(p[0] != "text" for p in parts):
                raise ValueError("images in a system message are not supported")
            system.append("\n".join(p[1] for p in parts))
        elif role in ("user", "assistant"):
            turns.append((role, parts))
        else:
            raise ValueError(f"unsupported message role {role!r}")
    if not turns or turns[-1][0] != "user":
        raise ValueError("the last message must be a user message")
    if len(turns) == 1:
        return "\n\n".join(system), list(turns[0][1])
    out = [("text", "The conversation so far follows in order; each turn starts with a role marker. "
                    "Answer the last [USER] turn as the assistant, in the format the system prompt asks for.")]
    for role, parts in turns:
        out.append(("text", "[USER]" if role == "user" else "[ASSISTANT (your earlier reply)]"))
        out.extend(parts)
    return "\n\n".join(system), out


def build_prompt(system: str, parts: list, scratch: Path) -> tuple[str, list]:
    """Prompt text for stdin and the image files written into the scratch dir (attached with -i, in order)."""
    files, text = [], []
    if system:
        text.append("SYSTEM INSTRUCTIONS:\n" + system)
    for p in parts:
        if p[0] == "text":
            text.append(p[1])
        else:
            f = scratch / f"frame_{len(files):02d}{_EXT[p[2]]}"
            f.write_bytes(p[1])
            files.append(str(f))
            text.append(f"[attached image {len(files)}: {f.name}]")
    return "\n\n".join(text), files


def build_argv(binary: str, model: str, effort: Optional[str], scratch: Path, files: Sequence[str],
               schema_file: Optional[str] = None) -> list:
    argv = [binary, "exec", "--json", "--ephemeral", "--skip-git-repo-check", "--ignore-user-config",
            "--ignore-rules", "--sandbox", "read-only", "--color", "never", "-m", model, "-C", str(scratch),
            "-o", str(scratch / ".last_message.txt")]
    if schema_file:
        argv += ["--output-schema", schema_file]
    if effort:
        argv += ["-c", f'model_reasoning_effort="{effort}"']
    for ov in CONFIG_OVERRIDES:
        argv += ["-c", ov]
    for feat in DISABLED_FEATURES:
        argv += ["--disable", feat]
    # The stdin marker goes BEFORE the images: `-i/--image <FILE>...` is variadic, and a trailing `-` after the last
    # image was counted as a third local image by codex-cli 0.166.0 (live telemetry local_image_input_count=3 for 2).
    argv.append("-")
    for f in files:
        argv += ["-i", f]
    return argv


def child_env(scratch: Path, codex_home: str, source: Optional[Mapping[str, str]] = None) -> dict:
    src = os.environ if source is None else source
    env = {k: src[k] for k in ENV_ALLOWLIST if k in src}
    env.update(HOME=str(scratch), CODEX_HOME=str(codex_home), RUST_LOG=RUST_LOG, NO_COLOR="1")
    bad = sorted(k for k in env if FORBIDDEN_ENV.search(k))
    if bad:
        raise CodexIsolationError(f"child environment would carry forbidden variables {bad}")
    return env


def check_scratch_root(root: str) -> Path:
    base = Path(root).resolve()
    for d in (base, *base.parents):
        hits = [m for m in MARKERS if (d / m).exists()]
        if hits:
            raise CodexIsolationError(f"scratch root {base} is under {d}, which holds {hits}: project instructions "
                                      "or settings could load; use a scratch root outside any project")
    return base


# ------------------------------------------------------------------------------------------------------ parsing
def _events(stdout: str) -> list:
    out = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _kv(line: str) -> dict:
    return {k: (v[1:-1] if v.startswith('"') else v) for k, v in _KV.findall(line)}


def parse_trace(stderr: str) -> dict:
    """Whitelisted facts from codex's trace-safe telemetry lines (never the raw text)."""
    out: dict = {"models": set(), "efforts": set(), "completed": [], "conversation": None, "user_prompt": None,
                 "skill_counts": [], "auth_mode": None, "app_version": None}
    for line in stderr.splitlines():
        if "enabled_skill_count=" in line:
            kv = _kv(line)
            out["skill_counts"].append(int(kv.get("enabled_skill_count", "-1")))
            continue
        if "codex_otel.trace_safe" not in line or "event.name=" not in line:
            continue
        kv = _kv(line)
        name = kv.get("event.name")
        for key in ("model", "slug"):
            if kv.get(key):
                out["models"].add(kv[key])
        out["auth_mode"] = kv.get("auth_mode", out["auth_mode"])
        out["app_version"] = kv.get("app.version", out["app_version"])
        if name == "codex.conversation_starts":
            out["conversation"] = {k: kv.get(k) for k in (
                "provider_name", "reasoning_effort", "approval_policy", "sandbox_policy", "mcp_server_count",
                "auth.env_openai_api_key_present", "auth.env_codex_api_key_present")}
            if kv.get("reasoning_effort"):
                out["efforts"].add(kv["reasoning_effort"])
        elif name == "codex.user_prompt":
            out["user_prompt"] = {k: kv.get(k) for k in ("prompt_length", "text_input_count", "image_input_count",
                                                         "local_image_input_count")}
        elif name == "codex.sse_event" and kv.get("event.kind") == "response.completed":
            rec = {k: kv.get(k) for k in ("input_token_count", "output_token_count", "cached_token_count",
                                          "reasoning_token_count", "ttft_ms", "model_reasoning_effort")}
            out["completed"].append(rec)
            if kv.get("model_reasoning_effort"):
                out["efforts"].add(kv["model_reasoning_effort"])
    out["models"] = sorted(out["models"])
    out["efforts"] = sorted(out["efforts"])
    return out


def parse_stdout(stdout: str, last_message: Path) -> dict:
    events = _events(stdout)
    text = last_message.read_text() if last_message.is_file() else ""
    replies, notices, tool_items, usage, thread_id, failed, reconnects = [], [], [], {}, None, [], []
    for e in events:
        t = str(e.get("type", ""))
        if t == "thread.started":
            thread_id = e.get("thread_id")
        elif t == "turn.completed":
            usage = e.get("usage") or {}
        elif t in ("turn.failed", "error") or t.endswith(".failed"):
            msg = str(e.get("error") or e.get("message") or t)
            if t == "error" and _RECONNECT.match(msg):
                reconnects.append(msg[:200])     # transient websocket retry (e.g. 403 under load); codex recovers
            else:
                failed.append(msg[:200])
        item = e.get("item") or {}
        it = item.get("type")
        if t == "item.completed" and it in REPLY_ITEMS and isinstance(item.get("text"), str):
            replies.append(item["text"])
        elif it in NOTICE_ITEMS:
            if t == "item.completed" and it == "error":
                notices.append(str(item.get("message", ""))[:160])
        elif it:
            tool_items.append(str(it))
    if not text and replies:
        text = replies[-1]
    return {"text": text, "thread_id": thread_id, "usage": usage, "notices": notices,
            "tool_items": sorted(set(tool_items)), "failed": failed, "reconnects": reconnects,
            "turn_completed": any(e.get("type") == "turn.completed" for e in events), "n_events": len(events)}


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json_reply(text: str) -> tuple[Optional[dict], Optional[str]]:
    """(object, None) or (None, error). Accepts a bare JSON object, a ```json fenced object, or one JSON object
    embedded in prose (the first balanced {...} that parses). Arrays/scalars are rejected."""
    t = (text or "").strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1).strip()
    try:
        obj = json.loads(t)
        return (obj, None) if isinstance(obj, dict) else (None, f"top-level JSON is {type(obj).__name__}, not object")
    except ValueError:
        pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(t[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj, None
    return None, "no JSON object in reply"


# ------------------------------------------------------------------------------------------------------ operator
class CodexPlanner:
    """Blocking completion through `codex exec`. Build one instance per episode (the call budget is per instance;
    `reset_episode()` starts a new episode explicitly). Thread-safe: concurrent calls run in separate scratch dirs."""

    backend = "codex"

    def __init__(self, model: str = MODEL_PIN, effort: Optional[str] = "medium", timeout_s: float = 180.0,
                 call_budget: int = 0, log_path: Optional[str] = None, binary: str = "codex",
                 scratch_root: str = DEFAULT_SCRATCH_ROOT, codex_home: Optional[str] = None,
                 require_model_id: bool = True, episode_id: Optional[str] = None) -> None:
        _refuse_astra(model)
        if effort is not None and effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS} or None")
        if isinstance(timeout_s, bool) or not 1 <= float(timeout_s) <= 3600:
            raise ValueError("timeout_s must be in [1, 3600]")
        if type(call_budget) is not int or call_budget < 0:
            raise ValueError("call_budget must be a non-negative int (0 = unlimited)")
        self.model, self.effort, self.timeout_s = model, effort, float(timeout_s)
        self.call_budget = call_budget
        self.log_path = Path(log_path) if log_path else None
        self.binary = binary
        self.scratch_root = scratch_root
        self.codex_home = str(Path(codex_home or os.environ.get("CODEX_HOME") or Path.home() / ".codex").resolve())
        self.require_model_id = require_model_id
        self.episode_id = episode_id
        self.calls = 0
        self.records: list = []
        self._lock = threading.Lock()
        self._procs: set = set()

    # -- configuration export -------------------------------------------------------------------------------
    def pin(self) -> dict:
        return planner_cli_pin(self.model, self.effort, self.timeout_s, self.call_budget, self.binary)

    # -- budget / lifecycle ---------------------------------------------------------------------------------
    def reset_episode(self, episode_id: Optional[str] = None) -> None:
        with self._lock:
            self.calls = 0
            self.episode_id = episode_id

    def _reserve(self) -> int:
        with self._lock:
            if self.call_budget and self.calls >= self.call_budget:
                raise BudgetExceeded(f"codex planner budget of {self.call_budget} calls per episode is spent")
            self.calls += 1
            return self.calls

    def close(self) -> None:
        """Kill every running codex process group (episode cancellation)."""
        with self._lock:
            procs = list(self._procs)
        for proc in procs:
            _killpg(proc)

    def _log(self, record: dict) -> None:
        with self._lock:
            self.records.append(record)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as fh:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")

    # -- calls ----------------------------------------------------------------------------------------------
    def ask(self, prompt: str, images: Iterable[Any] = (), system: Optional[str] = None,
            max_tokens: Optional[int] = None) -> str:
        return self._ask(prompt, images, system, None)

    def ask_json(self, prompt: str, images: Iterable[Any] = (), system: Optional[str] = None,
                 schema: Optional[Mapping[str, Any]] = None, validate=None, retries: int = 1) -> dict:
        """JSON-reply contract: the reply is constrained by `schema` (codex --output-schema, strict JSON Schema),
        parsed tolerantly (`parse_json_reply`) and checked by `validate(obj) -> None | error string`. A malformed
        reply is retried up to `retries` times with the error appended; every attempt is a charged call."""
        note = ""
        last_err = None
        for attempt in range(retries + 1):
            text = self._ask(prompt + note, images, system, schema)
            obj, err = parse_json_reply(text)
            if obj is not None and validate is not None:
                err = validate(obj)
                obj = obj if not err else None
            if obj is not None:
                return obj
            last_err = err
            note = (f"\n\nYOUR PREVIOUS REPLY WAS REJECTED ({err}). Reply again with ONE valid JSON object only.")
        raise CodexCallError(f"no valid JSON reply after {retries + 1} attempts: {last_err}")

    def _ask(self, prompt, images, system, schema):
        content: list = [{"type": "text", "text": prompt}]
        for img in images:
            data, mt = _image_bytes(img)
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mt};base64," + base64.b64encode(data).decode()}})
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        return self.complete(msgs, None, output_schema=schema)[0]

    def complete(self, messages: list, max_tokens: Optional[int] = None,
                 output_schema: Optional[Mapping[str, Any]] = None) -> tuple[str, dict]:
        """HarnessStudio Operator contract: (reply text, usage). Raises on budget, timeout, drift or isolation
        failure; the call is charged and logged in every case. `output_schema` (optional, strict JSON Schema)
        constrains the final reply through codex's --output-schema."""
        schema_text = json.dumps(output_schema, sort_keys=True) if output_schema is not None else None
        base = {"ts": _now(), "backend": self.backend, "pin": self.model, "effort_requested": self.effort,
                "timeout_s": self.timeout_s, "episode": self.episode_id,
                "requested_max_tokens": None if max_tokens is None else int(max_tokens),
                "output_schema_sha256": _sha(schema_text.encode()) if schema_text else None}
        try:
            index = self._reserve()
        except BudgetExceeded as exc:
            self._log({**base, "call": None, "status": "budget_refused", "error": str(exc),
                       "call_budget": self.call_budget})
            raise
        record = {**base, "call": index, "call_budget": self.call_budget}
        system, parts = flatten(messages)
        images = [p for p in parts if p[0] == "image"]
        record.update(n_images=len(images), image_sha256=[_sha(p[1]) for p in images],
                      request_sha256=_sha(json.dumps([system, [p[1] if p[0] == "text" else _sha(p[1])
                                                               for p in parts]]).encode()))
        root = check_scratch_root(self.scratch_root)
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        scratch = Path(tempfile.mkdtemp(prefix="call-", dir=str(root)))
        started = time.monotonic()
        try:
            if any(scratch.iterdir()):
                raise CodexIsolationError("scratch directory is not empty")
            prompt, files = build_prompt(system, parts, scratch)
            exe = self.binary if os.path.isabs(self.binary) else shutil.which(self.binary)
            if not exe:
                raise CodexCallError(f"codex binary {self.binary!r} not found on PATH")
            schema_file = None
            if schema_text:
                schema_file = str(scratch / ".output_schema.json")
                Path(schema_file).write_text(schema_text)
            argv = build_argv(exe, self.model, self.effort, scratch, files, schema_file)
            env = child_env(scratch, self.codex_home)
            proc = subprocess.Popen(argv, cwd=str(scratch), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_die_with_parent)
            with self._lock:
                self._procs.add(proc)
            try:
                out, err = proc.communicate(prompt.encode(), timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                _killpg(proc)
                proc.communicate()
                record.update(status="timeout", wall_s=round(time.monotonic() - started, 3),
                              exit_code=proc.returncode, error=f"exceeded {self.timeout_s} s; process group killed")
                self._log(record)
                raise TimeoutError(f"codex planner call exceeded {self.timeout_s} s; process group killed")
            finally:
                with self._lock:
                    self._procs.discard(proc)
            wall = time.monotonic() - started
            so = parse_stdout(out.decode(errors="replace"), scratch / ".last_message.txt")
            tr = parse_trace(err.decode(errors="replace"))
            problems = self._isolation_problems(so, tr, len(files))
            usage = so["usage"]
            reported = list(tr["models"])
            completed = tr["completed"]
            record.update(
                wall_s=round(wall, 3), exit_code=proc.returncode, thread_id=so["thread_id"],
                model_ids_reported=reported, effort_reported=tr["efforts"], cli_version=tr["app_version"],
                auth_mode=tr["auth_mode"], conversation=tr["conversation"], user_prompt=tr["user_prompt"],
                skill_counts=tr["skill_counts"],
                ttft_ms=[c.get("ttft_ms") for c in completed if c.get("ttft_ms")],
                input_tokens=int(usage.get("input_tokens") or 0),
                cached_input_tokens=int(usage.get("cached_input_tokens") or 0),
                output_tokens=int(usage.get("output_tokens") or 0),
                reasoning_output_tokens=int(usage.get("reasoning_output_tokens") or 0),
                notices=so["notices"], tool_items=so["tool_items"], cli_failures=so["failed"],
                reconnects=len(so["reconnects"]), reconnect_messages=so["reconnects"][:3],
                isolation_ok=not problems, isolation_problems=problems,
                reply_sha256=_sha(so["text"].encode()), reply_chars=len(so["text"]))
            error = None
            if any(w in str(i).lower() for i in reported for w in FORBIDDEN_MODEL_WORDS):
                error = CodexCallError("codex reported an Astra model id")
            elif any(not model_ok(self.model, i) for i in reported):
                error = CodexCallError(f"model drift: pin {self.model!r}, codex reported {reported}")
            elif self.require_model_id and not reported:
                error = CodexCallError("codex reported no model id; refusing an unverifiable reply")
            elif self.effort and tr["efforts"] and tr["efforts"] != [self.effort]:
                error = CodexCallError(f"effort drift: requested {self.effort!r}, codex reported {tr['efforts']}")
            elif problems:
                error = CodexIsolationError("isolation check failed: " + "; ".join(problems)[:300])
            elif proc.returncode != 0 or so["failed"] or not so["turn_completed"]:
                error = CodexCallError(f"codex exec failed (exit {proc.returncode}): {so['failed'][:2]}")
            elif not so["text"].strip():
                error = CodexCallError("codex returned an empty reply")
            record.update(status="ok" if error is None else "error",
                          error=None if error is None else f"{type(error).__name__}: {error}")
            self._log(record)
            if error is not None:
                raise error
            return so["text"], {"prompt_tokens": record["input_tokens"], "completion_tokens": record["output_tokens"],
                                "planner_cli": dict(record)}
        except (BudgetExceeded, TimeoutError, CodexCallError):
            raise
        except Exception as exc:
            if "status" not in record:
                record.update(status="error", wall_s=round(time.monotonic() - started, 3),
                              error=f"{type(exc).__name__}: {str(exc)[:200]}")
                self._log(record)
            raise
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _isolation_problems(self, so: dict, tr: dict, n_images: int) -> list:
        problems = []
        if so["tool_items"]:
            problems.append(f"tool use items {so['tool_items']}")
        conv = tr["conversation"]
        if conv is None:
            problems.append("no codex.conversation_starts telemetry")
        else:
            if conv.get("sandbox_policy") != "read-only":
                problems.append(f"sandbox_policy {conv.get('sandbox_policy')!r}")
            if conv.get("approval_policy") != "never":
                problems.append(f"approval_policy {conv.get('approval_policy')!r}")
            if conv.get("mcp_server_count") != "0":
                problems.append(f"mcp_server_count {conv.get('mcp_server_count')!r}")
            for k in ("auth.env_openai_api_key_present", "auth.env_codex_api_key_present"):
                if conv.get(k) not in ("false", None):
                    problems.append(f"{k}={conv.get(k)}")
        if any(c != 0 for c in tr["skill_counts"]):
            problems.append(f"enabled skills {tr['skill_counts']}")
        up = tr["user_prompt"]
        if up is not None:
            seen = int(up.get("image_input_count") or 0) + int(up.get("local_image_input_count") or 0)
            if seen != n_images:
                problems.append(f"codex saw {seen} images, {n_images} attached")
        return problems


try:  # loaded once in the parent: no import (and no import lock) happens between fork and exec
    import ctypes as _ctypes
    _LIBC = _ctypes.CDLL("libc.so.6", use_errno=True)
except Exception:  # noqa: BLE001 - non-Linux
    _LIBC = None


def _die_with_parent() -> None:
    """Linux: SIGKILL the codex child when the calling process dies (a killed episode worker must not leave an
    unbounded planner call running in its own session). Runs in the child between fork and exec."""
    if _LIBC is not None:
        _LIBC.prctl(1, 9, 0, 0, 0)                     # PR_SET_PDEATHSIG = 1, SIGKILL = 9


def _killpg(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def planner_cli_pin(model: str = MODEL_PIN, effort: Optional[str] = "medium", timeout_s: float = 180.0,
                    call_budget: int = 0, binary: str = "codex", name: str = "planner") -> dict:
    """The operator entry this backend would have in Run C's generic extension (OperatorPin kind planner_cli)."""
    _refuse_astra(model)
    return {"name": name, "model": model, "kind": "planner_cli", "base_url": "planner-cli", "api_key_env": "",
            "cli": {"backend": "codex", "effort": effort, "timeout_s": timeout_s, "episode_calls": call_budget,
                    "image_mode": "inline", "require_model_id": True, "binary": binary}}


def cli_version(binary: str = "codex") -> str:
    exe = shutil.which(binary) or binary
    r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30,
                       env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ})
    return r.stdout.strip()
