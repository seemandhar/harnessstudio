"""Q1 component qualification of the vlm-rehome VLM judge (amendment A4.2 'V: Q1'; critic C-V1..C-V4). EVALUATOR-OWNED.

Sends the frozen judge prompt (hs_vlm_rehome.judge) with S2 (init 10-29, development memory scenes) agent-view frames to
the pinned NRP operator, one shot per frame, no retries, no fallback, and computes the predeclared pass criteria.

Frames: ideation/scripts-r4-perception/q1_frames.json restricted (r4-perception A4.1) to every hard (t2,t3,t5,t6,t8,t9)
success/failure episode plus the first 30 easy-success episodes in S2 index order; offsets +10/30/50/80 env steps after
the first public boundary. Each frame is the stored RTCF S2 png_b64 of the decision that starts at that step (the
observation the policy received), decoded and re-encoded by judge.image_png -- the same function the controller applies
to the live obs['images']['agentview'] (C-V2); stored vs delivered PNG sha256 are both recorded.

Phases (resumable, append-only calls.jsonl; a frame that has a record is never called again in that phase):
  first   one call per selected frame (457 frames)
  reask   40 stratified re-asks (C-V3): up to 20 first-pass 'abnormal' + the rest first-pass 'normal', each in sha256(frame
          id) order; requires a complete first pass
  summary criteria (a)-(d), latency, model ids -> summary.json (no calls)
Hard cap: 500 NRP calls over both phases (A4.5), concurrency <= 12 streams (C-V4; default 8).

Credentials: NRP_API_KEY is loaded in this process by robometaharness.credentials.load_nrp_environment() (or taken from
the environment, e.g. scripts/with_keys.py --provider nrp); it is only placed in the Authorization header, never logged.

Usage (from the repository root):
  harnessstudio/.venv/bin/python harnessstudio/methods/vlm_rehome/scripts/q1_qualify.py --out OUT --phase first
  ... --phase reask ; ... --phase summary
  ... --phase smoke --smoke-calls 1      # coder pre-freeze transport smoke on a NON-Q1 frame (step-0 frame of t0_i10)
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[1]
REPO = HERE.parents[4]
sys.path.insert(0, str(PKG_ROOT))
from hs_vlm_rehome import judge  # noqa: E402

RUN = REPO / "runs/rtcfpp-20261101"
FRAMES = RUN / "ideation/scripts-r4-perception/q1_frames.json"
S2_INDEX = RUN / "analysis/s2_index.json"
HARD = {2, 3, 5, 6, 8, 9}
N_EASY_S = 30
MAX_CALLS = 500
MAX_WORKERS = 12
N_REASK, N_REASK_ABN = 40, 20
THRESH = {"a_valid": 0.95, "b_hardS_max": 2, "b_t9t3_max": 1, "b_easyS_max": 1, "c_hardF_min": 0.40,
          "d_agree": 0.90, "d_agree_abn": 0.80, "d_min_stratum": 10}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def frame_id(f: dict) -> str:
    return f"{f['key']}@{f['offset']}"


def select_frames(frames_path: Path, index_path: Path) -> list:
    q = json.load(open(frames_path))
    order = [e["key"] for e in json.load(open(index_path))]
    by_ep: dict = {}
    for f in q["frames"]:
        by_ep.setdefault(f["key"], []).append(f)
    easy_s = [k for k in order if k in by_ep and by_ep[k][0]["task"] not in HARD and by_ep[k][0]["success"]][:N_EASY_S]
    keep = [k for k in order if k in by_ep and (by_ep[k][0]["task"] in HARD or k in easy_s)]
    out = []
    for k in keep:
        for f in sorted(by_ep[k], key=lambda x: x["offset"]):
            g = ("hard" if f["task"] in HARD else "easy") + ("S" if f["success"] else "F")
            out.append({**f, "group": g, "id": frame_id(f)})
    return out


class FrameUnavailable(Exception):
    """The decision's event (and frame) is absent: the last decision of an episode that ended mid-chunk is never logged
    (PITFALLS P-20261101-09). Such frames are excluded structurally, recorded in unavailable.json, never called."""


class TraceCache:
    def __init__(self):
        self.path, self.trace = None, None

    def get(self, path: str, sha: str):
        if path != self.path:
            p = Path(path)
            got = sha_file(p)
            if got != sha:
                raise RuntimeError(f"trace {path} sha256 {got} != recorded {sha}")
            self.trace, self.path = json.load(open(p)), path
        return self.trace


def instruction_of(trace: dict) -> str:
    task = str(trace["task"])            # "<suite>/<name>#<task_id>"; LIBERO task_description == name
    return task.split("/", 1)[1].rsplit("#", 1)[0]


def load_frame(cache: TraceCache, f: dict, decision: int = None) -> dict:
    t = cache.get(f["trace"], f["trace_sha256"])
    ev = [e for e in t["events"] if e.get("t") == "decision"]
    d = f["decision"] if decision is None else decision
    if d >= len(ev):
        raise FrameUnavailable(f"{f.get('id', f['key'])}: decision {d} not logged ({len(ev)} decision events)")
    e = ev[d]
    start = int(e["step"]) - int(e["executed"])
    if start != 5 * d:
        raise RuntimeError(f"{f['id'] if 'id' in f else f['key']}: decision {d} starts at step {start}, expected {5 * d}")
    stored = base64.b64decode(e["diagnostics"]["action_op"]["png_b64"])
    arr = judge.decode_png(stored)
    if arr.shape != judge.IMAGE_SHAPE:
        raise RuntimeError(f"frame shape {arr.shape} != {judge.IMAGE_SHAPE}")
    png = judge.image_png(arr)
    instr = instruction_of(t)
    msgs = judge.build_messages(instr, png)
    payload = judge.build_payload(msgs)
    body = judge.body_bytes(payload)
    return {"decision_start_step": start, "instruction": instr, "stored_png_sha256": judge.sha256(stored),
            "delivered_png_sha256": judge.sha256(png), "stored_equals_delivered": stored == png,
            "prompt_sha256": judge.prompt_sha256(msgs), "body_sha256": judge.sha256(body), "_body": body}


class Client:
    def __init__(self):
        if not os.environ.get(judge.PIN["api_key_env"]):
            sys.path.insert(0, str(REPO))
            from robometaharness.credentials import load_nrp_environment
            load_nrp_environment()
        if not os.environ.get(judge.PIN["api_key_env"]):
            raise SystemExit(f"{judge.PIN['api_key_env']} is not configured")

    def _headers(self):
        return {"Authorization": "Bearer " + os.environ[judge.PIN["api_key_env"]], "Content-Type": "application/json"}

    def models(self) -> list:
        req = urllib.request.Request(judge.BASE_URL + "/models", headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=judge.TIMEOUT_S) as r:
            return sorted(str(m.get("id")) for m in json.loads(r.read().decode()).get("data", []))

    def call(self, body: bytes) -> dict:
        req = urllib.request.Request(judge.BASE_URL + "/chat/completions", data=body, headers=self._headers(),
                                     method="POST")
        t0 = time.monotonic()
        rec = {"text": None, "model": None, "usage": None, "http_status": None, "error": None, "error_detail": None}
        try:
            with urllib.request.urlopen(req, timeout=judge.TIMEOUT_S) as r:
                rec["http_status"] = r.status
                raw = r.read().decode()
            try:
                resp = json.loads(raw)
                rec["text"] = resp["choices"][0]["message"].get("content") or ""
                rec["model"] = resp.get("model")
                rec["usage"] = resp.get("usage")
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                rec["error"], rec["error_detail"] = "error", f"bad response envelope: {type(exc).__name__}"
        except urllib.error.HTTPError as exc:
            rec["error"], rec["http_status"], rec["error_detail"] = "http_error", exc.code, f"HTTP {exc.code}"
        except (socket.timeout, TimeoutError):
            rec["error"], rec["error_detail"] = "timeout", "client timeout"
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", None)
            is_to = isinstance(reason, (socket.timeout, TimeoutError)) or "timed out" in str(reason)
            rec["error"], rec["error_detail"] = ("timeout" if is_to else "error"), type(reason).__name__
        except Exception as exc:  # noqa: BLE001
            rec["error"], rec["error_detail"] = "error", type(exc).__name__
        rec["seconds"] = round(time.monotonic() - t0, 3)
        rec["verdict"] = judge.classify(rec["text"], rec["seconds"], rec["error"])
        return rec


def read_calls(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def run_calls(out: Path, todo: list, phase: str, workers: int, allowed_models: set) -> None:
    calls_path = out / "calls.jsonl"
    lock = threading.Lock()
    used = len(read_calls(calls_path))
    if used + len(todo) > MAX_CALLS:
        raise SystemExit(f"cap: {used} calls made + {len(todo)} planned > {MAX_CALLS}")
    client = Client()
    snap = out / "models_snapshot.json"
    ids = client.models()
    with lock:
        hist = json.loads(snap.read_text()) if snap.exists() else []
        hist.append({"at": now(), "phase": phase, "ids_sha256": judge.sha256("\n".join(ids).encode()), "ids": ids,
                     "pinned_listed": judge.PIN["model"] in ids})
        snap.write_text(json.dumps(hist, indent=1))
    if judge.PIN["model"] not in ids:
        raise SystemExit(f"pinned model {judge.PIN['model']!r} not listed by /models; no fallback")
    stop = threading.Event()
    cache = TraceCache()
    prepared = []
    for f in todo:                      # decode/encode every frame before the first call (no I/O races)
        try:
            fr = load_frame(cache, f)
        except FrameUnavailable:
            continue
        prepared.append((f, fr, fr.pop("_body")))

    def one(item):
        f, fr, body = item
        if stop.is_set():
            return
        started = now()
        rec = client.call(body)
        row = {"phase": phase, "id": f["id"], "key": f["key"], "task": f["task"], "success": f["success"],
               "group": f["group"], "offset": f["offset"], "decision": f["decision"], "boundary_step": f["boundary_step"],
               "trace": f["trace"], "started_at": started, **fr, **rec,
               "model_ok": rec["model"] in allowed_models if rec["model"] is not None else None}
        with lock:
            with open(calls_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        if rec["model"] is not None and rec["model"] not in allowed_models:
            stop.set()   # C-V1: served-model drift stops the run (infrastructure)

    with ThreadPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as ex:
        list(ex.map(one, prepared))
    if stop.is_set():
        raise SystemExit("served model id drifted from the pin: stopped (C-V1)")


def available_frames(out: Path, frames: list) -> list:
    """Frames whose stored image exists; writes unavailable.json once (structural, outcome-free)."""
    cache, ok, missing = TraceCache(), [], []
    for f in frames:
        try:
            load_frame(cache, f)
            ok.append(f)
        except FrameUnavailable as exc:
            missing.append({"id": f["id"], "group": f["group"], "reason": str(exc)})
    (out / "unavailable.json").write_text(json.dumps(missing, indent=1))
    return ok


def reask_selection(first: dict) -> list:
    order = lambda ids: sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())  # noqa: E731
    abn = order([i for i, r in first.items() if r["verdict"] == "abnormal"])
    nor = order([i for i, r in first.items() if r["verdict"] == "normal"])
    pick = abn[:N_REASK_ABN]
    pick += nor[: N_REASK - len(pick)]
    if len(pick) < N_REASK:
        pick += [i for i in abn[N_REASK_ABN:]][: N_REASK - len(pick)]
    return pick


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(out: Path, frames: list) -> dict:
    calls = read_calls(out / "calls.jsonl")
    first = {r["id"]: r for r in calls if r["phase"] == "first"}
    re = {r["id"]: r for r in calls if r["phase"] == "reask"}
    sel_ids = [f["id"] for f in frames]
    missing = [i for i in sel_ids if i not in first]
    valid = [r for r in calls if r["verdict"] in judge.SEMANTIC]
    trans = [r for r in calls if r["verdict"] in judge.TRANSPORT]
    eps: dict = {}
    for f in frames:
        eps.setdefault(f["key"], {"group": f["group"], "task": f["task"], "verdicts": []})
        if f["id"] in first:
            eps[f["key"]]["verdicts"].append(first[f["id"]]["verdict"])
    flagged = lambda e: any(v == "abnormal" for v in e["verdicts"])  # noqa: E731
    G = lambda g: [k for k, e in eps.items() if e["group"] == g]  # noqa: E731
    hs, hf, es = G("hardS"), G("hardF"), G("easyS")
    hs_flag = [k for k in hs if flagged(eps[k])]
    t9t3 = [k for k in hs_flag if eps[k]["task"] in (3, 9)]
    es_flag = [k for k in G("easyS") if flagged(eps[k])]
    hf_flag = [k for k in hf if flagged(eps[k])]
    pairs = [(first[i]["verdict"], re[i]["verdict"]) for i in re if i in first]
    agree = [a == b and a in judge.SEMANTIC for a, b in pairs]
    abn_pairs = [(a, b) for a, b in pairs if a == "abnormal"]
    nor_pairs = [(a, b) for a, b in pairs if a == "normal"]
    lat = [r["seconds"] for r in calls if r["verdict"] not in ("timeout", "http_error", "error")]
    models = sorted({str(r["model"]) for r in calls if r.get("model") is not None})
    a = len(valid) / len(calls) if calls else None
    res = {
        "generated_at": now(), "n_calls": len(calls), "cap": MAX_CALLS,
        "first_pass_complete": not missing, "first_pass_missing": len(missing), "n_selected_frames": len(sel_ids),
        "n_reask": len(re), "verdict_counts": {v: sum(1 for r in calls if r["verdict"] == v)
                                               for v in judge.SEMANTIC + judge.TRANSPORT},
        "transport_rate": len(trans) / len(calls) if calls else None,
        "stored_equals_delivered_all": all(r.get("stored_equals_delivered") for r in calls) if calls else None,
        "prompt_sha256": sorted({r["prompt_sha256"] for r in calls}),
        "model_ids_returned": models,
        "latency_s": {"p50": pct(lat, 0.5), "p95": pct(lat, 0.95), "max": max(lat) if lat else None, "n": len(lat)},
        "a": {"valid_rate": a, "pass": a is not None and a >= THRESH["a_valid"]},
        "b": {"hardS_episodes": len(hs), "hardS_flagged": hs_flag, "t9t3_flagged": t9t3,
              "easyS_episodes": len(es), "easyS_flagged": es_flag,
              "pass": len(hs_flag) <= THRESH["b_hardS_max"] and len(t9t3) <= THRESH["b_t9t3_max"]
              and len(es_flag) <= THRESH["b_easyS_max"]},
        "c": {"hardF_episodes": len(hf), "hardF_flagged": len(hf_flag),
              "rate": len(hf_flag) / len(hf) if hf else None,
              "pass": bool(hf) and len(hf_flag) / len(hf) >= THRESH["c_hardF_min"]},
        "d": {"pairs": len(pairs), "abnormal_stratum": len(abn_pairs), "normal_stratum": len(nor_pairs),
              "agreement": sum(agree) / len(agree) if agree else None,
              "agreement_abnormal": (sum(1 for a_, b_ in abn_pairs if b_ == "abnormal") / len(abn_pairs))
              if abn_pairs else None},
        "thresholds": THRESH,
        "notes": ["episode-level metrics use first-pass verdicts at offsets 10-80; transport abstentions never flag",
                  "agreement counts a transport abstention in either pass as disagreement",
                  "frame groups: hardS/hardF/easyS by S2 episode outcome (development scenes init 10-29 only)"],
    }
    d = res["d"]
    res["d"]["pass"] = (d["abnormal_stratum"] >= THRESH["d_min_stratum"] and d["normal_stratum"] >= THRESH["d_min_stratum"]
                        and d["agreement"] is not None and d["agreement"] >= THRESH["d_agree"]
                        and d["agreement_abnormal"] is not None and d["agreement_abnormal"] >= THRESH["d_agree_abn"])
    res["pass_all"] = bool(res["first_pass_complete"] and len(re) == N_REASK and all(
        res[k]["pass"] for k in ("a", "b", "c", "d")))
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("first", "reask", "summary", "smoke", "plan"), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke-calls", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames = select_frames(FRAMES, S2_INDEX)
    if args.phase not in ("smoke",):
        frames = available_frames(out, frames)
    allowed = {judge.EXPECTED_SERVED_MODEL}
    meta = {"at": now(), "phase": args.phase, "judge_sha256": sha_file(PKG_ROOT / "hs_vlm_rehome/judge.py"),
            "script_sha256": sha_file(HERE), "frames_sha256": sha_file(FRAMES), "s2_index_sha256": sha_file(S2_INDEX),
            "n_selected": len(frames), "pin": judge.PIN, "max_tokens": judge.MAX_TOKENS, "timeout_s": judge.TIMEOUT_S,
            "allowed_served_models": sorted(allowed), "pid": os.getpid()}
    with open(out / "runs.jsonl", "a") as fh:
        fh.write(json.dumps(meta) + "\n")
    if args.phase == "plan":
        groups: dict = {}
        for f in frames:
            groups[f["group"]] = groups.get(f["group"], 0) + 1
        print(json.dumps({"n_frames": len(frames), "by_group": groups, "episodes": len({f['key'] for f in frames}),
                          "planned_calls": len(frames) + N_REASK, "cap": MAX_CALLS}))
        return 0
    if args.phase == "smoke":
        src = next(f for f in frames if f["key"] == "t0_i10")
        f = {**src, "decision": 0, "offset": None, "id": "SMOKE:t0_i10@step0", "group": "smoke"}
        cache = TraceCache()
        fr = load_frame(cache, f, decision=0)
        body = fr.pop("_body")
        client = Client()
        ids = client.models()
        rows = []
        for _ in range(max(0, min(args.smoke_calls, 2))):
            rec = client.call(body)
            rows.append({"phase": "smoke", "id": f["id"], "at": now(), **fr, **rec,
                         "pinned_listed": judge.PIN["model"] in ids})
        with open(out / "smoke.jsonl", "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(json.dumps([{k: r[k] for k in ("verdict", "text", "model", "seconds", "http_status", "error",
                                             "stored_equals_delivered", "pinned_listed")} for r in rows]))
        return 0
    if args.phase == "first":
        done = {r["id"] for r in read_calls(out / "calls.jsonl") if r["phase"] == "first"}
        run_calls(out, [f for f in frames if f["id"] not in done], "first", args.workers, allowed)
    elif args.phase == "reask":
        calls = read_calls(out / "calls.jsonl")
        first = {r["id"]: r for r in calls if r["phase"] == "first"}
        if any(f["id"] not in first for f in frames):
            raise SystemExit("first pass incomplete; finish --phase first before re-asking")
        pick = reask_selection({i: first[i] for i in (f["id"] for f in frames)})
        done = {r["id"] for r in calls if r["phase"] == "reask"}
        byid = {f["id"]: f for f in frames}
        (out / "reask_selection.json").write_text(json.dumps({"ids": pick, "rule": "up to 20 first-pass abnormal + "
                                                              "normal fill, sha256(id) order"}, indent=1))
        run_calls(out, [byid[i] for i in pick if i not in done], "reask", args.workers, allowed)
    res = summarize(out, frames)
    print(json.dumps({k: res[k] for k in ("n_calls", "first_pass_complete", "n_reask", "verdict_counts", "a", "c",
                                          "latency_s", "model_ids_returned", "pass_all")}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
