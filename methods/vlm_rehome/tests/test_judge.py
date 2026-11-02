"""Judge + Q1 harness tests (critic C-V1..C-V4): no network."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts"))

from hs_vlm_rehome import judge  # noqa: E402


def frame(seed=0):
    return np.random.default_rng(seed).integers(0, 256, judge.IMAGE_SHAPE, dtype=np.uint8)


def test_pin_matches_endpoint_registry_pin():
    import subprocess
    out = subprocess.run([sys.executable, "-m", "harnessstudio.endpoints", "pin", "nrp", "qwen3-vl", "--name", "judge"],
                         capture_output=True, text=True, check=True).stdout
    import yaml
    reg = yaml.safe_load(out)[0]
    for k in ("name", "model", "base_url", "api_key_env", "original_model"):
        assert judge.PIN[k] == reg[k]
    assert judge.PIN["extra_body"]["chat_template_kwargs"] == reg["extra_body"]["chat_template_kwargs"]
    assert judge.PIN["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert judge.PIN["temperature"] == 0.0 and judge.MAX_TOKENS == 64 and judge.TIMEOUT_S == 60.0
    assert "astra" not in judge.PIN["model"].lower()


def test_payload_bytes_equal_framework_operator(monkeypatch):
    """The direct Q1 body equals what harnessstudio's OpenAICompatibleOperator sends for the same messages (C-V2)."""
    from harnessstudio.core.method import OperatorPin
    from harnessstudio.tools.llm import OpenAICompatibleOperator, build_operators
    pin = OperatorPin(**judge.PIN)
    op = build_operators([pin])["judge"]
    assert isinstance(op, OpenAICompatibleOperator)
    captured = {}

    def fake_request(path, payload=None):
        captured["path"], captured["body"] = path, json.dumps(payload).encode()
        return {"model": judge.EXPECTED_SERVED_MODEL, "choices": [{"message": {"content": '{"state": "normal"}'}}]}
    monkeypatch.setattr(op, "_request", fake_request)
    monkeypatch.setattr(op, "verify", lambda: None)
    png = judge.image_png(frame(1))
    msgs = judge.build_messages("put the bowl on the plate", png)
    text, _ = op.complete(msgs, judge.MAX_TOKENS)
    assert text == '{"state": "normal"}' and captured["path"] == "/chat/completions"
    assert captured["body"] == judge.body_bytes(judge.build_payload(msgs))
    assert op.base_url == judge.BASE_URL


def test_messages_are_frozen_and_image_is_png_of_the_array():
    a = frame(2)
    png = judge.image_png(a)
    assert np.array_equal(judge.decode_png(png), a)
    msgs = judge.build_messages("open the drawer", png)
    assert msgs[0] == {"role": "system", "content": judge.SYSTEM}
    c = msgs[1]["content"]
    assert [p["type"] for p in c] == ["text", "image_url", "text"]
    assert c[0]["text"] == "Task: open the drawer" and c[2]["text"] == judge.QUESTION
    assert base64.b64decode(c[1]["image_url"]["url"].split(",", 1)[1]) == png
    # prompt hash ignores pixels, depends on text
    assert judge.prompt_sha256(msgs) == judge.prompt_sha256(judge.build_messages("open the drawer", judge.image_png(frame(3))))
    assert judge.prompt_sha256(msgs) != judge.prompt_sha256(judge.build_messages("close the drawer", png))
    with pytest.raises(ValueError):
        judge.image_png(a.astype(np.float32))
    with pytest.raises(ValueError):
        judge.image_png(a[..., 0])


@pytest.mark.parametrize("text,verdict", [
    ('{"state": "abnormal"}', "abnormal"), ('{"state":"normal"}', "normal"), (' {"state": "Unsure"} ', "unsure"),
    ('```json\n{"state": "abnormal"}\n```', "abnormal"), ('```\n{"state": "normal"}\n```', "normal"),
    ('{\n    "state": "normal"\n}', "normal"),
    ('{"state": "abnormal"} extra', "invalid_json"), ('[{"state": "abnormal"}]', "invalid_json"),
    ('{"state": "maybe"}', "invalid_json"), ('{"verdict": "abnormal"}', "invalid_json"), ("", "invalid_json"),
    ('Based on the visual evidence:\n\n1.  **Gripper', "invalid_json"), (None, "invalid_json"),
    ('{"state": "abnormal"}{"state": "normal"}', "invalid_json"), ('{"state": 1}', "invalid_json"),
])
def test_parse_verdict(text, verdict):
    assert judge.parse_verdict(text) == verdict


def test_classify_keeps_transport_apart_and_enforces_60s():
    assert judge.classify('{"state": "abnormal"}', 59.9) == "abnormal"
    assert judge.classify('{"state": "abnormal"}', 60.01) == "timeout"
    assert judge.classify(None, None) == "timeout"
    assert judge.classify(None, 3.0, "http_error") == "http_error"
    assert judge.classify(None, 3.0, "weird") == "error"
    assert judge.classify("garbage", 3.0) == "invalid_json"
    assert set(judge.SEMANTIC).isdisjoint(judge.TRANSPORT)
    assert judge.flags("abnormal") and not any(judge.flags(v) for v in ("normal", "unsure") + judge.TRANSPORT)


def test_q1_selection_and_frame_identity():
    import q1_qualify as Q
    if not Q.FRAMES.exists():
        pytest.skip("run data not present")
    frames = Q.select_frames(Q.FRAMES, Q.S2_INDEX)
    assert len(frames) == 457
    groups = {}
    for f in frames:
        groups[f["group"]] = groups.get(f["group"], 0) + 1
    assert groups == {"easyS": 120, "hardS": 140, "hardF": 197}
    assert len(frames) + Q.N_REASK <= Q.MAX_CALLS
    # stored S2 frame decodes to 256x256x3 and judge re-encodes it to the identical bytes (C-V2)
    cache = Q.TraceCache()
    fr = Q.load_frame(cache, frames[0])
    assert fr["stored_equals_delivered"] and fr["decision_start_step"] == 5 * frames[0]["decision"]
    assert fr["instruction"] == "put both the alphabet soup and the tomato sauce in the basket"
    body = json.loads(fr["_body"])
    assert body["model"] == "qwen3-vl" and body["max_tokens"] == 64 and body["temperature"] == 0.0


def test_reask_selection_is_stratified_and_deterministic():
    import q1_qualify as Q
    first = {f"k{i}@10": {"verdict": "abnormal" if i % 5 == 0 else ("normal" if i % 5 != 1 else "unsure")}
             for i in range(200)}
    pick = Q.reask_selection(first)
    assert len(pick) == 40 and len(set(pick)) == 40
    assert sum(first[i]["verdict"] == "abnormal" for i in pick) == 20
    assert sum(first[i]["verdict"] == "normal" for i in pick) == 20
    assert pick == Q.reask_selection(dict(reversed(list(first.items()))))
    few = {f"k{i}@10": {"verdict": "abnormal" if i < 5 else "normal"} for i in range(100)}
    p2 = Q.reask_selection(few)
    assert sum(few[i]["verdict"] == "abnormal" for i in p2) == 5 and len(p2) == 40
