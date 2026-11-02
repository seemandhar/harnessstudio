"""K1-c1 tier-0 evidence: the surface-control prompt (original + ' .') and the clause prompt tokenize differently from
the original instruction in the pi0-FAST prefix, using openpi's FASTTokenizer prompt path (openpi rev 981483d,
openpi/models/tokenizer.py L67-75: cleaned = prompt.lower().strip().replace('_', ' '); prefix = f"Task: {cleaned},
State: {state_str};\\n"; PaliGemma sentencepiece encode(prefix, add_bos=True)). The state string is a fixed dummy
(the same in every prompt), so only the text part can differ.

Needs sentencepiece (not in harnessstudio/.venv; run with an interpreter that has it, e.g. the system python3):
    python3 harnessstudio/methods/bcp/scripts/tokenize_prompts.py --out OUT.json
CPU only; reads ~/.cache/openpi/big_vision/paligemma_tokenizer.model (the file openpi downloads).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import importlib.util  # noqa: E402

import sentencepiece  # noqa: E402

_spec = importlib.util.spec_from_file_location("bcp_rules", Path(__file__).resolve().parents[1] / "hs_bcp/rules.py")
_rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rules)   # rules.py only (no harnessstudio import needed)
control_prompt, split_clause = _rules.control_prompt, _rules.split_clause

MODEL = Path.home() / ".cache/openpi/big_vision/paligemma_tokenizer.model"
INSTR = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]
STATE_STR = " ".join(["128"] * 8)


def prefix_tokens(sp, prompt: str) -> list:
    cleaned = prompt.lower().strip().replace("_", " ")
    return sp.encode(f"Task: {cleaned}, State: {STATE_STR};\n", add_bos=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = MODEL.read_bytes()
    sp = sentencepiece.SentencePieceProcessor(model_proto=data)
    rows, ok = [], True
    for t, s in enumerate(INSTR):
        base = prefix_tokens(sp, s)
        ctl = prefix_tokens(sp, control_prompt(s))
        clause = split_clause(s)
        row = {"task": t, "instruction": s, "control": control_prompt(s), "clause": clause,
               "n_tokens_original": len(base), "n_tokens_control": len(ctl),
               "control_differs": ctl != base,
               "control_extra_pieces": [sp.id_to_piece(i) for i in ctl if i not in base][:4]}
        ok &= row["control_differs"]
        if clause is not None:
            cl = prefix_tokens(sp, clause)
            row["n_tokens_clause"] = len(cl)
            row["clause_differs"] = cl != base
            ok &= row["clause_differs"]
        rows.append(row)
    out = {"tokenizer_model": str(MODEL), "tokenizer_sha256": hashlib.sha256(data).hexdigest(),
           "openpi_prompt_path": "openpi/models/tokenizer.py FASTTokenizer.tokenize L67-75 (rev 981483d)",
           "all_control_and_clause_prompts_tokenize_differently": bool(ok), "rows": rows}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"ok": bool(ok), "sha": out["tokenizer_sha256"][:16]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
