"""Inspect the shared base without loading models or calling APIs."""
import argparse
import hashlib
import json
from pathlib import Path


def source_manifest():
    directory = Path(__file__).resolve().parent
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.glob("*.py"))}


def describe():
    from . import VERSION
    return {"name": "Unified Harness", "version": VERSION, "training_steps": 0,
        "default": "native-preserving identity selection, one proposal, empty frozen memory, no operator",
        "profiles": {"native": {"samples": 1, "operator_calls": 0},
                     "reference": {"samples": 1, "operator_calls": 0,
                         "policy_port": "existing reference harness with its frozen configuration"},
                     "consensus": {"requires": ["independent_sampling", "isolated_sampling", "action_geometry"],
                         "required_port_method": "infer_alternative",
                         "status": "optional experimental ablation; not a validated universal improvement"}},
        "extension_points": ["policy_port", "instruction_operator", "frozen_memory", "selector", "advisory_monitor", "trace_sink", "native_null_parity_guard"],
        "source_sha256": source_manifest(),
        "evaluation": "existing Campaign gates remain authoritative; pilot parity is not promotion"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["describe", "manifest"], nargs="?", default="describe")
    a = p.parse_args(argv)
    print(json.dumps(describe() if a.command == "describe" else source_manifest(), indent=2))


if __name__ == "__main__":
    main()
