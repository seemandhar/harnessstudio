#!/usr/bin/env bash
# FastHarness RoboDojo runs (predeclared subset in docs/results/fast_harness.md), sequential on GPU 4.
# usage: run_stage_c.sh STEP [STEP ...]   steps: C_prepare | C_cold | C_warm | C_base | C_null (stack_bowls)
# The OpenAI key is loaded from the private env file into THIS process tree only (never printed or logged).
set -o pipefail
ROOT=/home/sejain/research/cvpr27/RoboMetaHarness-claude/harnessstudio
CFG=$ROOT/methods/fast_harness/configs
RUNS=${FH_RUNS:-/bigdata/sejain/hs2-cache/robodojo/runs}
POLICY=ws://127.0.0.1:8311
COMMON=(--policy-url "$POLICY" --contract robodojo --profile robodojo-pi05-hs --no-docker --policy-timeout 900)

source /bigdata/sejain/hs2-cache/envs/hs2_robodojo_env.sh
set -a; . /home/sejain/research/cvpr27/RoboMetaHarness/.codex/agents/.env; set +a
cd "$ROOT"

run() {  # name, command...
  local name=$1; shift
  echo "[$(date -Is)] START $name: $*" | tee -a "$RUNS/stage.log"
  "$@" > "$RUNS/$name.log" 2>&1
  local rc=$?
  echo "[$(date -Is)] END $name rc=$rc" | tee -a "$RUNS/stage.log"
  return $rc
}

for step in "$@"; do
  case $step in
    C_prepare) run fh_C_prepare_seed1 hs prepare fast-harness --config "$CFG/robodojo_stack_bowls_dev.yaml" \
               --case-namespace seed1 --out "$RUNS/fh_C_prepare_seed1" "${COMMON[@]}" ;;
    C_cold) run fh_C_cold hs eval fast-harness --config "$CFG/robodojo_stack_bowls.yaml" --arms candidate \
               --out "$RUNS/fh_C_cold" "${COMMON[@]}" ;;
    C_warm) run fh_C_warm hs eval fast-harness --config "$CFG/robodojo_stack_bowls.yaml" --arms candidate \
               --prepared "$RUNS/fh_C_prepare_seed1" --out "$RUNS/fh_C_warm" "${COMMON[@]}" ;;
    C_base) run fh_C_base hs eval fast-harness --config "$CFG/robodojo_stack_bowls.yaml" --arms baseline \
               --out "$RUNS/fh_C_base" "${COMMON[@]}" ;;
    C_null) run fh_C_null hs eval fast-harness --config "$CFG/robodojo_stack_bowls.yaml" --arms null \
               --out "$RUNS/fh_C_null" "${COMMON[@]}" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
