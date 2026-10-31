#!/usr/bin/env bash
# FastHarness RoboDojo runs (predeclared subset in docs/results/fast_harness.md), sequential on GPU 4.
# usage: run_stage.sh STEP [STEP ...]   steps: smoke | tier2 | prepare | warm | stageB_cold | stageB_warm |
#                                               stageB_base | stageB_null
# The OpenAI key is loaded from the private env file into THIS process tree only (never printed or logged).
set -o pipefail
ROOT=/home/sejain/research/cvpr27/harnessstudio-v2
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
    smoke)   run fh_smoke hs eval fast-harness --config "$CFG/robodojo_build_tower_smoke.yaml" --arms candidate \
               --benchmark-overrides '{"max_steps": 60}' --out "$RUNS/fh_smoke" "${COMMON[@]}" ;;
    tier2)   run fh_tier2_seed0 hs verify fast-harness --tier 2 --config "$CFG/robodojo_build_tower.yaml" \
               --out "$RUNS/fh_tier2_seed0" "${COMMON[@]}" ;;
    prepare) run fh_prepare_seed1 hs prepare fast-harness --config "$CFG/robodojo_build_tower_dev.yaml" \
               --case-namespace seed1 --out "$RUNS/fh_prepare_seed1" "${COMMON[@]}" ;;
    warm)    run fh_warm_seed0 hs eval fast-harness --config "$CFG/robodojo_build_tower.yaml" --arms candidate \
               --prepared "$RUNS/fh_prepare_seed1" --out "$RUNS/fh_warm_seed0" "${COMMON[@]}" ;;
    stageB_cold) run fh_B_cold hs eval fast-harness --config "$CFG/robodojo_build_tower_stageB.yaml" --arms candidate \
               --case-namespace seed2 --out "$RUNS/fh_B_cold" "${COMMON[@]}" ;;
    stageB_warm) run fh_B_warm hs eval fast-harness --config "$CFG/robodojo_build_tower_stageB.yaml" --arms candidate \
               --case-namespace seed2 --prepared "$RUNS/fh_prepare_seed1" --out "$RUNS/fh_B_warm" "${COMMON[@]}" ;;
    stageB_base) run fh_B_base hs eval fast-harness --config "$CFG/robodojo_build_tower_stageB.yaml" --arms baseline \
               --case-namespace seed2 --out "$RUNS/fh_B_base" "${COMMON[@]}" ;;
    stageB_null) run fh_B_null hs eval fast-harness --config "$CFG/robodojo_build_tower_stageB.yaml" --arms null \
               --case-namespace seed2 --out "$RUNS/fh_B_null" "${COMMON[@]}" ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
