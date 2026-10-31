#!/usr/bin/env bash
# FastHarness RoboDojo runs (predeclared subset in docs/results/fast_harness.md), sequential on GPU 4.
# usage: run_stage_d.sh D1 D2 D3   (chained development on a repeated layout; dev-only diagnostic)
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

prev=$RUNS/fh_prepare_seed1
for step in "$@"; do
  case $step in
    D1|D2|D3|D4|D5) run fh_$step hs prepare fast-harness --config "$CFG/robodojo_build_tower_stageD.yaml" \
               --prepared "$prev" --case-namespace "seed1-rep${step#D}" --out "$RUNS/fh_$step" "${COMMON[@]}" || exit 1
            prev=$RUNS/fh_$step ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
