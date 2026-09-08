#!/usr/bin/env bash
set -u

# Fixed two-arm RealMemBench ablation runner. This intentionally does not
# implement parameter sweep: once quota is available, it resumes each arm
# until a finished manifest exists.
ROOT=/data/wz/agent_memory/iconip2026/TCMem
PY=/data/wz/anaconda3/envs/tcmem/bin/python
ENV_FILE="$ROOT/.llm_runtime_qwen/grok.env"
RUN_ROOT=/data/wz/agent_memory/iconip2026/work/ablation_20260908
DATASET=/data/wz/agent_memory/iconip2026/RealMemBench/dataset/Adeleke_Okonjo_dialogues_256k.json

cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

load_env() {
  set -a
  . "$ENV_FILE"
  set +a
}

status_finished() {
  "$PY" - "$1" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "manifest.json"
try:
    print(json.loads(p.read_text()).get("status") == "finished")
except Exception:
    print(False)
PY
}

preflight() {
  "$PY" - <<'PY'
from tcmem.utils.llm_client import OpenAICompatibleLLMClient
import os
c = OpenAICompatibleLLMClient(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
    model=os.environ["OPENAI_MODEL"],
    timeout=int(os.environ.get("OPENAI_TIMEOUT", "120")),
)
try:
    c.generate("Reply with OK.", system_prompt="availability check", temperature=0.0, max_tokens=16)
except Exception as exc:
    print(f"llm_preflight_failed: {type(exc).__name__}: {exc}", flush=True)
    raise SystemExit(1)
print("llm_preflight_ok", flush=True)
PY
}

run_arm() {
  local arm="$1" gpu="$2" module="$3" source="$4"
  local out="$RUN_ROOT/${arm}_final/results"
  local logs="$RUN_ROOT/${arm}_final/logs"
  mkdir -p "$out" "$logs"
  if [[ "$(status_finished "$out")" == "True" ]]; then
    echo "[$arm] already finished"
    return 0
  fi
  echo "[$arm] resume_from=$source gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m "$module" \
    --dataset "$DATASET" \
    --model "$OPENAI_MODEL" \
    --build-model "$OPENAI_MODEL" \
    --eval-model "$OPENAI_MODEL" \
    --embedding-model /data/wz/models/bgem3 \
    --embedding-device cuda \
    --vector-index-backend chroma \
    --session-ks 5,10,20,30 \
    --retrieval-record-k 200 \
    --evidence-top-k 20 \
    --run-name "${arm}_final" \
    --output-dir "$out" \
    --log-dir "$logs" \
    --resume-from "$source" \
    --skip-llm-preflight \
    --state-save-every-records 10 \
    --progress-every-records 1 \
    --fail-fast --verbose > "$RUN_ROOT/${arm}_final/run.stdout.log" 2>&1
  "$PY" scripts/audit_full_baseline_semantics.py "$out" \
    --output "$out/semantic_audit.json" >> "$RUN_ROOT/${arm}_final/run.stdout.log" 2>&1 || true
}

while :; do
  # Reload the private file each cycle so a refreshed key/model takes effect
  # without requiring a new SSH session.
  load_env
  if ! preflight; then
    echo "[$(date -Is)] LLM preflight unavailable; retrying in 10 minutes" >&2
    sleep 600
    continue
  fi

  baseline_source="$RUN_ROOT/full_baseline/results"
  [[ -f "$RUN_ROOT/full_baseline_final/results/memory_state_latest.json" ]] && baseline_source="$RUN_ROOT/full_baseline_final/results"
  no_chain_source="$RUN_ROOT/full_no_task_chain/results"
  [[ -f "$RUN_ROOT/full_no_task_chain_final/results/memory_state_latest.json" ]] && no_chain_source="$RUN_ROOT/full_no_task_chain_final/results"

  run_arm full_baseline 0 tcmem.evals.realmem_top_session "$baseline_source" &
  pid_baseline=$!
  run_arm full_no_task_chain 1 tcmem.evals.realmem_no_task_chain "$no_chain_source" &
  pid_no_chain=$!
  wait "$pid_baseline"; rc_baseline=$?
  wait "$pid_no_chain"; rc_no_chain=$?

  baseline_done="$(status_finished "$RUN_ROOT/full_baseline_final/results")"
  no_chain_done="$(status_finished "$RUN_ROOT/full_no_task_chain_final/results")"
  if [[ "$baseline_done" == "True" && "$no_chain_done" == "True" ]]; then
    echo "ablation_finished"
    exit 0
  fi
  echo "arms incomplete (baseline_rc=$rc_baseline no_chain_rc=$rc_no_chain); retrying in 10 minutes" >&2
  sleep 600
done
