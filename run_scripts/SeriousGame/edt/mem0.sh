#!/usr/bin/env bash
# Run EDT (Enterprise Digital Twin) serious game evaluation with A-mem agent.
#
# NOTE: EDT evaluation is scenario-level and requires a *BPTK template repo root*
# that contains BOTH:
#   - scenarios/interactive.json
#   - simulation_models/ (Python package used by the scenario)
#
# If you cloned the official tutorial repo, a typical value is:
#   <bptk_py_tutorial-master>/model_library/enterprise_digital_twin
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PROJECT_ROOT"

SAVE_DIR="results"

python -m SeriousGame.run_edt \
  --mode eval_only \
  --agent_method mem0 \
  --api_provider openai \
  --generator_model deepseek-v3 \
  --max_tokens 4096 \
  --save_dir "${SAVE_DIR}" \
  --bptk_script SeriousGame/run_bptk_server.py \
  --bptk_host 127.0.0.1 \
  --edt_mcp_server_path SeriousGame/edt_mcp_server_local.py \
  --bptk_repo_root ./SeriousGame \
  "$@"
