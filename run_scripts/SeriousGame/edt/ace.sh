#!/usr/bin/env bash
# ACE EDT evaluation script
# 放置路径：run_scripts/SeriousGame/edt/ace.sh
# 用法：
#   chmod +x ace.sh
#   ./ace.sh                          # 使用默认数据路径
#   ./ace.sh path/to/your_edt.json    # 或显式指定数据文件

set -euo pipefail

SAVE_DIR="results"

echo "[ACE][EDT] SAVE_DIR=${SAVE_DIR}"

mkdir -p "${SAVE_DIR}"

python -m SeriousGame.run_edt \
  --api_provider "openai" \
  --generator_model "deepseek-v3" \
  --save_dir "${SAVE_DIR}" \
  --mode "eval_only" \
  --max_tokens 1024 \
  --agent_method "ace" \
  --reflector_model "deepseek-v3" \
  --curator_model "deepseek-v3" \
  --bptk_script SeriousGame/run_bptk_server.py \
  --bptk_host 127.0.0.1 \
  --edt_mcp_server_path SeriousGame/edt_mcp_server_local.py \
  --bptk_repo_root ./SeriousGame \

echo "[ACE][EDT] Done."
