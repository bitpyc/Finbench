# AOA（Agent-of-Agents）（StructuredReasoning）

当前我们在用的流程是：
- 先对 test 样本做 **capability / difficulty_score** 标注（`llm_reclassify`）；
- 再把各 base agents 的 `test_results.json` 对齐到这些标签，生成 `capability_eval_mode/**/per_sample.jsonl`；
- 基于各个 base agents 在 **`*_test.jsonl`（online）** 上产生的推理记录/日志，做 **trace-based 经验抽取**；
- 再用抽取到的经验，通过 `Agents.aoa.as_agent_capability_eval` 进行 **样本级路由评测**（默认规则路由；也可开启 router LLM 做样本级路由选择，但下游推理结果仍复用已有 agents 的 `per_sample.jsonl`）。

本仓库里 AOA 主要有四类用法：
- **offline_eval**：只用已有 `per_sample.jsonl` 做统计上界（oracle），不调用 LLM。
- **rule_eval**：只用 `experience.json` 里的规则做确定性路由（if/else），不调用 LLM。
- **as_agent_capability_eval**：把 AOA“物化”为一个新 agent 列：逐样本路由（规则/LLM router 两种），但**推理结果复用**已有 agents 的 `per_sample.jsonl`（不重跑下游 agent 推理）。
- **StructuredReasoning.run --agent_method aoa（真在线 AOA）**：每个样本先路由，再让被选 agent 真的跑一次推理（最慢，最贵）。

## 你应该用哪个脚本？
- **想要“经验更像路由器（看 trace）”**：用 `Agents.aoa.experience_from_traces` 生成 `experience.latest.json`（推荐）。
- **想要“快速从表格总结规则”**：用 `Agents.aoa.experience_extractor`（table-based，速度快但可能偏粗）。
- **想要把 AOA 放进 capability_eval 表格里做对比**：用 `Agents.aoa.as_agent_capability_eval`（规则或 router LLM 两种）。
- **想跑 oracle 上界**：用 `Agents.aoa.offline_eval`。
- **想验证规则质量（不落地 per_sample）**：用 `Agents.aoa.rule_eval`。

## Step 0：准备 capability_eval_mode（生成各 agent 的 per_sample.jsonl）

AOA 的评测/路由（尤其是 `as_agent_capability_eval`）依赖 `capability_eval_mode` 目录下每个 base agent 的：
- `per_sample.jsonl`：每个样本的 `capability / difficulty_score / difficulty_bucket / correct`
- `summary.json`：capability×difficulty 聚合后的准确率

这一步由两部分组成：

### 0.1 跑 base agents 的 online（得到 test_results.json）
你需要先保证 base agents 的 online 结果已经存在，例如：

`results/StructuredReasoning_run/<Task>/<agent_method>/online/<timestamp>/test_results.json`

（这一步通常你已经用 `run_scripts/StructuredReasoning/online/*` 跑过了。）

### 0.2 对 test 做 capability/difficulty_score 标注（llm_reclassify）
产物会写到：

`results/StructuredReasoning_run/llm_reclassify_mode/<output_name>/test/classifications.jsonl`

示例：

```bash
python3 -m utils.llm_reclassify \
  --config_path StructuredReasoning/data/task_config.json \
  --split test \
  --output_name capability_difficulty_score_v1_merged_finer_try50 \
  --output_root results/StructuredReasoning_run/llm_reclassify_mode
```

### 0.3 生成 capability_eval_mode（写 per_sample.jsonl / summary.json / capability_eval.csv）
示例（只聚合 online）：

```bash
python3 -m utils.capability_eval \
  --results_root results/StructuredReasoning_run \
  --classify_root results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v1_merged_finer_try50 \
  --out_dir results/StructuredReasoning_run/capability_eval_mode \
  --only_mode online \
  --export_csv
```

完成后你会得到：
- `results/StructuredReasoning_run/capability_eval_mode/<agent>/online/<timestamp>/per_sample.jsonl`
- `results/StructuredReasoning_run/capability_eval_mode/<agent>/online/<timestamp>/summary.json`
- `results/StructuredReasoning_run/capability_eval_mode/_tables/online/<timestamp>/capability_eval.csv`

## Step 1：Trace-based Experience（从详细 trace 抽取经验 + meta 合成）

当你更希望经验来自“样本级推理痕迹（trace）”而不是聚合表格时，使用：

```bash
python3 -m Agents.aoa.experience_from_traces \
  --results_root results/StructuredReasoning_run \
  --mode online \
  --task_config StructuredReasoning/data/task_config.json \
  --classifications_jsonl results/StructuredReasoning_run/llm_reclassify_mode/capability_difficulty_score_v1_merged_finer_try50/test/classifications.jsonl \
  --k_per_task_agent 50 \
  --seed 42 \
  --workers 8 \
  --meta_max_bullets 600 \
  --api_provider usd_guiji \
  --model USD-guiji/deepseek-v3 \
  --out results/StructuredReasoning_run/aoa_mode/experience/experience.latest.json
```

说明：
- 会在 `.../aoa_mode/experience/<timestamp>/` 下增量落盘：
  - `bullets.jsonl`：每条 bullet 一行
  - `sampled_records.jsonl`：每条 bullet 对应的 (task,agent,index,trace_log)
  - `progress.json`：进度快照
  - `experience.latest.json`：最终汇总（含 meta_experience + routing_policy）
- 支持断点续跑：加 `--resume`
- 支持只重跑 meta 合成（不重新抽 bullets）：加 `--meta_only`
- 默认排除不想看的任务：`--exclude_tasks factset`（默认即为 factset；想包含可传 `--exclude_tasks ""`）
- meta 合成 token 可能爆上下文时：用 `--meta_compact`（压缩 bullets）并调小 `--meta_bullet_max_chars / --meta_diagnosis_max_chars`
- meta 合成输入 bullets 的抽样策略：`--meta_sampling capability_stratified|stratified|shuffle|head`

## Step 2：用经验做样本级路由评测，并落地成新 agent=aoa

```bash
python3 -m Agents.aoa.as_agent_capability_eval \
  --experience_json results/StructuredReasoning_run/aoa_mode/experience/experience.latest.json \
  --cap_eval_root results/StructuredReasoning_run/capability_eval_mode \
  --mode online \
  --out_cap_eval_root results/StructuredReasoning_run/capability_eval_mode
```

输出：
- `results/StructuredReasoning_run/capability_eval_mode/aoa/online/<aligned_ts>/as_agent_capability_eval/per_sample.jsonl`
- `results/StructuredReasoning_run/capability_eval_mode/aoa/online/<aligned_ts>/as_agent_capability_eval/summary.json`

最后再导出总表（现在 `aoa` 会作为新列出现）。

### 规则路由 vs LLM 路由（as_agent_capability_eval）
- **规则路由（默认）**：不加 `--router_llm`，完全确定性、速度快。
- **LLM 路由（样本级 router LLM）**：加 `--router_llm`，但下游 agent 推理结果仍复用已有 `per_sample.jsonl`。

LLM 路由建议同时开启“经验子集（避免上下文爆炸）”：

```bash
python3 -m Agents.aoa.as_agent_capability_eval \
  --experience_json <path_to_experience.json> \
  --cap_eval_root results/StructuredReasoning_run/capability_eval_mode \
  --mode online \
  --timestamp <aligned_ts> \
  --agents cot,self_refine,reflexion,debate,discussion,dynamic_cheatsheet,gepa,ace,amem \
  --router_llm \
  --router_experience_compact \
  --router_experience_max_bullets 120 \
  --router_experience_bullet_max_chars 160 \
  --router_experience_diagnosis_max_chars 80 \
  --task_config StructuredReasoning/data/task_config.json \
  --router_api_provider usd_guiji \
  --router_model USD-guiji/deepseek-v3 \
  --router_max_tokens 512 \
  --router_temperature 0.0 \
  --router_workers 8 \
  --out_cap_eval_root results/StructuredReasoning_run/capability_eval_mode
```

## offline_eval / rule_eval 快速用法

### offline_eval（oracle 上界，不调用 LLM）
输入是已有 `capability_eval_mode/<agent>/<mode>/<timestamp>/per_sample.jsonl`。

```bash
python3 -m Agents.aoa.offline_eval \
  --cap_eval_root results/StructuredReasoning_run/capability_eval_mode \
  --mode online \
  --timestamp <aligned_ts> \
  --agents cot,self_refine,reflexion,debate,discussion,dynamic_cheatsheet,gepa,ace,amem \
  --policy capdiff_best
```

### rule_eval（验证 routing_policy 规则质量，不落地 per_sample）
```bash
python3 -m Agents.aoa.rule_eval \
  --experience_json <path_to_experience.json> \
  --cap_eval_root results/StructuredReasoning_run/capability_eval_mode \
  --mode online \
  --timestamp <aligned_ts> \
  --agents cot,self_refine,reflexion,debate,discussion,dynamic_cheatsheet,gepa,ace,amem
```




