# FirmBench

<!-- Introduction 暂略 -->

## Quick Start（仅 Online 模式）

下面以 **StructuredReasoning** 与 **Consulting** 两个子基准为例说明如何运行 Online 实验、以及如何汇总 capability / difficulty 维度的结果。

> 约定：所有命令都在仓库根目录执行（即本仓库顶层目录）。

### 0) 前置准备

- **Python**：确保可用 `python` / `python3`（建议 Python 3.10+）。
- **LLM 调用配置**：脚本中默认使用 `--api_provider usd_guiji`（以及对应模型名），请按你的环境配置好对应的密钥/网关/SDK。
- **Online 模式数据**：Online 模式只会用到各任务的 **`*_test.jsonl`**（例如 `StructuredReasoning/data/*_test.jsonl`）。

### 1) 生成 capability / difficulty 划分（LLM reclassify）

在做 capability/difficulty 维度汇总之前，需要先对样本做 LLM 重标注，产物会写入：

- `results/StructuredReasoning_run/llm_reclassify_mode/<output_name>/<split>/classifications.jsonl`

运行（Online 模式通常只需要 `test`）：

```bash
bash run_scripts/StructuredReasoning/data_process/reclassify_test.sh
```

可选（如你也需要 train/val 的标注）：

```bash
bash run_scripts/StructuredReasoning/data_process/reclassify_train.sh
bash run_scripts/StructuredReasoning/data_process/reclassify_valid.sh
```

### 2) StructuredReasoning：运行各智能体 Online 实验

StructuredReasoning 的 Online 脚本位于 `run_scripts/StructuredReasoning/online/<agent_method>/*.sh`，
运行后会在 `results/StructuredReasoning_run/<Task>/<agent_method>/online/<timestamp>/` 下生成对应的 `test_results.json` 等文件。

为了避免读者逐个 `bash`，可以一键启动所有 agent 的 Online 脚本（会排除 `capability_eval` 目录）：

```bash
bash run_scripts/StructuredReasoning/online/run_all_agents_online.sh
```

> 说明：这些脚本大多使用 `nohup ... &` 后台启动，会同时起很多进程；如需控制规模，请自行挑选 agent 或数据集脚本运行。

### 3) StructuredReasoning：按 capability / difficulty 汇总 Online 结果

当 `results/StructuredReasoning_run/**/online/**/test_results.json` 已生成，并且第 1 步的 `classifications.jsonl` 已准备好后，运行：

```bash
bash run_scripts/StructuredReasoning/online/capability_eval/online/capability_eval_online.sh
```

默认输出目录：

- `results/StructuredReasoning_run/capability_eval_mode/`
- CSV 表格通常会在：`results/StructuredReasoning_run/capability_eval_mode/_tables/online/<timestamp>/capability_eval.csv`

### 4) Consulting：运行各智能体 Online 实验

Consulting 的 Online 脚本位于 `run_scripts/Consulting/online/*.sh`，运行后会在 `results/Consulting/<agent_method>/online/<timestamp>/` 生成 `test_results.json` 等文件。

同样提供一键启动：

```bash
bash run_scripts/Consulting/online/run_all_online.sh
```


