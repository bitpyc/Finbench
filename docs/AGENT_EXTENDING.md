# Agents 扩展规范（详细版）

本文档旨在帮助**未接触过本项目**的开发者快速理解 Agent 扩展流程。内容覆盖架构全景、接口约定、样例代码、脚本联动与验证 checklist，确保新方法可以快捷落地。

---

## 1. 全局架构速览

```
CLI (bizbench/run.py)
├── 解析命令行参数 (--agent_method / --task_name / --mode / ...)
├── 加载配置 + 数据 → DataProcessor
└── 根据 agent_method 分派到 Agents/ 子包

Agents/
├── ace/  → 完整 ACE 流程：Generator → Reflector → Curator → Playbook
└── cot/  → 轻量 Baseline：单 Generator + 统一 Prompt

DataProcessor
└── 负责样本预处理、答案判定、准确率计算

Results
└── <save_path>/<task>/<agent>/<mode>/<timestamp>/...
```

### 流程对比（ACE vs CoT）

| 步骤 | ACE | CoT baseline |
| --- | --- | --- |
| Prompt | 三角色不同 Prompt（详见 `Agents/ace/prompts/`） | 单 Prompt，输出 JSON（`Agents/cot/generator.py`） |
| 反馈循环 | 生成 → 反思 → 策展（更新 playbook） | 无反馈，单次生成 |
| 支持模式 | offline / online / eval_only | online / eval_only（可扩展） |
| 结果 | `final_results.json`、`train_results.json`、Playbook | `test_results.json` + `run_config.json` |

---

## 2. Quick Start Checklist

1. 阅读本文档，确认需要支持的模式与功能。
2. 在 `Agents/` 下创建新子包，仿照 `ace` / `cot`。
3. 实现核心组件 & Agent 类，遵守统一接口。
4. 修改 `bizbench/run.py` 将 `--agent_method` 映射到新 Agent。
5. 更新相应 shell 脚本（如有需要）。
6. 执行 smoke test，确认目录结构与结果输出正常。
7. 文档化使用方式（README / docs）。

---

## 3. 目录结构与职责详情

```
Agents/
├── __init__.py                       # 统一导出
├── ace/
│   ├── ace.py                        # ACE System 主类
│   ├── core/                         # Generator / Reflector / Curator 实现
│   └── prompts/                      # 三角色 Prompt 模板
└── cot/
    ├── agent.py                      # CoT Agent（run() 逻辑）
    └── generator.py                  # Prompt builder + timed_llm_call
```

- **BizBench 入口**：`bizbench/run.py` 通过 `--agent_method` 选择 Agent；ACE 默认，CoT 作为 baseline。
- **结果目录规范**：所有 Agent 写入 `save_dir/<task>/<agent>/<mode>/<timestamp>/`，并在 `run_config.json` 中记录 `run_subdir`，便于追踪。

---

## 4. 统一接口与代码模板

### 4.1 Agent 构造签名

```python
class YourAgent:
    def __init__(
        self,
        api_provider: str,
        generator_model: str,
        max_tokens: int,
        agent_method: str,
        **kwargs,
    ):
        self.agent_method = agent_method
        self.generator = ...  # 初始化模型 client
```

> 如需 Reflector/Curator，可在此处依次初始化。

### 4.2 run() 方法骨架

```python
def run(
    self,
    mode: str,
    train_samples: Optional[List[Dict[str, Any]]],
    val_samples: Optional[List[Dict[str, Any]]],
    test_samples: Optional[List[Dict[str, Any]]],
    data_processor,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    supported = {"online", "eval_only"}  # 根据需求调整
    if mode not in supported:
        raise ValueError(f"{self.agent_method} only supports {supported}")

    task_name = config["task_name"]
    save_dir = config["save_dir"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subdir = os.path.join(task_name, self.agent_method, mode, timestamp)
    resolved_save_path = os.path.join(save_dir, run_subdir)
    os.makedirs(resolved_save_path, exist_ok=True)

    # 执行主要逻辑（评测 / 训练）
    results, error_log = evaluate_test_set(...)

    # 写入配置与结果
    with open(os.path.join(resolved_save_path, "run_config.json"), "w") as f:
        json.dump({"config": config, "run_subdir": run_subdir}, f, indent=2)

    with open(os.path.join(resolved_save_path, "test_results.json"), "w") as f:
        json.dump({"test_results": results, "error_log": error_log}, f, indent=2)

    return results
```

### 4.3 Prompt 示例（CoT）

```python
prompt_lines = [
    "You are a finance-domain reasoning assistant...",
    "Return JSON with keys reasoning / bullet_ids / final_answer",
    "- reasoning: ...",
    "- bullet_ids: [] because no playbook",
    "- final_answer: ...",
    "",
    "Question:",
    question_text,
    "",
    "Context:",
    context_text,
    "",
    "Respond strictly in JSON.",
]
```

> 统一输出格式可直接被 `utils.extract_answer()` 解析。

---

## 5. BizBench 入口对接

### 5.1 命令行参数

`bizbench/run.py` 中的 `--agent_method` 需包含新方法，例如：

```python
parser.add_argument(
    "--agent_method",
    choices=["ace", "cot", "your_agent"],
    ...
)
```

### 5.2 main() 分支

```python
if args.agent_method == "ace":
    ace_system = ACE(..., agent_method=args.agent_method)
    ace_system.run(...)
elif args.agent_method == "cot":
    cot_agent = ChainOfThoughtAgent(...)
    cot_agent.run(...)
else:
    new_agent = YourAgent(..., agent_method=args.agent_method)
    new_agent.run(...)
```

### 5.3 Shell 脚本

`run_scripts/binbenzh/...` 中的脚本已经变量化，新增 agent 时可参照以下模板：

```bash
BENCHMARK_MODULE="bizbench.run"
BENCHMARK_NAME="bizbench"
AGENT_METHOD="your_agent"
TASK_NAME="CodeFinQA"
MODE="online"
CONFIG_PATH="bizbench/data/task_config.json"
SAVE_PATH="results/${BENCHMARK_NAME}_run"
LOG_NAME="${BENCHMARK_NAME}_run_${TASK_NAME}_${AGENT_METHOD}_${MODE}.log"

nohup python -m "${BENCHMARK_MODULE}" \
  --agent_method "${AGENT_METHOD}" \
  --task_name "${TASK_NAME}" \
  --mode "${MODE}" \
  --config_path "${CONFIG_PATH}" \
  --save_path "${SAVE_PATH}" \
  "$@" \
  > "${LOG_NAME}" 2>&1 &
```

---

## 6. DataProcessor 与工具复用

- **DataProcessor 职责**（位于 `bizbench/data_processor.py`）：
  - `process_task_data`：将原始样本转为标准字段（context/question/target）。
  - `answer_is_correct(pred, target)`：判断模型输出是否正确。
  - `evaluate_accuracy(answers, targets)`：统计准确率。
- **评测工具**：`utils.evaluate_test_set()` 会调用 `generator.generate()`、解析输出并交给 DataProcessor 判分。所有 Agent 建议复用该函数，以获得一致的 `test_results` 格式。
- **模型输出**：确保 Prompt 的 `final_answer` 字段能被 `utils.extract_answer()` 提取；如有自定义格式，需同步修改解析逻辑。

---

## 7. 注意事项

1. **模式支持**：如不支持 `offline`，需显式 `raise ValueError`，避免静默错误。
2. **API Client**：推荐使用 `utils.initialize_clients(api_provider)` 初始化，保证与 ACE 一致的重试/日志策略。
3. **日志与错误**：
   - ACE 使用 `logger.py` 记录细粒度调用；轻量 Agent 至少打印关键节点（启动、样本数、准确率）。
   - `timed_llm_call()` 已内置重试与问题请求记录，可复用。
4. **结果命名**：
   - 必须生成 `run_config.json`（含 `run_subdir`），`test_results.json`，必要时包含 `final_results.json`、`train_results.json` 等。
   - playbook 相关文件只在需要时写入。
5. **Prompt 约束**：若没有 playbook，需在 Prompt 中明确说明以避免模型引用不存在的内容（如 CoT 的 `bullet_ids` 永远为空）。
6. **目录结构**：任何新 Agent 都应遵循 `<task>/<agent>/<mode>/<timestamp>`，以便离线结果管理工具（如 `results/bizbench_run/...`）自动识别。

---

## 8. 验证 Checklist

1. **命令行运行**  
   ```bash
   python -m bizbench.run \
     --agent_method your_agent \
     --task_name FinCode \
     --mode online \
     --save_path results/bizbench_run \
     --config_path bizbench/data/task_config.json
   ```
2. **目录检查**  
   - `results/bizbench_run/FinCode/your_agent/online/<timestamp>/` 是否存在。
   - `run_config.json` 中是否记录 `run_subdir` 和正确配置。
3. **结果文件**  
   - `test_results.json` / `final_results.json`（如适用）格式正确，可读。
   - `detailed_llm_logs/` 等日志目录是否创建。
4. **日志输出**  
   - 控制台/日志文件中是否包含启动提示、样本数、准确率等关键信息。
5. **Glue 脚本**  
   - 如有对应 shell 脚本，`nohup` 输出是否指向正确日志文件。
6. **断言**  
   - 如果 Agent 不支持某模式，运行时是否会抛出友好的错误信息。

通过以上步骤，即可确认新的 Agent 方法已完整融入当前 BizBench 体系。

---

## 9. 示例参考

- **ACE**：`Agents/ace/ace.py`（多角色协作 + playbook 演化）；适合需要复杂记忆管理的 Agent。
- **CoT**：`Agents/cot/agent.py` + `generator.py`（单 Prompt Baseline）；适合作为最小实现模板。

在这两者之间，你可以实现任意复杂度的方案，只要遵循本文档约定，便能快速集成到现有 CLI、结果存储与日志体系中。

