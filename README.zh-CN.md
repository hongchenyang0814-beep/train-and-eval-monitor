# Train and Eval Monitor

[English](README.md)

Train and Eval Monitor 是一个面向 Codex 的 Skill 和零第三方依赖 Python 工具，统一处理两类工作：

- 同步 StreamLake / 万擎平台的微调与正式评测记录；
- 在同一个可通过 SSH 访问的网页中查看本地 LLaMA-Factory 训练监控和评测日志分析。

它会在受控流程允许时下载完整评测日志，但不会下载模型权重、checkpoint、数据集或预测文件。

## 强制路径约束

每台机器的用户目录、代码目录、训练输出目录、单次运行目录和 SSH 配置都不同。本仓库只提供通用模板，不是作者机器的配置副本。

每个使用者在启动前必须完成以下检查：

1. 选择当前机器上的 Skill 安装目录；
2. 默认目录不合适时，显式传入 `--target-dir` 和 `--output-dir`；
3. 用 `--training-output-root` 传入一个或多个**当前机器上已经存在的绝对训练输出目录**，或设置支持的环境变量；
4. 打开 `<target-dir>/training_monitor_config.json`，确认 `outputs_roots` 和 `targets.*` 中的每个路径都属于当前机器；
5. 如果配置了显式任务，必须逐项替换 `output_dir`、`metrics_path`、`log_path` 和 `config_path`。
6. 加载已有运行目录后，执行 `python "$MONITOR_TOOL" repair-paths`，或部署时添加 `--repair-paths`，自动删除另一台机器遗留的失效路径。

禁止直接复制另一台机器的 `/root/...`、`/home/...`、Windows 盘符路径、个人 SSH 别名、Cookie、project ID、Hugging Face Token、PID、缓存或服务日志。路径不存在时必须修正或删除。这是本项目的硬性约束。

新部署会保守地自动检查以下来源：命令行 `--training-output-root`、环境变量 `STREAMLAKE_TRAINING_OUTPUT_ROOT`、`LLAMA_FACTORY_OUTPUT_ROOT`、`LLAMA_FACTORY_OUTPUT_DIR`、`TRAINING_OUTPUT_ROOT`、`TRAINING_OUTPUT_DIR`，以及当前用户的 `~/output`、`~/LLaMA-Factory/output`、`./output`。只有已经存在的目录才会写入配置，不会扫描整个文件系统，也不会覆盖已经存在的运行配置。

固定的运行目录模板是机器相关的 `$HOME/.local/share/streamlake-eval-monitor`，其下保存 `config/` 和 `data/`。训练目录先以 `<TRAINING_OUTPUT_ROOT>` 占位，部署时再由当前机器自动发现或显式传入。`repair-paths` 会在当前电脑上解析模板、删除失效的 `outputs_roots` 和 `targets`，保留真实存在的本地路径。

训练输出目录尽量统一为参考 Monitor 的格式：`<TRAINING_OUTPUT_ROOT>/lora_sft/<run-id>/`、`<TRAINING_OUTPUT_ROOT>/ai_infra/benchmark/<run-id>/` 或 `<TRAINING_OUTPUT_ROOT>/ai_infra/profiler/<run-id>/`。每个运行目录内放置 `training_config.yaml`、`trainer_log.jsonl`、`run_manifest.json` 和生成的 `training_task.md`；checkpoint 放在该目录下的 `checkpoint-<step>/`。其他分类仍会被扫描，但只有配置了对应 profile 后才允许 checkpoint 上传。

## 安装

要求：Git、Python 3.10 或更高版本，以及 Codex。

```bash
SKILL_DIR="$HOME/.codex/skills/streamlake-experiment-analyst"
mkdir -p "$(dirname "$SKILL_DIR")"
git clone "https://github.com/<OWNER>/train-and-eval-monitor.git" "$SKILL_DIR"
python "$SKILL_DIR/scripts/streamlake_experiments.py" --help
```

将 `<OWNER>` 替换为项目维护者提供的仓库所有者；如果你已经拿到完整克隆地址，直接使用那个地址即可。

重启 Codex 或新建任务，让 Skill 重新被发现。GitHub 仓库名称是 `Train and Eval Monitor`；为了兼容已有提示，Codex Skill 标识仍是 `streamlake-experiment-analyst`。

更新已有安装：

```bash
git -C "$HOME/.codex/skills/streamlake-experiment-analyst" pull --ff-only
```

## 按当前机器路径部署

以下命令在训练/评测服务器上执行。把训练输出目录替换为服务器上的真实绝对路径：

```bash
SKILL_DIR="$HOME/.codex/skills/streamlake-experiment-analyst"
MONITOR_TOOL="$SKILL_DIR/scripts/eval_monitor.py"
TRAINING_ROOT="/absolute/path/to/your/training/output"
TARGET_DIR="$HOME/.local/share/train-and-eval-monitor"

python "$MONITOR_TOOL" deploy \
  --target-dir "$TARGET_DIR" \
  --output-dir "$TARGET_DIR/data" \
  --training-output-root "$TRAINING_ROOT"
```

如果训练输出分散在多个目录，可以重复 `--training-output-root`。如果输出 `Training output roots: none detected`，必须先编辑 `<target-dir>/training_monitor_config.json` 再使用训练页。服务默认只监听 `127.0.0.1:18280`。

管理服务：

```bash
python "$MONITOR_TOOL" status --target-dir "$TARGET_DIR"
python "$MONITOR_TOOL" stop --target-dir "$TARGET_DIR"
python "$MONITOR_TOOL" start --target-dir "$TARGET_DIR"
```

再次执行 `deploy` 会升级网页和后端文件，但会保留运行配置、凭据、缓存、上传 registry、PID 和日志。升级后仍要重新检查路径；Skill 不会静默改写已有个人配置。

## 配置训练上传

训练页右上角齿轮只负责保存或替换 Hugging Face Write Access Token。Token 保存于 `<target-dir>/config/huggingface_token`，权限为 `600`，不会回显，也不会进入训练配置。

仓库创建和 checkpoint 上传按运行配置中的“分类 profile”执行。profile 的 key 与真实目录分类精确对应：一级分类使用 `<输出根目录>/<分类>/<run-id>/`，二级分类使用 `<输出根目录>/<分类>/<子分类>/<run-id>/`。可上传的运行目录必须同时存在 `trainer_log.jsonl` 和 `training_config.yaml`，且 `run_id` 必须等于目录名。没有精确匹配 profile 的分类仍会被监控，但不能上传。每个要上传的分类应配置独立的仓库前缀：

```json
{
  "outputs_roots": ["/absolute/path/to/your/training/output"],
  "targets": [],
  "auto_upload": {
    "config_file": "training_config.yaml",
    "profiles": {
      "lora_sft": {
        "hub_endpoint": "https://huggingface.co",
        "hub_owner": "YOUR_NAMESPACE",
        "hub_repo_prefix": "lora-sft",
        "hub_private_repo": true,
        "hub_index_repo_id": "YOUR_NAMESPACE/lora_sft",
        "base_model_id": "ORG/BASE_MODEL"
      },
      "ai_infra/benchmark": {
        "hub_endpoint": "https://huggingface.co",
        "hub_owner": "YOUR_NAMESPACE",
        "hub_repo_prefix": "benchmark",
        "hub_private_repo": true,
        "base_model_id": "ORG/BASE_MODEL"
      }
    }
  }
}
```

以后路径变化时，只需增删或修改 `profiles` 的 key。key 是 `outputs_roots` 下、最终运行目录之前的相对路径。即使显式添加 `targets`，也必须遵循同一目录规则：`config_path` 必须是运行目录内的 `training_config.yaml`，`run_id` 必须与目录名一致。Monitor 会按 `<owner>/<prefix>-<run-id>-<五位step>` 创建不可变的独立仓库，准备评测文件、创建仓库、上传并更新可选索引；同一个 run/step 成功后不会重复上传。

## 训练说明文档硬性规范

AI 每次准备执行训练前，都必须在本次运行目录生成并校验说明文档：

```bash
MANIFEST_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/experiment_manifest.py"
python "$MANIFEST_TOOL" init --output-dir "/absolute/path/to/run-output"
# AI 填写 run_manifest.json 的必填字段
python "$MANIFEST_TOOL" validate --output-dir "/absolute/path/to/run-output"
python "$MANIFEST_TOOL" render --output-dir "/absolute/path/to/run-output"
```

必填字段为 `schema_version`、`run_id`、`title`、`purpose`、`hypothesis`、`changes`、`comparison_run`、`dataset`、`model`、`config_file`、`expected_result`、`notes` 和 `created_at`。`config_file` 只能是运行目录内的文件名，不能写另一台机器的绝对路径。校验失败时，AI 不得启动训练命令；训练参数或基线改变时必须重新更新并渲染。机器可读文件是 `run_manifest.json`，可读文件是同目录的 `training_task.md`。

评测页标题区的“绑定训练任务”会读取训练 Monitor 发现的运行目录，并在概览中展示可用的说明。绑定不再要求 `run_manifest.json` 或 `training_task.md` 存在且通过校验；下拉框显示“运行日期/ID · 训练任务名称”，没有说明的任务仍可绑定，概览中只提示暂无训练说明。这样评测记录能明确对应哪一次训练，而不会把模型名当作绑定关系。

## 配置 StreamLake

打开评测页设置，从已登录的 StreamLake 浏览器 Network 面板复制完整请求头。优先选择 Referer 包含 `/wanqing/proj-.../` 的请求；Monitor 会解析 Cookie 和 project ID，且不会回显 Cookie。

CLI 默认将它们保存到 `<target-dir>/config/cookie` 和 `<target-dir>/config/project_id`。project ID 也可通过 `STREAMLAKE_PROJECT_ID` 或 `--project-id` 提供。凭据禁止进入 Git、命令历史、README 或截图。

## 启动和访问网页

服务器端：

```bash
python "$MONITOR_TOOL" deploy --target-dir "$TARGET_DIR" --training-output-root "$TRAINING_ROOT"
```

在运行浏览器的当前设备上新开终端并保持 SSH 隧道：

```bash
ssh -N -L 18280:127.0.0.1:18280 USER@SERVER
```

将 `USER@SERVER` 换成真实 SSH 登录信息，需要非默认端口时添加 `-p SSH_PORT`。个人 SSH 别名只有在当前设备已经写入 SSH 配置时才能使用。浏览器打开 `http://127.0.0.1:18280/` 查看训练，打开 `http://127.0.0.1:18280/eval/` 查看评测日志。

训练页显示指标、进度、GPU、run manifest、checkpoint、最近日志，并保留原 checkpoint 上传流程。评测页提供同步、任务分层、Think/No-think 样本、多输出查看、完整日志下载、对比、排行和持久化增量分析。普通同步硬性限制为 `2026-08-01T00:00:00Z`（含）之后的记录；无输出的失败任务会过滤，运行中的任务等出现输出后再下载日志，日志下载不设大小上限。

评测页顶部工具区提供“上传”和“同步”按钮，上传位于同步之前且使用不同颜色。上传文件会先用同一解析器校验，至少要识别出任务、样本或评测元数据；通过后原始文件永久保存到与平台同步相同的 `<output-dir>/logs/<eval-task-id>/evaluation.log`，同时保存分析缓存和 `evaluation_note.md`，左侧显示“自主上传”标签。上传时可以直接选择训练任务，之后也可以在详情页更换绑定；平台同步不会清理这些手工记录。

评测上传不要求用户另写说明文档。系统生成的 `evaluation_note.md` 会记录来源、原文件名、评测 ID、上传时间、SHA-256、解析器版本、识别任务数、展示样本数、完成任务数、解析异常数和可选的训练绑定。训练绑定允许没有 `run_manifest.json` 或 `training_task.md` 的运行目录，但 AI 启动训练前仍必须完成训练说明文档的初始化、校验和渲染。

## CLI 常用命令

```bash
TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py"
python "$TOOL" probe
python "$TOOL" sync
python "$TOOL" status
python "$TOOL" list
python "$TOOL" compare 评测ID1 评测ID2 --baseline 评测ID1 --primary-metric score
python "$TOOL" download-log 评测ID
python "$TOOL" context
```

只有在确实需要把日志放到其他位置时，才使用 `sync --log-dir 绝对路径`。`error_count` 或 `log_downloads.error_count` 非零时只能报告部分覆盖。

## 测试

```bash
python tests/test_streamlake_experiments.py
python tests/test_eval_monitor.py
python -m py_compile scripts/streamlake_experiments.py scripts/eval_monitor.py scripts/experiment_manifest.py scripts/release_check.py
python scripts/release_check.py
```

`release_check.py` 是发布前的便携性和隐私检查器。分享或发布前必须执行；它会检查 Monitor 文件是否齐全、模板是否含有机器路径或任务、是否混入运行时产物，以及是否出现非占位的个人路径、账号地址或凭据。

## 安全边界

StreamLake 客户端只使用记录在案的只读端点，并通过受控流程下载日志；不会把 StreamLake Cookie 发送给日志 CDN。Monitor 只监听回环地址，应通过 SSH 转发访问，不应直接暴露公网。不要提交运行凭据、生成数据、checkpoint、模型文件或个人路径配置。

## License

MIT，详见 [LICENSE](LICENSE)。
