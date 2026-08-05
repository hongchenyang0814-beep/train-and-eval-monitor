# StreamLake 实验分析 Skill

[English](README.md)

这是一个面向 Codex 的 Skill 和零第三方依赖 Python CLI，用于把 StreamLake 万擎平台上的微调与比赛评估实验同步到本地，并自动对比指标和训练参数。它内置可通过 SSH 访问的评测日志 Monitor，可通过受控命令下载和分析完整评测日志，但不会下载 checkpoint、模型权重、数据集或预测文件。

> 告别手动翻实验、抄指标。StreamLake Experiment Analyst 可以一键同步万擎训练与评估记录，在本地自动整理指标、参数与实验关联，并生成适合 Codex 等 Vibe Coding 工具直接分析的上下文。只拉取轻量实验元数据，不下载模型权重、checkpoint、数据集或预测文件。

## 功能

- 全量分页同步训练与评估实验，并按实验 ID 去重。
- 强制只同步创建时间在 `2026-08-01T00:00:00Z`（含）之后的实验；`sync --since ISO-8601时间戳` 只能进一步缩小范围，不能请求更早记录。
- 对平台明确标记为 `FAILED` 且 `hasOutput=false` 的评测任务，在请求详情和结果前自动过滤。
- 可按精确评测 ID 下载完整日志，不限制日志文件大小，且不会把 StreamLake Cookie 转发给日志 CDN。
- 每次普通 `sync` 都会自动补齐所有已成功评测的完整日志；默认保存到 `<output-dir>/logs/<evalTaskId>/evaluation.log`，已有完整文件会复用并重新计算哈希。
- 正在运行或排队的评测会保留元数据，但在 `hasOutput=true` 前不会请求日志；后续同步会在评测成功后自动补下。
- 保存 SQLite、脱敏 JSON、CSV、实验目录和 LLM 上下文。
- 比较指标差值、缺失指标和训练参数差异。
- 脱敏 Cookie/JWT/签名 URL，排除二进制与大文本内容。
- 只允许 StreamLake 官方域名上的只读查询端点。
- 一条命令部署本地评测日志 Monitor，支持平台配置、同步、11 个子任务、Think/No-think 样本、多输出切换、日志分页和评测对比。
- 每次同步成功后只在后台分析新增或变化的日志，完整解析结果压缩保存到 `<output-dir>/analysis_cache`；页面打开不触发批量分析，也不显示缓存进度，后续页面或服务重启直接复用有效缓存。

## 安装

需要本地已安装 Git、Python 3.10 或更高版本，以及 Codex。

### 首次安装

```bash
mkdir -p ~/.codex/skills
git clone https://github.com/hongchenyang0814-beep/streamlake-experiment-analyst.git \
  ~/.codex/skills/streamlake-experiment-analyst
```

### 验证安装

下面的检查只读取本地文件，不会访问 StreamLake：

```bash
test -f ~/.codex/skills/streamlake-experiment-analyst/SKILL.md
python ~/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py --help
```

重启 Codex 或新建任务，让 Codex 重新发现 Skill。之后可以直接让 Codex 使用 `streamlake-experiment-analyst` 同步或对比实验。

### 更新已安装的 Skill

```bash
git -C ~/.codex/skills/streamlake-experiment-analyst pull --ff-only
```

更新后请重启 Codex 或新建任务。身份认证和项目 ID 属于安装后的配置；不要在克隆或更新命令中放入 Cookie。

## 配置

推荐先运行 Monitor 的 `deploy`，再通过网页设置粘贴完整请求头。Cookie 和 project ID 会自动保存到统一运行目录，不要提交 Cookie，也不要把它放进命令行参数。

```bash
mkdir -p ~/.local/share/streamlake-eval-monitor/config
chmod 700 ~/.local/share/streamlake-eval-monitor/config
# 使用你信任的本地编辑器把 Cookie 写入下面的文件
chmod 600 ~/.local/share/streamlake-eval-monitor/config/cookie
```

默认路径是 `~/.local/share/streamlake-eval-monitor/config/cookie`，也可以通过 `STREAMLAKE_COOKIE_FILE` 修改。然后设置项目 ID：

```bash
export STREAMLAKE_PROJECT_ID="proj-your-project-id"
```

也可以将项目 ID 单独保存到 `~/.local/share/streamlake-eval-monitor/config/project_id`，或给 `probe` / `sync` 传入 `--project-id`。优先级依次为命令行参数、环境变量和该配置文件。

## 使用

```bash
TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/streamlake_experiments.py"

python "$TOOL" probe
python "$TOOL" sync
python "$TOOL" download-log 评测ID
# 可选：进一步缩小时间范围
python "$TOOL" sync --since 2026-08-15T00:00:00Z
python "$TOOL" status
python "$TOOL" list
python "$TOOL" compare 实验ID1 实验ID2 \
  --baseline 实验ID1 --primary-metric score
python "$TOOL" context
```

### 部署评测日志 Monitor

```bash
MONITOR_TOOL="$HOME/.codex/skills/streamlake-experiment-analyst/scripts/eval_monitor.py"

# 部署或升级，并在 127.0.0.1:18280 后台启动
python "$MONITOR_TOOL" deploy

# 查看、停止、重新启动
python "$MONITOR_TOOL" status
python "$MONITOR_TOOL" stop
python "$MONITOR_TOOL" start
```

首次打开网页后，按设置弹窗说明从 StreamLake 开发者工具复制完整请求头并粘贴保存。服务自动提取 Cookie 和 project ID，不会在页面回显 Cookie。

先在服务器执行上面的 `deploy`。然后在运行浏览器的当前设备上新开终端并保持 SSH 隧道运行，不要在远端 SSH 会话中执行：

```bash
ssh -N -L 18280:127.0.0.1:18280 USER@SERVER
```

将 `USER@SERVER` 替换为真实的 `用户名@服务器地址`；非默认 SSH 端口需添加 `-p SSH端口`。个人 SSH 别名只有在当前设备已经配置时才能使用，skill 不会内置或假定任何个人别名。随后在当前设备打开 `http://127.0.0.1:18280/`。部署到自定义路径或端口可运行 `python "$MONITOR_TOOL" deploy --help`。

可用 `sync --log-dir 路径` 修改自动日志目录。每次同步的下载数、复用数和错误会写入 `sync_state.json` 的 `log_downloads` 字段；只有 `error_count=0` 且 `log_downloads.error_count=0` 时，元数据与日志覆盖才完整。

本地输出包括：

- `experiments.sqlite`：实验、参数、指标、关联和同步错误。
- `raw/`：脱敏后的轻量原始证据。
- `exports/`：CSV 表格。
- `catalog.md`：实验索引。
- `context.md`：供 Codex 或其他 Vibe Coding 工具讨论的上下文。
- `sync_state.json`：更新时间、覆盖数量和 `error_count`。
- `analysis_cache/`：Monitor 生成的压缩日志解析结果，供后续页面和服务重启复用。

`error_count` 不为 0 表示部分接口未成功，不能声称完整覆盖。缺失指标不会被当成 0。

## 安全边界

客户端只允许访问 `https://console.streamlake.com` 上已记录的列表、详情、训练指标和评估输出查询接口。`download-log` 可继续访问评测输出返回的 `safetyimg.com` HTTPS 日志地址，且不会转发 StreamLake Cookie；其他域名、修改类接口、模型下载和推理结果下载仍会被拒绝。

## 测试

```bash
python tests/test_streamlake_experiments.py
python tests/test_eval_monitor.py
python -m py_compile scripts/streamlake_experiments.py
```

## 注意事项

- API 来自万擎前端的实际调用，平台升级后可能需要更新。
- 只有评估任务、数据版本、split、候选集、指标 cutoff 和推理设置一致时，才能严谨比较。
- 参数与结果之间默认只能描述为相关，除非有受控重复实验支持因果结论。

## License

MIT，详见 [LICENSE](LICENSE)。
