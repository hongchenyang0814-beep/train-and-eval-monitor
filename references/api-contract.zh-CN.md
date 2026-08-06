# StreamLake 万擎只读 API 契约

本契约基于 2026-07-18 生产环境万擎前端资源中的实现整理。实时 `probe` 命令是验证认证状态和响应结构的最终依据。

## 传输

- 源站：`https://console.streamlake.com`
- 前端使用的 Axios 基础路径：`/api/console/open-api`
- CLI 的有效源站：`https://console.streamlake.com/api/console/open-api`
- 必需请求头：`Cookie`、`Accept: application/json`、POST 请求的 `Content-Type: application/json`、`X-Requested-With: XMLHttpRequest`、`open-api-product: WANQING`、`Accept-Language: zh-CN`
- 项目 ID：传入 `--project-id proj-your-project-id`，或设置 `STREAMLAKE_PROJECT_ID`。

下列 POST 端点仅用于列表或指标查询，在语义上是只读的。本技能不允许使用其他 POST 端点。

## 微调实验

### 列表

- 方法/路径：`POST /api/customized/commercial/v1/train-task/list`
- 请求体分页字段：`page`、`pageSize`
- 请求体项目字段：`projectId`
- 稳定排序：`sortBy: createTime`、`sortOrder: DESC`
- 记录：`responseData.list`
- 总数：`responseData.total`
- ID：`taskId`

默认请求体还会为 `keyword`、`tags`、`taskStatus`、`creator`、`taskType`、`fineTuningType` 和 `baseModelName` 传入空过滤条件。

### 详情与指标

- 详情：`GET /api/customized/commercial/v1/train-task/{taskId}?projectId={projectId}`
- 仪表盘定义：`GET /api/customized/commercial/v1/train-task/analysis/dashboard?taskId={taskId}&fineTuningType={fineTuningType}`
- 指标序列：`POST /api/customized/commercial/v1/train-task/metric-query`

指标查询请求体：

```json
{
  "projectId": "project-id",
  "taskIds": ["task-id"],
  "metrics": [{"name": "metric-name", "seriesNameFormat": "display-name"}]
}
```

仪表盘指标定义会从 `metrics` 数组中递归收集。系统会保留完整的脱敏仪表盘和序列响应；每个返回序列的最新数值点会被标准化，以便进行比较。

## 竞赛评测实验

### 列表

- 方法/路径：`POST /api/customized/commercial/v1/competition-eval-task/list`
- 请求体分页字段：`page`、`pageSize`
- 请求体项目字段：`projectId`
- 稳定排序：`sortBy: createTime`、`sortOrder: DESC`
- 记录：`responseData.items`
- 总数：`responseData.total`
- ID：`evalTaskId`

### 详情与输出

- 详情：`GET /api/customized/commercial/v1/competition-eval-task/{evalTaskId}`
- 评测输出：`GET /api/customized/commercial/v1/competition-eval-task/{evalTaskId}/output`

当任务创建时间不早于 `2026-08-01T00:00:00Z`、状态为 `SUCCEEDED` 且 `hasOutput=true` 时，`download-log` 可以跟随该接口返回的 `safetyimg.com` 或 `yximgs.com` CDN HTTPS `.log` 地址下载完整日志。下载不设置文件大小上限，也不会向日志域名发送 StreamLake Cookie。

列表级的 `r1`、`r2`、`r3` 等数值，以及详情/输出载荷中的数值评分字段，都会被标准化为指标。完整的轻量级载荷会保留在脱敏原始 JSON 中。

## 禁止的端点类别

不得调用包含 create、publish、terminate、submit、delete、upload、download、model-package、log 或 inference-result 行为的端点。不得跟随工件 URL 或预签名 URL。CLI 客户端仅接受 GET 和 POST；调用方仍必须只使用上述明确列出的查询端点。

## 漂移处理

当记录路径不再是列表、认证重定向到登录页，或响应不再是 JSON 时，停止处理受影响的数据源。保留最后一份有效的本地仓库，并且只有在重新核验当前前端请求实现后，才能更新本契约。
