# StreamLake Wanqing Read-Only API Contract

Contract observed from the production Wanqing frontend assets on 2026-07-18. The live `probe` command is the final authentication and response-shape check.

## Transport

- Origin: `https://console.streamlake.com`
- Axios base path used by the frontend: `/api/console/open-api`
- Effective CLI origin: `https://console.streamlake.com/api/console/open-api`
- Required headers: `Cookie`, `Accept: application/json`, `Content-Type: application/json` for POST, `X-Requested-With: XMLHttpRequest`, `open-api-product: WANQING`, `Accept-Language: zh-CN`
- Project ID: supply `--project-id proj-your-project-id` or set `STREAMLAKE_PROJECT_ID`.

POST endpoints below are list or metric queries and are semantically read-only. The skill does not permit other POST endpoints.

## Fine-tuning experiments

### List

- Method/path: `POST /api/customized/commercial/v1/train-task/list`
- Body pagination: `page`, `pageSize`
- Body project field: `projectId`
- Stable ordering: `sortBy: createTime`, `sortOrder: DESC`
- Records: `responseData.list`
- Total: `responseData.total`
- ID: `taskId`

The default body also supplies empty filters for `keyword`, `tags`, `taskStatus`, `creator`, `taskType`, `fineTuningType`, and `baseModelName`.

### Detail and metrics

- Detail: `GET /api/customized/commercial/v1/train-task/{taskId}?projectId={projectId}`
- Dashboard definition: `GET /api/customized/commercial/v1/train-task/analysis/dashboard?taskId={taskId}&fineTuningType={fineTuningType}`
- Metric series: `POST /api/customized/commercial/v1/train-task/metric-query`

Metric query body:

```json
{
  "projectId": "project-id",
  "taskIds": ["task-id"],
  "metrics": [{"name": "metric-name", "seriesNameFormat": "display-name"}]
}
```

Dashboard metric definitions are collected recursively from `metrics` arrays. The complete sanitized dashboard and series responses are retained; the latest numeric point from each returned series is normalized for comparisons.

## Competition evaluation experiments

### List

- Method/path: `POST /api/customized/commercial/v1/competition-eval-task/list`
- Body pagination: `page`, `pageSize`
- Body project field: `projectId`
- Stable ordering: `sortBy: createTime`, `sortOrder: DESC`
- Records: `responseData.items`
- Total: `responseData.total`
- ID: `evalTaskId`

### Detail and output

- Detail: `GET /api/customized/commercial/v1/competition-eval-task/{evalTaskId}`
- Evaluation output: `GET /api/customized/commercial/v1/competition-eval-task/{evalTaskId}/output`

When the task was created on or after `2026-08-01T00:00:00Z`, is `SUCCEEDED`, and has `hasOutput=true`, `download-log` may follow the returned HTTPS `.log` URL on the approved `safetyimg.com` or `yximgs.com` CDN families. The complete log is downloaded without a file-size limit and without forwarding the StreamLake Cookie to the log host.

List-level values such as `r1`, `r2`, and `r3`, plus numeric score fields in detail/output payloads, are normalized as metrics. Full lightweight payloads remain in sanitized raw JSON.

## Forbidden endpoint classes

Do not call endpoints containing create, publish, terminate, submit, delete, upload, download, model-package, log, or inference-result behavior. Do not follow artifact or pre-signed URLs. The CLI client accepts only GET and POST; callers must still use only the explicit query endpoints above.

## Drift handling

Stop the affected source when the record path is not a list, authentication redirects to login, or the response is non-JSON. Keep the last valid repository and update this contract only after re-checking the current frontend request implementation.
