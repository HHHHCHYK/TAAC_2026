# 仓库日志管理

## 目标

当 GitHub Actions 运行次数和 GitHub Pages 部署次数持续增长时，仓库日志会快速累积，影响排查效率。
本指南提供统一脚本与 API 说明，用于批量清理历史日志，仅保留近期记录。

## 覆盖范围

脚本命令：`uv run taac-clean-github-logs`

清理对象：

1. Actions 工作流运行日志（Actions 页面）
2. GitHub Pages 部署记录（Deployments 页面中 environment=github-pages）

## 使用前准备

1. 设置具备仓库写权限的 Token（建议最小权限覆盖 actions 与 deployments）。
2. 在仓库根目录执行命令。
3. 先 dry-run 预览，再执行真实删除。

PowerShell 示例：

```bash
$env:GITHUB_TOKEN = "ghp_xxx"
uv run taac-clean-github-logs --repo Puiching-Memory/TAAC_2026 --only-completed-runs
```

## 脚本参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--repo` | string | 读取 `GITHUB_REPOSITORY` | 目标仓库，格式 `owner/repo` |
| `--token` | string | 读取 `GITHUB_TOKEN` | GitHub API 访问令牌 |
| `--api-base` | string | `https://api.github.com` | API 基地址 |
| `--keep-action-runs` | int | `30` | 保留最新 N 条 workflow run；更旧的 run 删除其 logs |
| `--keep-pages-deployments` | int | `20` | 保留最新 N 条 Pages deployment；更旧的 deployment 删除 |
| `--per-page` | int | `100` | 分页大小，范围 `1..100` |
| `--only-completed-runs` | flag | `false` | 仅处理 `status=completed` 的 workflow run |
| `--actions-only` | flag | `false` | 只清理 Actions 日志 |
| `--pages-only` | flag | `false` | 只清理 Pages 部署记录 |
| `--execute` | flag | `false` | 开启真实删除；未传入时为 dry-run |

参数校验要点：

1. `--repo` 必须包含 `/`。
2. `--token` 不能为空。
3. `--actions-only` 与 `--pages-only` 互斥。
4. `--keep-*` 必须大于等于 0。

## 推荐执行流程

1. Dry-run 预览：确认将被删除的 run/deployment 数量。
2. 缩小范围：必要时使用 `--actions-only` 或 `--pages-only`。
3. 正式执行：加 `--execute`。
4. 复查：在 Actions 与 Deployments 页面确认历史记录已清理。

## 常用命令

```bash
# 1) 预览（默认 dry-run）
uv run taac-clean-github-logs --repo Puiching-Memory/TAAC_2026 --only-completed-runs

# 2) 正式执行
uv run taac-clean-github-logs --repo Puiching-Memory/TAAC_2026 --only-completed-runs --execute

# 3) 只清理 Actions
uv run taac-clean-github-logs --repo Puiching-Memory/TAAC_2026 --actions-only --execute

# 4) 只清理 Pages
uv run taac-clean-github-logs --repo Puiching-Memory/TAAC_2026 --pages-only --execute
```

## 输出解释

脚本结束时会输出 4 行摘要：

```text
repo=ChinG-Lynn/TAAC_2026
mode=dry-run
actions=listed:5,targeted:0,deleted:0,failed:0
pages=listed:3,targeted:0,deleted:0,failed:0
```

字段含义：

1. `repo`：本次操作的仓库。
2. `mode`：`dry-run` 表示仅预览；`execute` 表示真实删除。
3. `actions.listed`：拉取到的 workflow run 总数。
4. `actions.targeted`：将被处理（删除日志）的 run 数量。
5. `actions.deleted`：实际已处理数量。在 dry-run 下表示“模拟会删除”的数量。
6. `actions.failed`：处理失败数量。
7. `pages.listed`：拉取到的 GitHub Pages deployment 总数。
8. `pages.targeted`：将被处理（删除）的 deployment 数量。
9. `pages.deleted`：实际已处理数量。在 dry-run 下表示“模拟会删除”的数量。
10. `pages.failed`：处理失败数量。

为什么会出现 `targeted:0`：

1. 默认保留策略是 `--keep-action-runs 30`、`--keep-pages-deployments 20`。
2. 你的仓库当前只有 5 条 run 和 3 条 deployment，均小于保留阈值。
3. 所以没有旧记录进入删除目标，属于正常输出。

如果你想验证删除链路：

```bash
# 先 dry-run：保留 1 条，其余都作为删除目标
uv run taac-clean-github-logs --repo ChinG-Lynn/TAAC_2026 --keep-action-runs 1 --keep-pages-deployments 1

# 再真实执行
uv run taac-clean-github-logs --repo ChinG-Lynn/TAAC_2026 --keep-action-runs 1 --keep-pages-deployments 1 --execute
```

## 返回码约定

返回码是命令执行结束后由进程返回的状态值，用于让你或 CI 判断是否成功。

怎么看返回码：

1. PowerShell：运行命令后执行 `$LASTEXITCODE`。
2. Bash/Zsh：运行命令后执行 `echo $?`。

示例（PowerShell）：

```bash
uv run taac-clean-github-logs --repo ChinG-Lynn/TAAC_2026 --only-completed-runs
$LASTEXITCODE
```

示例（Bash/Zsh）：

```bash
uv run taac-clean-github-logs --repo ChinG-Lynn/TAAC_2026 --only-completed-runs
echo $?
```

状态值含义：

1. `0`：执行成功（包括 dry-run 或 execute 全部成功）。
2. `1`：执行完成但存在失败项（例如部分日志/部署删除失败）。
3. `2`：参数校验失败（例如缺 token、repo 格式非法、参数冲突）。

在 CI 中常见判断方式：

1. 返回码 `0` 继续后续步骤。
2. 返回码非 `0` 直接失败并告警。

## GitHub API 接口清单

脚本内部调用以下接口。

### 1) 列出 workflow runs

- 方法：`GET`
- 路径：`/repos/{owner}/{repo}/actions/runs`
- 关键查询参数：
  - `per_page`：分页大小
  - `page`：页码

示例：

```bash
GET https://api.github.com/repos/Puiching-Memory/TAAC_2026/actions/runs?per_page=100&page=1
```

### 2) 删除 workflow run 日志

- 方法：`DELETE`
- 路径：`/repos/{owner}/{repo}/actions/runs/{run_id}/logs`

示例：

```bash
DELETE https://api.github.com/repos/Puiching-Memory/TAAC_2026/actions/runs/123456789/logs
```

### 3) 列出 Pages deployments

- 方法：`GET`
- 路径：`/repos/{owner}/{repo}/deployments`
- 关键查询参数：
  - `environment=github-pages`
  - `per_page`
  - `page`

示例：

```bash
GET https://api.github.com/repos/Puiching-Memory/TAAC_2026/deployments?environment=github-pages&per_page=100&page=1
```

### 4) 将 deployment 标记为 inactive

删除 deployment 前，需要先将其状态设置为 inactive。

- 方法：`POST`
- 路径：`/repos/{owner}/{repo}/deployments/{deployment_id}/statuses`
- 请求体：

```json
{
  "state": "inactive",
  "auto_inactive": false,
  "description": "taac2026 cleanup script marks deployment inactive before deletion"
}
```

### 5) 删除 deployment

- 方法：`DELETE`
- 路径：`/repos/{owner}/{repo}/deployments/{deployment_id}`
