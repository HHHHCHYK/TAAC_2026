---
icon: lucide/wrench
---

# 仓库缓存清理

## 目标

在频繁运行 pytest、训练 CLI 或导入实验包之后，仓库里会积累大量 `__pycache__` 目录。
本指南说明如何用统一命令批量清理这些缓存目录，同时避免默认误扫常见的虚拟环境和第三方依赖目录。

## 命令入口

脚本命令：`uv run taac-clean-pycache`

默认行为：

1. 从仓库根目录开始扫描。
2. 删除所有命中的 `__pycache__` 目录。
3. 默认跳过 `.venv`、`venv`、`env`、`.tox`、`node_modules`、`.git` 以及常见工具缓存目录。
4. 输出本次扫描与处理摘要，便于确认清理范围。

## 常用命令

```bash
# 1) 直接清理整个仓库下的 __pycache__
uv run taac-clean-pycache

# 2) 先预览，不执行删除
uv run taac-clean-pycache --dry-run

# 3) 只扫描某个子目录
uv run taac-clean-pycache --root tests

# 4) 连虚拟环境等目录里的 __pycache__ 也一起扫描
uv run taac-clean-pycache --include-env-dirs
```

## 脚本参数

| 参数                 | 类型   | 默认值     | 说明                                                               |
| -------------------- | ------ | ---------- | ------------------------------------------------------------------ |
| `--root`             | string | 仓库根目录 | 指定扫描起点；可传相对路径或绝对路径                               |
| `--dry-run`          | flag   | `false`    | 只输出将被处理的目录，不执行删除                                   |
| `--include-env-dirs` | flag   | `false`    | 把 `.venv`、`venv`、`env`、`.tox`、`node_modules` 等目录也纳入扫描 |

参数校验要点：

1. `--root` 指向的路径必须存在。
2. `--root` 必须是目录，不能是单个文件。
3. 未传 `--include-env-dirs` 时，工具会主动跳过常见环境目录，减少误删和无效扫描。

## 输出解释

执行时，每个命中的目录都会先输出一行处理结果：

- 真实删除时前缀为 `[removed]`
- dry-run 预览时前缀为 `[dry-run]`
- 删除失败时前缀为 `[failed]`

随后会输出摘要：

```text
root=/desay120T/ct/dev/uid01954/TAAC_2026
mode=delete
matched_dirs=3
processed_dirs=3
matched_files=27
matched_size_mib=0.1432
include_env_dirs=false
failures=0
```

字段含义：

1. `root`：本次扫描的根目录。
2. `mode`：`dry-run` 表示只预览；`delete` 表示真实删除。
3. `matched_dirs`：找到的 `__pycache__` 目录数量。
4. `processed_dirs`：已处理的目录数量；dry-run 下表示“已预览”的数量。
5. `matched_files`：命中目录内的缓存文件总数。
6. `matched_size_mib`：命中缓存文件总大小，单位 MiB。
7. `include_env_dirs`：是否把常见环境目录纳入扫描。
8. `failures`：处理失败的目录数量。

## 推荐使用时机

1. 切换分支后，想去掉旧 bytecode 缓存。
2. 调整模块路径、实验包导入或 pytest 收集逻辑后，想排除历史缓存干扰。
3. 提交前想快速清理工作区里自动生成的 Python 缓存目录。

## 返回码约定

1. `0`：执行成功；包括 dry-run 和真实删除全部成功。
2. `1`：执行完成，但至少有一个目录删除失败。
3. `2`：参数非法，例如 `--root` 不存在，或 `--root` 不是目录。