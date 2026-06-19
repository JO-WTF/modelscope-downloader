# modelscope-downloader

一个很小的 ModelScope 命令行下载器，主要用于官方 SDK 因 SSL 证书校验问题无法下载时的替代方案。

## 安装

```bash
cd /Users/zhaoyu/Projects/modelscope-downloader
python3 -m venv .venv
.venv/bin/pip install -e .
```

运行依赖：

- `requests`：HTTP 下载。
- `rich`：对齐的多任务下载进度界面。

## 单文件下载

```bash
ms download Qwen/Qwen2.5-0.5B README.md --local-dir ./downloads --no-verify-ssl
```

这会请求：

```text
https://www.modelscope.cn/models/Qwen/Qwen2.5-0.5B/resolve/master/README.md
```

## 整仓下载

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/Qwen2.5-0.5B --no-verify-ssl
```

指定并发数：

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/Qwen2.5-0.5B --max-workers 8 --no-verify-ssl
```

预览下载计划，不实际下载：

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/Qwen2.5-0.5B --dry-run --no-verify-ssl
```

如果终端 resize 后仍然显示异常，可以使用非动态进度输出：

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/Qwen2.5-0.5B --simple-progress --no-verify-ssl
```

只下载某类文件：

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/qwen --include "*.safetensors" --include "*.json" --no-verify-ssl
```

排除文件：

```bash
ms download Qwen/Qwen2.5-0.5B --local-dir ./downloads/qwen --exclude "*.bin" --no-verify-ssl
```

## 私有模型

```bash
export MODELSCOPE_TOKEN="your-token"
ms download owner/private-model config.json --no-verify-ssl
```

也可以显式传入：

```bash
ms download owner/private-model config.json --token "your-token" --no-verify-ssl
```

## 说明

- 推荐入口模仿 Hugging Face CLI 的 `hf download`：`ms download <model_id> [filename] --local-dir <dir>`。
- 默认会校验证书；只有传 `--no-verify-ssl` 时才关闭 TLS 证书验证。
- 下载使用 `.part` 临时文件，默认支持断点续传。
- 下载界面使用固定列宽进度表，包含总体进度、单文件进度、已下载大小、下载速度和预计剩余时间。
- 下载过程中 resize 终端窗口时，会自动清屏并重绘进度界面，减少旧宽度残影导致的错乱。
- 整仓下载启动时会扫描目标目录：已完整下载的文件会直接计入完成数，已有 `.part` 文件会计入总体已下载字节，并从断点继续下载。
- 启动时会自动扫描官方 ModelScope cache 目录，默认是 `~/.cache/modelscope`。如果发现匹配的半成品，会复制到 `--local-dir` 对应路径的 `.part` 后继续下载。
- 下载开始前会计算剩余下载量并检查目标磁盘空间；空间不足会直接停止。
- 每个文件下载结束前会做大小完整性校验；如果实际字节数和 HTTP 声明的总大小不一致，会触发重试。
- 每个文件默认重试 5 次。某个文件重试后仍失败时会跳过它，继续下载其他文件，最后汇总失败列表。
- `TOTAL` 行会显示已完成文件数和剩余文件数，例如 `TOTAL files 12/80 left 68`。
- 整仓下载默认 `--max-workers 4`，可以按网络和磁盘情况调大或调小。
- 单文件下载使用 ModelScope 的 `resolve/{revision}/{path}` URL。
- 整仓下载需要先调用 ModelScope 文件列表 API。这个项目内置了几个常见 API 形态候选，如果 ModelScope 后续调整接口，可能需要更新 `msdl/client.py` 里的 `_list_file_candidates`。

关闭 SSL 校验会降低连接安全性，建议只在你确认网络环境和目标站点可信时使用。
