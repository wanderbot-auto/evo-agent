# Evo Agent

Agent 开发与运行观测项目，当前基于 smolagents + MiniMax + FastAPI，支持模型、工具、MCP、记忆和上下文压缩轨迹。

本地运行需 Python 3.12+。

```bash
git clone https://github.com/wanderbot-auto/evo-agent.git
cd evo-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 `OPENAI_API_KEY`，并将 `APP_ACCESS_TOKEN` 改为自己的应用口令，然后启动：

```bash
python -m uvicorn index:app --host 127.0.0.1 --port 8081
```

打开 http://127.0.0.1:8081，在配置中输入上述应用口令。Windows 可用 `.venv\Scripts\Activate.ps1` 激活虚拟环境，用 `Copy-Item .env.example .env` 复制配置。
默认模型 `MiniMax-M3`；配置默认折叠。单次最多 220 秒，模型请求失败最多重试一次，不重复执行工具。

- `runtime.py`：smolagents 运行、模型流与工具事件。
- `index.py`：访问保护、输入校验、SSE 与运行时限。
- `ui.html`：任务输入、实时轨迹、筛选和导出。
- `mcp_reference.py`：内置学习笔记 MCP 服务（stdio）。

记忆与历史保存在当前浏览器；上下文大小由字符数估算，真实 token 用量由模型返回。

验证：`pip install pytest httpx`，然后运行 `python -m pytest -q tests`。
Vercel 配置见 `vercel.json`，项目函数默认区域同时设为香港 `hkg1`；密钥只放服务端环境变量。`.env`、`ACCESS.txt`、测试和本地验证文件不部署。
