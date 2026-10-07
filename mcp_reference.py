"""Bundled teaching MCP server: real stdio transport, curated local knowledge."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("agent-learning-reference")

@mcp.tool()
def knowledge_lookup(topic: str) -> str:
    """Look up a curated local learning note about agent, memory, mcp or compression."""
    notes = {
        "agent": "Agent 将模型决策、工具执行和执行结果反馈组织成循环。模型收到任务、工具定义与历史消息，返回工具名与参数；运行时校验并执行工具，将结果作为 observation 写回上下文。final_answer 是 smolagents 的结束工具。工具失败应记录错误并让模型有机会修正参数。max_steps 控制执行步数，而不保证与模型调用次数完全一致。",
        "memory": "工作记忆是当前任务的消息历史，随着工具调用逐步增长。长期记忆需要独立存储，并通过显式读取和写入参与决策。本教学应用将 key/value 记忆保存在当前浏览器 localStorage，下一次运行时传入服务端；它不是云端数据库，不跨浏览器同步。记忆读取和写入都有事件，便于检查 Agent 究竟保存了什么。避免在浏览器记忆中放入凭证。",
        "mcp": "MCP 是模型上下文协议，用统一接口连接工具与资源。本示例使用 Python MCP SDK，经 stdio 启动内置 agent-learning-reference 服务，完成 initialize 握手，通过 tools/list 发现工具，再通过 tools/call 调用 knowledge_lookup。结果是预先编写的本地学习笔记，不是实时互联网检索。服务端连接固定配置，浏览器不能指定任意服务器或命令。",
        "compression": "上下文压缩将旧消息归纳成较短摘要，保留任务、重要事实、工具结果、失败和未完成步骤。本应用在字符数估算达到阈值时调用模型生成摘要，保留最新完整工具调用组，避免遗留缺少结果的 tool_call。压缩可丢失信息，不等于长期记忆。界面记录压缩前后估算大小、耗时、摘要和实际 token 用量；字符数除以四仅为粗略估算，中文误差尤其明显。",
    }
    key = next((k for k in notes if k in topic.lower()), None)
    if key is None:
        aliases = {"记忆":"memory", "压缩":"compression", "智能体":"agent", "工具":"mcp"}
        key = next((v for k,v in aliases.items() if k in topic), "agent")
    return notes[key]

if __name__ == "__main__":
    mcp.run(transport="stdio")
