# Team & Workflow

Agentica 提供两种多智能体协作模式：**Team**（动态委派）和 **Workflow**（确定性编排）。

## Team（智能体团队）

Team 模式下，一个主 Agent 充当协调者，根据任务需要动态委派给团队成员。

### 基本用法

```python
import asyncio
from agentica import Agent, OpenAIChat, BaiduSearchTool

# 专业研究员
researcher = Agent(
    name="Researcher",
    model=OpenAIChat(id="gpt-4o"),
    tools=[BaiduSearchTool()],
    description="负责信息搜索和资料整理",
    instructions=["搜索相关信息并整理要点"],
)

# 专业写手
writer = Agent(
    name="Writer",
    model=OpenAIChat(id="gpt-4o"),
    description="负责内容写作和文章润色",
    instructions=["基于研究材料撰写高质量文章"],
)

# 团队协调者
team = Agent(
    name="Editor",
    team=[researcher, writer],
    instructions=[
        "你是一个编辑，协调研究员和写手完成任务",
        "研究任务交给 Researcher",
        "写作任务交给 Writer",
    ],
)

async def main():
    await team.print_response_stream("写一篇关于 AI Agent 最新发展的文章")

asyncio.run(main())
```

### 工作原理

1. 团队成员通过 `as_tool()` 自动转换为工具注册到主 Agent
2. 主 Agent 根据任务描述决定调用哪个成员
3. 成员 Agent 独立执行任务并返回结果
4. 主 Agent 整合结果，生成最终响应

### Agent 作为工具

也可以手动将 Agent 转换为工具：

```python
# 将 Agent 转换为工具
research_tool = researcher.as_tool(
    tool_name="research",
    tool_description="搜索并整理指定主题的资料",
)

# 在其他 Agent 中使用
main_agent = Agent(
    tools=[research_tool],
)
```

### 设计原则

| 原则 | 说明 |
|------|------|
| **单一职责** | 每个成员专注一个领域 |
| **清晰描述** | `description` 帮助协调者正确委派 |
| **适量成员** | 3-5 个成员最佳，过多会降低委派准确性 |
| **适量工具** | 每个成员 3-7 个工具 |

## Workflow（工作流）

Workflow 模式下，步骤顺序是确定性的，适合需要严格控制流程的场景。

### 何时选择 Workflow

| 场景 | Team | Workflow |
|------|:----:|:--------:|
| 步骤顺序固定 | | **适合** |
| 需要非 LLM 步骤（数据验证、计算） | | **适合** |
| 不同步骤需要不同模型 | | **适合** |
| 任务分解不确定 | **适合** | |
| 需要动态判断下一步 | **适合** | |

### 基本用法

继承 `Workflow` 类，实现 `run()` 方法：

```python
import asyncio
from typing import List
from pydantic import BaseModel, Field
from agentica import Agent, OpenAIChat, Workflow, RunResponse


class AnalysisReport(BaseModel):
    summary: str
    key_findings: List[str]
    recommendations: List[str]


class ResearchWorkflow(Workflow):
    """研究工作流：搜索 -> 分析 -> 报告"""

    description: str = "搜索、分析并生成研究报告"

    # 不同步骤用不同模型优化成本
    extractor: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),  # 低成本模型做提取
        name="Extractor",
        instructions=["提取文本中的关键信息"],
    )

    analyst: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),  # 高质量模型做分析
        name="Analyst",
        instructions=["深入分析数据，给出洞见和建议"],
        response_model=AnalysisReport,
    )

    def _validate_data(self, data: str) -> str:
        """纯 Python 验证步骤，无需 LLM"""
        lines = [l.strip() for l in data.split("\n") if l.strip()]
        return "\n".join(lines)

    async def run(self, topic: str) -> RunResponse:
        # Step 1: LLM 提取
        extracted = await self.extractor.run(f"提取关键信息：{topic}")

        # Step 2: Python 验证（无 LLM 成本）
        cleaned = self._validate_data(extracted.content)

        # Step 3: LLM 分析
        result = await self.analyst.run(f"分析以下内容：\n{cleaned}")

        return RunResponse(content=result.content)


async def main():
    wf = ResearchWorkflow()
    result = await wf.run("2024年全球AI芯片市场分析")
    print(result.content)

asyncio.run(main())
```

### Workflow 特点

- `run()` 方法是 **async** 的，也提供 `run_sync()` 同步适配器
- 可以混合 LLM 步骤和纯 Python 步骤
- 不同步骤可以使用不同模型（成本优化）
- 步骤间可以传递结构化数据（Pydantic 模型）
- 支持会话持久化

### 同步运行

```python
wf = ResearchWorkflow()
result = wf.run_sync("2024年全球AI芯片市场分析")
print(result.content)
```

## 实际示例

### 新闻报道流水线

```python
class NewsWorkflow(Workflow):
    researcher: Agent = Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[BaiduSearchTool()],
        instructions=["搜索最新新闻"],
    )

    writer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=["写一篇新闻报道，包含标题、导语、正文"],
    )

    reviewer: Agent = Agent(
        model=OpenAIChat(id="gpt-4o"),
        instructions=["审核文章的准确性和可读性，提出修改建议"],
    )

    async def run(self, topic: str) -> RunResponse:
        # 1. 搜索
        research = await self.researcher.run(f"搜索最新新闻：{topic}")

        # 2. 撰写
        draft = await self.writer.run(
            f"基于以下资料撰写新闻报道：\n{research.content}"
        )

        # 3. 审核
        final = await self.reviewer.run(
            f"审核并改进以下文章：\n{draft.content}"
        )

        return RunResponse(content=final.content)
```

### 辩论模式

```python
import asyncio
from agentica import Agent, OpenAIChat

pro = Agent(
    name="Pro",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["你支持这个观点，给出论据"],
)

con = Agent(
    name="Con",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["你反对这个观点，给出论据"],
)

judge = Agent(
    name="Judge",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["总结双方观点，给出客观评价"],
)

async def debate(topic: str, rounds: int = 2):
    context = f"辩题：{topic}\n"

    for i in range(rounds):
        pro_arg = await pro.run(f"{context}\n请给出第 {i+1} 轮正方论点")
        context += f"\n正方：{pro_arg.content}"

        con_arg = await con.run(f"{context}\n请给出第 {i+1} 轮反方论点")
        context += f"\n反方：{con_arg.content}"

    verdict = await judge.run(f"{context}\n请总结并评判")
    print(verdict.content)

asyncio.run(debate("远程办公比到办公室办公更高效"))
```

## 下一步

- [Agent 核心概念](agent.md) — 回顾 Agent 基础
- [工具系统](../guides/tools.md) — 为 Agent 配备工具
- [API 参考](../api/agent.md) — 完整 API
