
#### Agent Components
<img src="https://github.com/shibing624/agentica/blob/main/docs/llm_agentv2.png" width="800" />

- **规划（Planning）**：任务拆解、生成计划、反思
- **记忆（Memory）**：短期记忆（prompt实现）、长期记忆（RAG实现）
- **工具使用（Tool use）**：function call能力，调用外部API，以获取外部信息，包括当前日期、日历、代码执行能力、对专用信息源的访问等

#### Agentica Workflow

**Agentica** can also build multi-agent systems and workflows.

**Agentica** 还可以构建多Agent系统和工作流。

<img src="https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png" width="800" />

- **Planner**：负责让LLM生成一个多步计划来完成复杂任务，生成相互依赖的“链式计划”，定义每一步所依赖的上一步的输出
- **Worker**：接受“链式计划”，循环遍历计划中的每个子任务，并调用工具完成任务，可以自动反思纠错以完成任务
- **Solver**：求解器将所有这些输出整合为最终答案