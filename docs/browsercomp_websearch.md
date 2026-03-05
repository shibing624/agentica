# BrowseComp-Plus WebSearch 升级方案

## 一、目标

在 BrowseComp-Plus 榜单上取得高分。该榜单考核多跳深度搜索、跨源信息整合、干扰信息甄别、精确答案提取等能力。

## 二、当前架构分析

### 现有能力（DeepAgent）

| 模块 | 现状 | 评价 |
|------|------|------|
| 工具调用循环 | Model 层递归 tool_call，DeepAgent 通过 hooks 注入控制 | 成熟 |
| 并行工具执行 | `asyncio.TaskGroup + Semaphore` | 成熟 |
| 上下文管理 | 双阈值（软限压缩/硬限终止） | 可用 |
| 重复检测 | 连续相同工具调用检测 | 可用 |
| 反思 | 每 N 步注入反思 prompt | 基础 |
| 子 Agent | explore/research/code 三类，并行执行，隔离上下文 | 可用 |
| 搜索工具 | BaiduSearch，支持批量 query | 基础 |
| URL 抓取 | UrlCrawlerTool，带缓存 | 可用 |

### 核心短板

1. **搜索是"LLM 自由驱动"的**：没有程序化的搜索编排，全靠 LLM 自己决定搜几次、搜什么
2. **无 Query 分解/改写**：复杂多跳问题直接丢给搜索，召回率低
3. **无证据验证闭环**：找到候选答案后不做交叉验证，容易被 Hard Negatives 误导
4. **无实体/证据状态追踪**：搜索结果当纯文本处理，信息碎片化
5. **子 Agent 间无信息共享**：多个 research 子 Agent 各自为战

## 三、升级架构设计

### 整体架构

```
用户问题
    │
    ▼
┌──────────────────────────────────────────────┐
│  WebSearchAgent (继承 DeepAgent)              │
│                                              │
│  ┌─────────────────────────────────┐         │
│  │ SearchOrchestrator (搜索编排器)   │         │
│  │                                 │         │
│  │  1. QueryDecomposer             │         │
│  │     - 问题分解为子查询           │         │
│  │     - 子查询改写/多角度表述       │         │
│  │                                 │         │
│  │  2. EvidenceStore               │         │
│  │     - 结构化证据收集             │         │
│  │     - 实体关系追踪               │         │
│  │     - 信息充分性判断             │         │
│  │                                 │         │
│  │  3. AnswerVerifier              │         │
│  │     - 候选答案验证               │         │
│  │     - 多源交叉验证               │         │
│  │     - 置信度评估                 │         │
│  └─────────────────────────────────┘         │
│                                              │
│  工具层复用 DeepAgent:                        │
│    web_search / fetch_url / task(research)    │
└──────────────────────────────────────────────┘
```

### 关键设计原则

1. **独立 py 文件**，不修改 `deep_agent.py`，继承 `DeepAgent`
2. **新增组件可复用**：`QueryDecomposer`、`EvidenceStore`、`AnswerVerifier` 独立于 `WebSearchAgent`，后续可给 `DeepAgent` / `Agent` 使用
3. **不搞多搜索引擎 fallback**，太重，投入产出比不高
4. **充分利用现有的 hooks 机制**和子 Agent 并行能力

## 四、新增文件结构

```
agentica/
├── web_search_agent.py          # WebSearchAgent 主类
├── search/                      # 搜索增强模块（新增目录）
│   ├── __init__.py
│   ├── orchestrator.py          # SearchOrchestrator 搜索编排器
│   ├── query_decomposer.py      # QueryDecomposer 查询分解/改写
│   ├── evidence_store.py        # EvidenceStore 证据收集与追踪
│   └── answer_verifier.py       # AnswerVerifier 答案验证
├── prompts/
│   └── base/
│       └── md/
│           ├── query_decompose.md       # 查询分解 prompt
│           ├── evidence_extract.md      # 证据抽取 prompt
│           ├── answer_verify.md         # 答案验证 prompt
│           └── search_reflection.md     # 搜索反思 prompt
```

## 五、核心组件详细设计

### 5.1 WebSearchAgent (`agentica/web_search_agent.py`)

继承 `DeepAgent`，在其基础上：
- 注入 `SearchOrchestrator` 管理搜索全流程
- 重写 system prompt，注入搜索策略指导
- 在 `_post_tool_hook` 中集成证据收集和搜索状态更新
- 新增 `search_and_answer` 工具，作为顶层搜索入口

```python
@dataclass(init=False)
class WebSearchAgent(DeepAgent):
    """
    WebSearchAgent - 面向深度搜索的增强 Agent

    在 DeepAgent 基础上增加：
    1. 搜索编排器：程序化控制搜索流程
    2. Query 分解/改写：提升召回率
    3. 证据收集与追踪：结构化管理搜索发现
    4. 答案验证闭环：交叉验证 + 置信度评估
    """

    # 搜索编排器配置
    max_search_rounds: int = 15         # 最大搜索轮次
    max_queries_per_round: int = 5      # 每轮最大查询数
    min_evidence_count: int = 2         # 最少需要多少条独立证据
    confidence_threshold: float = 0.8   # 答案置信度阈值
    enable_query_decomposition: bool = True
    enable_answer_verification: bool = True
    enable_evidence_tracking: bool = True

    # 内部组件
    _orchestrator: SearchOrchestrator   # 搜索编排器

    def __init__(self, *, max_search_rounds=15, ...  **kwargs):
        # 存储配置
        # 强制启用 web_search + fetch_url
        kwargs['include_web_search'] = True
        kwargs['include_fetch_url'] = True

        super().__init__(**kwargs)

        # 初始化搜索编排器
        self._orchestrator = SearchOrchestrator(
            model=self.model,
            max_rounds=max_search_rounds,
            max_queries_per_round=max_queries_per_round,
            min_evidence_count=min_evidence_count,
            confidence_threshold=confidence_threshold,
        )

        # 注册增强工具
        self._register_search_tools()

        # 增强 hooks
        self._enhance_hooks()
```

**核心方法**：

```python
async def deep_search(self, question: str) -> str:
    """
    对外的主搜索方法，编排完整搜索流程：
    1. 分解问题为子查询
    2. 多轮搜索循环（搜索 -> 抽取证据 -> 判断充分性）
    3. 验证答案
    4. 生成最终回答
    """
    # Step 1: 分解问题
    sub_queries = await self._orchestrator.decompose_query(question)

    # Step 2: 多轮搜索
    for round_idx in range(self.max_search_rounds):
        # 获取待搜索的查询（编排器决定）
        queries = self._orchestrator.get_next_queries()
        if not queries:
            break  # 信息已充分

        # 并行搜索
        results = await self._parallel_search(queries)

        # 抽取证据，更新 EvidenceStore
        await self._orchestrator.process_results(results, question)

        # 判断信息充分性
        if self._orchestrator.is_sufficient():
            break

        # 生成新查询（基于已有证据的信息缺口）
        new_queries = await self._orchestrator.generate_followup_queries(question)
        if not new_queries:
            break

    # Step 3: 生成候选答案
    answer = await self._orchestrator.synthesize_answer(question)

    # Step 4: 验证答案
    if self.enable_answer_verification:
        verified = await self._orchestrator.verify_answer(question, answer)
        if not verified.is_confident:
            # 补充搜索
            ...

    return answer
```

### 5.2 SearchOrchestrator (`agentica/search/orchestrator.py`)

搜索编排器，管理搜索全流程的状态和决策。

```python
@dataclass
class SearchOrchestrator:
    """
    搜索编排器 - 管理搜索状态和流程决策

    职责：
    1. 调用 QueryDecomposer 分解/改写查询
    2. 管理搜索轮次和查询队列
    3. 调用 EvidenceStore 存储和追踪证据
    4. 判断信息充分性
    5. 调用 AnswerVerifier 验证答案
    """
    model: Any  # LLM model，用于推理
    query_decomposer: QueryDecomposer
    evidence_store: EvidenceStore
    answer_verifier: AnswerVerifier

    max_rounds: int = 15
    max_queries_per_round: int = 5
    min_evidence_count: int = 2
    confidence_threshold: float = 0.8

    # 内部状态
    _current_round: int = 0
    _query_queue: List[str] = field(default_factory=list)
    _searched_queries: Set[str] = field(default_factory=set)

    async def decompose_query(self, question: str) -> List[str]:
        """分解原始问题为子查询"""
        sub_queries = await self.query_decomposer.decompose(question)
        self._query_queue.extend(sub_queries)
        return sub_queries

    def get_next_queries(self) -> List[str]:
        """获取下一批待搜索的查询，去重"""
        batch = []
        while self._query_queue and len(batch) < self.max_queries_per_round:
            q = self._query_queue.pop(0)
            if q not in self._searched_queries:
                batch.append(q)
                self._searched_queries.add(q)
        return batch

    async def process_results(self, results: List[SearchResult], question: str):
        """处理搜索结果，抽取证据"""
        for result in results:
            evidence = await self.evidence_store.extract_and_store(result, question)

    def is_sufficient(self) -> bool:
        """判断已收集的证据是否足够回答问题"""
        return self.evidence_store.get_evidence_count() >= self.min_evidence_count \
            and self.evidence_store.get_confidence() >= self.confidence_threshold

    async def generate_followup_queries(self, question: str) -> List[str]:
        """基于已有证据的信息缺口生成后续查询"""
        evidence_summary = self.evidence_store.get_summary()
        new_queries = await self.query_decomposer.generate_followup(
            question, evidence_summary
        )
        self._query_queue.extend(new_queries)
        return new_queries

    async def synthesize_answer(self, question: str) -> str:
        """基于收集的证据合成答案"""
        evidence = self.evidence_store.get_all_evidence()
        # 用 LLM 生成答案
        ...

    async def verify_answer(self, question: str, answer: str) -> VerificationResult:
        """验证候选答案"""
        return await self.answer_verifier.verify(
            question, answer, self.evidence_store
        )
```

### 5.3 QueryDecomposer (`agentica/search/query_decomposer.py`)

查询分解与改写。

```python
@dataclass
class QueryDecomposer:
    """
    查询分解器

    职责：
    1. 将复杂多跳问题分解为多个独立子查询
    2. 对每个子查询生成多角度表述（改写）
    3. 根据已有证据生成后续查询（填补信息缺口）
    """
    model: Any  # LLM model

    async def decompose(self, question: str) -> List[str]:
        """
        将问题分解为子查询列表

        策略：
        1. 识别问题中的关键实体和约束条件
        2. 为每个实体/约束生成独立的搜索查询
        3. 同时生成原始问题的多角度改写

        示例：
        输入: "获得2024年图灵奖的研究者在哪所大学任职？"
        输出: [
            "2024 Turing Award winner",
            "2024 ACM Turing Award recipient",
            "图灵奖 2024 获奖者",
        ]
        """
        prompt = QUERY_DECOMPOSE_PROMPT.format(question=question)
        response = await self.model.response([
            Message(role="system", content="You are a search query expert."),
            Message(role="user", content=prompt),
        ])
        return self._parse_queries(response.content)

    async def generate_followup(
        self, question: str, evidence_summary: str
    ) -> List[str]:
        """
        根据已有证据生成后续查询

        分析 evidence_summary 中的信息缺口，
        生成针对性的补充搜索查询
        """
        prompt = f"""Original question: {question}

Evidence collected so far:
{evidence_summary}

What key information is still missing to answer the question?
Generate 1-3 targeted search queries to fill the gaps.
If the evidence is sufficient, return empty list.
Output as JSON array of strings."""

        response = await self.model.response([
            Message(role="user", content=prompt),
        ])
        return self._parse_queries(response.content)
```

### 5.4 EvidenceStore (`agentica/search/evidence_store.py`)

证据收集与管理。

```python
@dataclass
class Evidence:
    """单条证据"""
    content: str                    # 证据内容
    source_url: str                 # 来源 URL
    source_title: str               # 来源标题
    query: str                      # 产生此证据的搜索查询
    entities: List[str]             # 抽取的实体
    relevance_score: float = 0.0    # 与问题的相关性评分
    timestamp: str = ""             # 信息时间戳（如有）

@dataclass
class EvidenceStore:
    """
    证据存储与管理

    职责：
    1. 从搜索结果中抽取结构化证据
    2. 去重和冲突检测
    3. 实体关系追踪
    4. 信息充分性评估
    """
    model: Any  # LLM model

    _evidence_list: List[Evidence] = field(default_factory=list)
    _entities: Dict[str, List[str]] = field(default_factory=dict)  # entity -> [related facts]

    async def extract_and_store(
        self, search_result: SearchResult, question: str
    ) -> Optional[Evidence]:
        """
        从搜索结果中抽取证据并存储

        使用 LLM 从搜索结果中抽取：
        1. 与问题直接相关的事实
        2. 关键实体及其属性
        3. 评估相关性评分
        """
        prompt = EVIDENCE_EXTRACT_PROMPT.format(
            question=question,
            content=search_result.content,
            source=search_result.url,
        )
        response = await self.model.response([
            Message(role="user", content=prompt),
        ])
        evidence = self._parse_evidence(response.content, search_result)
        if evidence and evidence.relevance_score > 0.3:
            self._evidence_list.append(evidence)
            # 更新实体追踪
            for entity in evidence.entities:
                self._entities.setdefault(entity, []).append(evidence.content)
        return evidence

    def get_evidence_count(self) -> int:
        """获取有效证据数量"""
        return len(self._evidence_list)

    def get_confidence(self) -> float:
        """
        评估当前证据的整体置信度

        基于：
        1. 证据数量
        2. 多源一致性（不同来源是否指向相同结论）
        3. 平均相关性评分
        """
        if not self._evidence_list:
            return 0.0

        # 平均相关性
        avg_relevance = sum(e.relevance_score for e in self._evidence_list) / len(self._evidence_list)

        # 来源多样性
        unique_sources = len(set(e.source_url for e in self._evidence_list))
        source_factor = min(unique_sources / 2.0, 1.0)  # 至少2个不同来源

        return avg_relevance * source_factor

    def get_summary(self) -> str:
        """获取当前证据摘要，用于生成后续查询"""
        if not self._evidence_list:
            return "No evidence collected yet."
        lines = []
        for i, e in enumerate(self._evidence_list, 1):
            lines.append(f"{i}. [{e.source_title}] {e.content[:200]}")
        return "\n".join(lines)

    def get_all_evidence(self) -> List[Evidence]:
        """获取所有证据"""
        return sorted(self._evidence_list, key=lambda e: e.relevance_score, reverse=True)
```

### 5.5 AnswerVerifier (`agentica/search/answer_verifier.py`)

答案验证器。

```python
@dataclass
class VerificationResult:
    """验证结果"""
    is_confident: bool          # 是否有信心
    confidence_score: float     # 置信度评分
    reasoning: str              # 验证推理过程
    conflicting_evidence: List[str]  # 矛盾证据

@dataclass
class AnswerVerifier:
    """
    答案验证器

    职责：
    1. 检查候选答案是否与所有证据一致
    2. 识别矛盾证据
    3. 通过反向搜索验证（可选）
    4. 参考 BrowseComp-Plus 的 JUDGE_PROMPT 做 self-judge
    """
    model: Any  # LLM model

    async def verify(
        self,
        question: str,
        candidate_answer: str,
        evidence_store: EvidenceStore,
    ) -> VerificationResult:
        """
        验证候选答案

        步骤：
        1. 检查答案与每条证据的一致性
        2. 识别矛盾
        3. 评估整体置信度
        """
        evidence_list = evidence_store.get_all_evidence()
        evidence_text = "\n".join(
            f"[{e.source_title}] {e.content}" for e in evidence_list
        )

        prompt = ANSWER_VERIFY_PROMPT.format(
            question=question,
            candidate_answer=candidate_answer,
            evidence=evidence_text,
        )

        response = await self.model.response([
            Message(role="user", content=prompt),
        ])

        return self._parse_verification(response.content)

    async def reverse_verify(
        self, question: str, answer: str, search_fn
    ) -> bool:
        """
        反向验证：用答案作为关键词搜索，确认答案的正确性

        例如：问题是"2024年图灵奖获得者在哪里任职"
        答案是"MIT"
        反向搜索 "2024 Turing Award MIT" 确认关联
        """
        reverse_query = f"{answer} {question[:50]}"
        results = await search_fn(reverse_query)
        # 用 LLM 判断搜索结果是否支持答案
        ...
```

## 六、与 DeepAgent 的集成方式

### WebSearchAgent 如何复用 DeepAgent

```
DeepAgent
  │
  ├─ 内置工具（web_search, fetch_url, task 等）    ← 直接复用
  ├─ hooks 机制（pre/tool/post_tool_hook）         ← 增强，不替换
  ├─ 上下文管理（双阈值）                          ← 直接复用
  ├─ 重复检测                                      ← 直接复用
  └─ 子 Agent 并行执行                              ← 直接复用

WebSearchAgent (extends DeepAgent)
  │
  ├─ SearchOrchestrator                            ← 新增
  │   ├─ QueryDecomposer                           ← 新增
  │   ├─ EvidenceStore                             ← 新增
  │   └─ AnswerVerifier                            ← 新增
  │
  ├─ 增强的 post_tool_hook                         ← 在原有基础上追加
  │   └─ 每次搜索/抓取后自动调用 EvidenceStore
  │
  └─ deep_search 工具                              ← 新增注册的工具方法
```

### hooks 增强（不替换）

```python
def _enhance_hooks(self):
    """在 DeepAgent 的 hooks 基础上追加 WebSearchAgent 的逻辑"""
    original_post_hook = self.model._post_tool_hook

    def enhanced_post_hook(function_call_results):
        # 先执行原有的 post_hook（反思/检查点）
        if original_post_hook:
            original_post_hook(function_call_results)

        # 追加：从工具结果中抽取证据
        for msg in function_call_results:
            if msg.role == "tool" and hasattr(msg, 'tool_name'):
                if msg.tool_name in ("web_search", "fetch_url"):
                    # 异步提取证据（fire-and-forget 或同步等待）
                    asyncio.ensure_future(
                        self._orchestrator.evidence_store.extract_and_store(
                            SearchResult(content=msg.content, ...),
                            self._current_question
                        )
                    )

    self.model._post_tool_hook = enhanced_post_hook
```

## 七、共性组件后续复用计划

以下组件设计为独立模块，后续可直接给 `DeepAgent` 和 `Agent` 使用：

| 组件 | 当前用途 | 后续复用场景 |
|------|---------|-------------|
| `QueryDecomposer` | 搜索查询分解 | DeepAgent 的 web_search 调用前自动分解 |
| `EvidenceStore` | 搜索证据管理 | DeepAgent 的 research 子 Agent 共享证据 |
| `AnswerVerifier` | 答案验证 | Agent 的通用输出质量验证 |
| `SearchOrchestrator` | 搜索流程编排 | DeepAgent 的 research 子 Agent 的默认编排 |

## 八、评测适配（BrowseComp-Plus）

### 评测流程

```python
# evaluation/run_browsecomp_plus.py

async def evaluate_browsecomp_plus():
    agent = WebSearchAgent(
        model=OpenAIChat(id="gpt-4o"),
        max_search_rounds=15,
        confidence_threshold=0.8,
        enable_query_decomposition=True,
        enable_answer_verification=True,
        enable_step_reflection=True,
        enable_context_overflow_handling=True,
    )

    for instance in dataset:
        result = await agent.deep_search(instance["question"])
        # ... 评测逻辑
```

### BrowseComp-Plus 特殊处理

1. **精确答案提取**：在 `synthesize_answer` 中强制输出精确、简短的答案（不是长文）
2. **Hard Negatives 对抗**：`AnswerVerifier` 的交叉验证机制天然对抗误导信息
3. **固定语料库适配**：如果评测使用固定语料库，可以替换 `web_search` 工具为基于语料库的检索（向量检索 + BM25），其余架构不变

## 九、实现优先级

| 阶段 | 任务 | 预期收益 |
|------|------|---------|
| P0 | `QueryDecomposer` + 查询分解/改写 | 大幅提升多跳问题的搜索召回率 |
| P0 | `EvidenceStore` + 证据抽取/追踪 | 结构化管理搜索发现，支撑后续决策 |
| P0 | `WebSearchAgent` 主框架 + `SearchOrchestrator` | 程序化搜索编排 |
| P1 | `AnswerVerifier` + 交叉验证 | 降低被 Hard Negatives 误导的概率 |
| P1 | 评测脚本 `evaluation/run_browsecomp_plus.py` | 端到端跑通 |
| P2 | EvidenceStore 的信息充分性自动判断 | 精准控制搜索终止时机 |
| P2 | 反向验证搜索 | 进一步提升答案可靠性 |

## 十、与 openJiuwen 方案的对比

| 能力 | openJiuwen | 我们的方案 | 差异说明 |
|------|-----------|-----------|---------|
| 状态空间搜索 | Action Pool + 概率采样 | SearchOrchestrator + 查询队列 | 我们用更简洁的队列模型，避免过度工程化 |
| 实体认知引擎 | 完整实体图谱 | EvidenceStore 的实体追踪 | 轻量级实现，实体作为证据的属性而非独立图谱 |
| 并行路径探索 | 多分支推理树 | 子 Agent 并行 + 批量查询 | 复用现有 task(research) 并行能力 |
| 上下文引擎 | 分层存储 + 异步压缩 | 双阈值压缩 + EvidenceStore 独立缓冲 | 证据不受压缩影响，与上下文分离 |
| 多搜索引擎 | 多引擎 fallback | 单引擎 + 多角度查询改写 | 通过改写弥补单引擎的召回不足 |
| 答案验证 | 证据链闭环 | AnswerVerifier + 反向搜索 | 功能对等 |
| 自演进 | 外置记忆 + 自优化 | 暂不实现 | 非本阶段目标 |
