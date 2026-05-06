[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: AIエージェントの構築
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.12%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#コミュニティとサポート)

**Agentica** は軽量な Python フレームワークで、AIエージェントの構築に使用します。Async-First アーキテクチャで、ツール呼び出し、RAG、マルチエージェントチーム、ワークフローオーケストレーション、MCP プロトコルをサポートします。

## アーキテクチャ

Agentica は、低レベルのモデルルーティングから高レベルのマルチエージェントオーケストレーションまで、完全な抽象化スタックを提供します：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### コア実行エンジン (Agentic Loop)

Agent のコアは、ツール呼び出しによって厳密に駆動される決定論的な `while(true)` エンジン内で実行され、無限ループ防止、コスト追跡、コンテキストのマイクロ圧縮 (Compaction)、および4層のガードレールシステムが組み込まれています：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## インストール

```bash
pip install -U agentica
```

## クイックスタート

`asyncio` を学ぶ必要はありません。`run_sync` は内部で完全な agentic loop
（並列ツール呼び出し、ストリーミング、圧縮、リトライ）を実行しますが、
外から見れば普通の同期関数です：

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
result = agent.run_sync("北京を一文で紹介してください")
print(result.content)
```

```
北京は中国の首都であり、三千年以上の歴史を持つ文化都市で、政治・文化・国際交流の中心地です。
```

まず API キーを設定してください：

```bash
export OPENAI_API_KEY="sk-xxx"              # OpenAI
export ZHIPUAI_API_KEY="your-api-key"       # ZhipuAI（glm-4.7-flash は無料）
export DEEPSEEK_API_KEY="your-api-key"      # DeepSeek
```

### 同期 vs 非同期

| コードスタイル | 推奨 API |
|---|---|
| 通常スクリプト / Jupyter / FastAPI ルート（デフォルト） | `agent.run_sync(...)`、`agent.print_response_sync(...)`、`for chunk in agent.run_stream_sync(...)` |
| 既に asyncio イベントループ内 / 複数 agent を並列実行したい | `await agent.run(...)`、`async for chunk in agent.run_stream(...)` |

`run_sync` は内部的には `asyncio.run(self.run(...))` で、ツール呼び出しは
`asyncio.gather` により並行実行されます。**同期 API は性能を犠牲にしません**——
イベントループを隠しているだけです。

```python
import asyncio
from agentica import Agent, OpenAIChat

async def main():
    agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
    result = await agent.run("上海を一文で紹介してください")
    print(result.content)

asyncio.run(main())
```

### 推奨インポート方法

コア SDK と組み込みツールはトップレベルでエクスポート済みで、長いパスを覚える必要はありません：

```python
from agentica import (
    Agent, DeepAgent, Workspace, tool,
    OpenAIChat,                                       # openai はコア依存
    BuiltinFileTool, BuiltinExecuteTool,              # ファイル / 実行
    BuiltinFetchUrlTool, BuiltinWebSearchTool,        # Web
    BuiltinTodoTool, BuiltinTaskTool,                 # タスク管理 / サブエージェント
    HistoryConfig, WorkspaceMemoryConfig, RunConfig,  # 設定
)

# 他のモデル / 重いツールはサブモジュール経由（起動時の依存読み込みを回避）
from agentica.model.anthropic.claude import Claude   # pip install anthropic
from agentica.model.ollama.chat import Ollama
from agentica.tools.shell_tool import ShellTool
```

## 機能

- **Async-First** — ネイティブ async API、`asyncio.gather()` による並列ツール実行、同期アダプター対応
- **Runner Agentic Loop** — LLM ↔ ツール呼び出し自動ループ、多ターン連鎖推論、無限ループ検出、コスト予算、圧縮パイプライン、API リトライ
- **20以上のモデル** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM など
- **40以上の組み込みツール** — 検索、コード実行、ファイル操作、ブラウザ、OCR、画像生成
- **RAG** — ナレッジベース管理、ハイブリッド検索、Rerank、LangChain / LlamaIndex 統合
- **マルチエージェント** — `Agent.as_tool()`（軽量合成）、Swarm（並列 / 自律）、Workflow（確定的オーケストレーション）
- **Actor-Critic 精錬** — `refine()` による複数 Critic 並列レビュー、`SchemaCritic` のゼロコストプログラム検証 / `AgentCritic` の異種強モデル監査、ループ検出による自動早期停止
- **ガードレール** — 入力 / 出力 / ツールレベルのガードレール、ストリーミングリアルタイム検出
- **MCP / ACP** — Model Context Protocol と Agent Communication Protocol のサポート
- **スキルシステム** — Markdown ベースのスキル注入、モデル非依存
- **マルチモーダル** — テキスト、画像、音声、動画の理解
- **永続メモリ** — インデックス / コンテンツ分離、関連性ベースの想起、4タイプ分類、ドリフト防御

## Workspace メモリ

Workspace はセッション間で永続するメモリを提供し、インデックス / 想起設計を採用しています：

```python
from agentica import Workspace

workspace = Workspace("./workspace")
workspace.initialize()

# 型付きメモリエントリを書き込み（各エントリは独立ファイル、インデックス自動更新）
await workspace.write_memory_entry(
    title="Python Style",
    content="User prefers concise, typed Python.",
    memory_type="feedback",              # user|feedback|project|reference
    description="python coding style",   # 関連性スコアリング用キーワード
)

# 関連性ベースの想起（クエリに最も関連する上位 ≤5 件を返す）
memory = await workspace.get_relevant_memories(query="how to write python")
```

Agent は現在のクエリに最も関連するメモリを自動的に想起し、全メモリを注入することはありません：

```python
from agentica import Agent, Workspace
from agentica.agent.config import WorkspaceMemoryConfig

agent = Agent(
    workspace=Workspace("./workspace"),
    long_term_memory_config=WorkspaceMemoryConfig(
        max_memory_entries=5,  # 最大 5 件の関連メモリを注入
    ),
)
```

## 自己進化（Self-Evolution）

Agentica は「事実を覚える」だけでなく、**「やり方を覚える」** ことができます。Agent がツールを実行する過程で発生するすべてのシグナル（ツール失敗、ユーザーからの修正、成功シーケンス）は **Experience イベント** として収集され、ルールベースで **経験カード（cards）** にコンパイルされます。同一のルールが N 回繰り返し確認されると、**`SKILL.md` が自動生成され**、workspace の `generated_skills/` ディレクトリに保存されます。次のセッションで新しい Agent を起動すると、`SkillTool` がそのスキルを自動検出して注入し、過去に学んだやり方を新セッションでそのまま再利用できます。

パイプライン全体はローカル・監査可能・外部依存ゼロで動作します。決定論的な収集（tool error / success）は LLM コストゼロ、「ユーザー修正の分類」と「新スキルを生成すべきか」の 2 ステップのみが `auxiliary_model` を使用します。

### フロー図

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

イベント収集（黄）→ 経験コンパイルとライフサイクル（青ストレージ + 破線グレー枠）→ スキル生成ゲート + LLM 判定（ピンク）→ 次セッションでの自動再利用。全工程が監査可能で外部依存ゼロ。

### 有効化の方法

最小変更：`ExperienceCaptureHooks` を Agent に取り付け、`ExperienceConfig.skill_upgrade=SkillUpgradeConfig(mode="shadow")` を設定するだけで自己進化の完全クローズドループが有効になります。

```python
from agentica import Agent, Workspace, OpenAIChat
from agentica.agent.config import ExperienceConfig, SkillUpgradeConfig
from agentica.hooks import (
    ConversationArchiveHooks,
    ExperienceCaptureHooks,
    MemoryExtractHooks,
    _CompositeRunHooks,
)

workspace = Workspace("./workspace", user_id="alice")
workspace.initialize()

model = OpenAIChat(id="gpt-4o-mini")

hooks = _CompositeRunHooks([
    ConversationArchiveHooks(),                # 会話を自動アーカイブ
    MemoryExtractHooks(),                      # LLM が長期メモリを抽出
    ExperienceCaptureHooks(
        ExperienceConfig(
            capture_tool_errors=True,          # 決定論的、LLM コスト 0
            capture_success_patterns=True,     # 決定論的、LLM コスト 0
            capture_user_corrections=True,     # auxiliary_model で分類
            feedback_confidence_threshold=0.6,
            promotion_count=3,                 # 同一ルールが 3 回確認 → tier=hot
            skill_upgrade=SkillUpgradeConfig(
                mode="shadow",                 # off | draft | shadow
                min_repeat_count=3,            # spawn 検討に最低 3 回必要
                min_tier="warm",
                min_success_applications=1,    # コールドスタート demo は 0 に
            ),
        )
    ),
])

agent = Agent(
    model=model,
    auxiliary_model=model,                     # 修正分類 / skill 判定で使用
    workspace=workspace,
)
agent._default_run_hooks = hooks

# 通常のワークロードを実行 — 失敗 / 修正 / 成功は workspace に蓄積される
agent.run_sync("./docs/agent.md を読んでください")
```

数セッション実行後、workspace は次のような状態になります：

```
workspace/users/alice/
├── experiences/
│   ├── events.jsonl                        # 全生イベント（append-only）
│   ├── EXPERIENCE.md                       # カードインデックス
│   └── <hash>__list_dir_before_read.md     # コンパイル済み経験カード
├── generated_skills/
│   ├── INDEX.md                            # L1 キーワードルーター
│   └── list-dir-before-read/
│       ├── SKILL.md                        # 自動生成された再利用可能スキル
│       └── meta.json                       # status: shadow / draft / promoted
└── reports/learning/                       # 各 run の学習レポート
```

完全な e2e demo（Session 1 で自己進化により skill 生成 → Session 2 で全く新しい Agent がクロスセッションで再利用）：[`examples/workspace/03_self_evolution_e2e.py`](examples/workspace/03_self_evolution_e2e.py)。

> **トレードオフ**：`mode="shadow"` は workspace ローカルに自動インストールされ、他ユーザーには影響しません。`mode="draft"` はドラフトのみ生成しインストールせず、人間レビュー向きです。`mode="off"` は skill 自動生成を無効化（経験カードの収集は継続）。`min_success_applications` は「最低 N 回の `tool_recovery` イベントが必要」という安全ゲート — Agent が永遠に解決できないタスクから skill を生成するのを防ぎます。コールドスタート demo のときのみ `0` に設定してください。

## Actor-Critic 精錬（refine）

Agentica は **プロトコルレベルの Actor-Critic パターン** を提供します：Actor がドラフトを生成し、複数の Critic が並列でレビュー、却下されたドラフトはフィードバックに基づいて修正され、全 Critic 承認またはループ早期停止で終了します。これは [CarePilot 論文（arXiv:2603.24157）](https://arxiv.org/abs/2603.24157) で検証されたアーキテクチャ — 7B のファインチューニングモデル + Actor-Critic フレームワークが医療 GUI ベンチマークで 48.9% のタスク完了率を達成し、ゼロショット GPT-5 を約 13 ポイント上回りました。アブレーションでは Critic ループを除外するとタスク精度が 48.9% → 12.5% に急落しています。

agentica の設計原則は **「SDK は能力ではなくプロトコルを提供する」** です。基盤モデル内の汎用的な自己批判は LLM 自体に任せ、SDK は LLM では代替不可能な 3 つを担当します —

- **ビジネス制約の注入** — `SchemaCritic` は Pydantic スキーマ検証によるゼロ LLM コストの決定論的検証器（スキーマ適合性ではいかなる LLM critic にも勝つ）
- **監査可能なリフレクション trail** — `RefineResult.history` は各ラウンドのドラフトと critic ごとの verdict を記録、全プロセスが観測・再現可能
- **異種構成** — 安価な Actor + 強力な Critic、複数 Critic 並列レビュー、決定論的検証器と LLM 検証器の混在 — SDK レイヤだけが表現可能

```python
from pydantic import BaseModel
from agentica import Agent, OpenAIChat
from agentica.critic import SchemaCritic, AgentCritic, CritiqueStyle, refine

class Reply(BaseModel):
    intent: str
    confidence: float

actor = Agent(name="writer", model=OpenAIChat(id="gpt-4o-mini"))
reviewer = Agent(
    name="reviewer",
    model=OpenAIChat(id="gpt-4o"),
    instructions="正しさを確認し、問題なければ APPROVED と返答、そうでなければ問題を箇条書きしてください。",
)

result = await refine(
    actor,
    task="分類: 'バスは何時に出発しますか？'",
    critics=[
        SchemaCritic(Reply),                                  # プログラムレベル（ゼロコスト）
        AgentCritic(reviewer, style=CritiqueStyle.STRICT),    # LLM レベル（スタイル指定可）
    ],
    max_iter=3,
)

print(result.final_draft)        # 最終ドラフト
print(result.approved)           # True / False
print(result.stopped_reason)     # approved / max_iter / loop_detected / no_critics
print(result.iterations)         # 実際のレビュー回数
for round_ in result.history:    # ラウンドごとの完全 trail（監査可能）
    print(round_.draft, [v.approved for v in round_.verdicts])
```

**主な特徴**：

- `Critic` Protocol — duck-typed、カスタム critic（regex / API call / 任意のビジネスルール）は 20 行未満で実装可能
- 複数 Critic 並列実行（`asyncio.gather`）、LLM critic とゼロコストプログラム critic を自由に混在
- `CritiqueStyle.STRICT/NEUTRAL/LENIENT` — LLM critic のレビュー温度を制御（論文ではデフォルト NEUTRAL を推奨）
- **ループ検出による早期停止** — 連続 2 ラウンドで同じ verdict が出た場合、自動的に終了しトークンの浪費を防止
- `RefineResult.history` は完全なリフレクション trail を提供し、CHAT ログ（`logger.chat`）と連携してマルチエージェントの可観測性を実現

**使うべきでない場合**：基盤モデルが GPT-5+ クラスでタスクにビジネス固有の制約がない場合、`refine()` は限られたゲインのためにトークンを浪費します。使うべきは：(1) 出力がプログラム的スキーマを満たす必要がある、(2) 異種 actor + critic（安価な actor + 強力な critic）が欲しい、(3) 汎用的な自己批判では強制できないビジネス上のレッドラインがある場合。

完全なサンプル：[`examples/agent_patterns/04_debate.py`](examples/agent_patterns/04_debate.py)（`AgentCritic` で構造化反論を抽出するマルチエージェント討論）。

## Agent レシピ（Recipes）

`Agent` のパラメータは多いですが、よく使う組み合わせは以下の 5 つで十分です。コピペしてどうぞ：

### ワンショットスクリプト（最小）

```python
agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
print(agent.run_sync("北京を一文で紹介してください").content)
```

### マルチターン会話

```python
agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=5,
)
agent.run_sync("私の名前は Alice、ML エンジニアです。")
agent.run_sync("私の名前は？")  # モデルは覚えています
```

### ツール型 Agent（カスタムツール組み合わせ）

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Python 3.13 の新機能を調べて features.md に書いてください")
```

### マルチユーザー + 長期記憶 + 会話アーカイブ

ユーザーごとに 1 つの Agent インスタンス。`session_id` は通常 `user_id` と同じで OK：

```python
from agentica import Agent, OpenAIChat, Workspace, WorkspaceMemoryConfig

def create_agent(user_id: str) -> Agent:
    return Agent(
        model=OpenAIChat(id="gpt-4o-mini"),
        workspace=Workspace("~/.agentica/workspace", user_id=user_id),
        session_id=user_id,                      # 会話ログは ~/.agentica/projects/.../{user_id}.jsonl に保存
        enable_long_term_memory=True,            # ← 必須：明示的に有効化
        long_term_memory_config=WorkspaceMemoryConfig(
            auto_archive=True,                   # 各 run 後に会話をアーカイブ
            auto_extract_memory=True,            # LLM が記憶エントリを自動抽出
        ),
        add_history_to_context=True,
        num_history_turns=5,
    )
```

> **よくある落とし穴**：`long_term_memory_config` を設定しても `enable_long_term_memory=True` を忘れると、すべての記憶/アーカイブ機能が静かに無視されます。v1.4.1 から `Agent.__init__` がこの設定ミスを警告します。

### 長セッションのトークン削減：履歴メッセージのカスタマイズ

検索ツールの結果は通常巨大で後続ターンでは不要なことが多いので、履歴から削除し、AI の返答は切り詰められます：

```python
from agentica import Agent, OpenAIChat, HistoryConfig

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    add_history_to_context=True,
    num_history_turns=10,
    history_config=HistoryConfig(
        excluded_tools=["search_*", "web_search"],   # マッチするツール結果を削除、対応する tool_calls も自動的に剥離
        assistant_max_chars=200,                      # AI の返答を 200 文字に切り詰め
    ),
)
```

より複雑なフィルタ（ユーザープロンプトの prefix 削除、metadata によるメッセージ削除など）には `history_filter` コールバックを使います。`examples/memory/03_history_filter.py` を参照。

### フルパワー（CLI / Gateway / 長時間タスク）

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ 組み込みツール + 圧縮 + 長期記憶 + skills + MCP、すぐに使える
```

## CLI

```bash
agentica --model_provider zhipuai --model_name glm-4.7-flash
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

## Web UI

[agentica-gateway](https://github.com/shibing624/agentica-gateway) を通じて Web ページを提供し、Feishu アプリや企業微信から Agentica を直接利用することもできます。

## サンプル

完全なサンプルは [examples/](https://github.com/shibing624/agentica/tree/main/examples) をご覧ください：

| カテゴリ | 内容 |
|----------|------|
| **基本** | Hello World、ストリーミング、構造化出力、マルチターン、マルチモーダル、**Agentic Loop 比較** |
| **ツール** | カスタムツール、Async ツール、検索、コード実行、並列ツール、並行安全、コスト追跡、サンドボックス隔離、圧縮 |
| **エージェントパターン** | Agent-as-Tool、並列実行、チームコラボレーション、ディベート、ルーティング、Swarm、サブエージェント、モデルレイヤーフック、セッション復元 |
| **ガードレール** | 入力 / 出力 / ツールレベルのガードレール、ストリーミングガードレール |
| **メモリ** | セッション履歴、WorkingMemory、コンテキスト圧縮、Workspace メモリ、LLM 自動メモリ |
| **RAG** | PDF Q&A、高度な RAG、LangChain / LlamaIndex 統合 |
| **ワークフロー** | データパイプライン、投資リサーチ、ニュースレポート、コードレビュー |
| **MCP** | Stdio / SSE / HTTP トランスポート、JSON 設定 |
| **可観測性** | Langfuse、トークン追跡、Usage 集約 |
| **アプリケーション** | LLM OS、ディープリサーチ、カスタマーサービス、**金融リサーチ（6-Agent パイプライン）** |

[→ 完全なサンプルディレクトリを見る](https://github.com/shibing624/agentica/blob/main/examples/README.md)

## ドキュメント

完全なドキュメント：**https://shibing624.github.io/agentica**

## コミュニティとサポート

- **GitHub Issues** — [issue を開く](https://github.com/shibing624/agentica/issues)
- **WeChat Group** — WeChat で `xuming624` を追加し、「llm」と伝えて開発者グループに参加

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/wechat.jpeg" width="200" />

## 引用

研究で Agentica を使用する場合は、以下を引用してください：

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

## ライセンス

[Apache License 2.0](LICENSE)

## 貢献

貢献を歓迎します！[CONTRIBUTING.md](CONTRIBUTING.md) をご覧ください。

## 謝辞

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
