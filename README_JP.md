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

**Agentica** は単なる LLM API のチャットラッパーではなく、Async-First の agent harness です。エージェントを本当に「動かす」ためのものです：ツール呼び出し、長時間タスク、マルチエージェント協調、セッションをまたぐ記憶の保持、そして Skill system による進化可能な self-learn ワークフローへの接続。

| 能力 | 説明 |
|------|------|
| **Long-running Agent Loop** | `Runner` が駆動する LLM ↔ ツールループ。圧縮・リトライ・コスト予算・無限ループ防止を内蔵 |
| **Works Beyond Chat** | ファイル・実行・検索・ブラウザ・MCP・マルチエージェント・Workflow。単一 IDE シナリオに依存しない |
| **Memory That Survives Sessions** | Workspace 記憶はエントリ単位で保存・関連性想起でき、確認済みの好みを `~/.agentica/AGENTS.md` に同期 |
| **Skill-Based Self-Learn** | SkillTool が外部スキルをロード；内蔵 Agent の継続学習戦略 |
| **Self-Evolution** | ツール失敗 / ユーザー修正 / 成功シーケンス → 経験カード → `SKILL.md` を自動生成、セッションをまたいで再利用 |
| **Open, Composable Harness** | モデル・ツール・記憶・Skill・Guardrails・MCP はすべて置換可能な部品。閉鎖的な SaaS ブラックボックスではない |

## 🔥 News

- [2026/07/05] **v1.4.7**：CLI に統一 braille スピナー（thinking/tool/answering 全フェーズで回転し、稼働中とハングを目視で判別可能）を追加；`ask_user_input` の入力フリーズと `/btw` が主モデルを汚染するバグを修正；cron ランタイム（`/cron` コマンド + デーモン）、自己管理（`/upgrade`、`/config set|env`）を追加；設定を `~/.agentica/config.yaml` に統一（main + aux model、`cli_config.json`/`task_model` を削除、コメントを保持）；`/resume` が完全/プレフィックス/省略 session id に対応。stream upload の OOM と `/api/upload` のパストラバーサル（CWE-22）も修正。詳細は [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**：クロスプロバイダー fallback がツール呼び出しターンに対応——fallback モデルがツールを呼び出して最終回答を生成でき、そのプロバイダー固有の履歴は圧縮され、主モデルへのリプレイがクリーンに保たれます。fallback モデルは run ごとにクローンされ並行安全性を確保。編集時 LSP 診断 CLI フラグ（`--enable-diagnostics`/`--diagnostics-server`）、強化版 `agentica doctor`、`/checkpoint restore --yes` 確認、`/goal` 予算フラグを追加。詳細は [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**：MemoryExtractHooks の最適化——新しい `auto_extract_memory_background` がメモリ抽出をバックグラウンドで実行（`on_agent_end` をブロックしなくなりました）、抽出は高速・低コストな `auxiliary_model` を優先。詳細は [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**：Skill ライフサイクルのリファクタリング + VaG の分離——VaG 実験コードは `evaluation/vag/` 研究モジュールへ移動、統一された `SkillLifecycleHooks` 拡張ポイントを追加。詳細は [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

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

## 機能

- **Async-First** — ネイティブ async API、`asyncio.gather()` による並列ツール実行、同期アダプター対応
- **Runner Agentic Loop** — LLM ↔ ツール呼び出し自動ループ、多ターン連鎖推論、無限ループ検出、コスト予算、圧縮パイプライン、API リトライ
- **20以上のモデル** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM など
- **40以上の組み込みツール** — 検索、コード実行、ファイル操作、ブラウザ、OCR、画像生成
- **RAG** — ナレッジベース管理、ハイブリッド検索、Rerank、LangChain / LlamaIndex 統合
- **マルチエージェント** — `Agent.as_tool()`（軽量合成）、Swarm（並列 / 自律）、Workflow（確定的オーケストレーション）
- **Actor-Critic 精錬** — `refine()` による複数 Critic 並列レビュー、`SchemaCritic` のゼロコストプログラム検証 / `AgentCritic` の異種強モデル監査、ループ検出による自動早期停止
- **`/goal` 長時間タスク** — `await agent.run_goal("xxx")` で目標に向けて継続的に推進、完了・再開・一時停止を自動判定；token / wall-clock / turn の 3 種ハードキャップ対応；CLI の `/goal /subgoal` はそのまま使えます。詳細は [ドキュメント](https://shibing624.github.io/agentica/advanced/goals)
- **ガードレール** — 入力 / 出力 / ツールレベルのガードレール、ストリーミングリアルタイム検出
- **MCP / ACP** — Model Context Protocol と Agent Communication Protocol のサポート
- **スキルシステム** — Markdown ベースのスキル注入、モデル非依存
- **マルチモーダル** — テキスト、画像、音声、動画の理解
- **永続メモリ** — インデックス / コンテンツ分離、関連性ベースの想起、4タイプ分類、ドリフト防御

## 自己進化（Self-Evolution）

Agentica は「事実を覚える」だけでなく、**「やり方を覚える」** ことができます。

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

## Agent レシピ（Recipes）

### カスタムツールの組み合わせ

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Python 3.13 の新機能を調べて features.md に書いてください")
```

### フルパワー（CLI / Gateway / 長時間タスク）

```python
from agentica import DeepAgent
agent = DeepAgent()  # 40+ 組み込みツール + 圧縮 + 長期記憶 + skills + MCP、すぐに使える
```

## CLI

```bash
agentica 
```

<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/cli_snap.png" width="800" />

### 長時間タスク：`/goal`

Agent に目標へ向けて継続的に推進させ、各ラウンド終了時に自動で完了を判定、未完了なら続行——judge が done と言うか、予算が尽きるか、ユーザーが手動で止めるまで。

CLI：

```text
/goal xxx 機能を実装し pytest を通す    # 目標設定 + 自動開始
/goal status                         # 状態・予算・subgoals を表示
/goal pause | resume | clear
/subgoal 単体テストを追加する            # 目標に受入条件を追加
```

完全な解説：[Standing Goal Loop ドキュメント](https://shibing624.github.io/agentica/advanced/goals)。

## Web UI / Gateway

**Gateway は現在 `agentica` メインライブラリに統合されています。**

Gateway ランタイムをインストール：

```bash
pip install -U "agentica[gateway]"
```

起動：

```bash
agentica-gateway
```
<img src="https://github.com/shibing624/agentica/blob/main/docs/assets/agentica-web.png" width="800" />

デフォルトでは `http://127.0.0.1:8789/chat` で起動します。

QQ / 飛書 / 微信 / 企微 / Telegram / Discord / Slack への接続をサポート。

定期タスクをサポート。

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
