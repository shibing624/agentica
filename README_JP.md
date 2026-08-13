[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/logo.png" height="150" alt="Agentica Logo">
  </a>
</div>

-----------------

# Agentica

**Agent を「何時間でも走らせる」——暴走せず、実際に手を動かし、使うほど強くなる。**
Async-first Python agent harness · 40+ ツール · 20+ モデル · MCP · CLI + Web Gateway

[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![GitHub stars](https://img.shields.io/github/stars/shibing624/agentica?style=social)](https://github.com/shibing624/agentica)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/shibing624/agentica/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://github.com/shibing624/agentica/blob/main/requirements.txt)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#コミュニティとサポート)

**Agentica** は単なる LLM API のチャットラッパーではなく、Async-First の agent harness です——エージェントを本当に「動かす」：ツール呼び出し、長時間タスク、マルチエージェント協調、セッションをまたぐ記憶、そして継続的な自己進化。

|  | |
|------|------|
| **長く走る、暴走しない** | `Runner` 駆動の LLM ↔ ツール長ループ。コンテキスト圧縮・コスト予算・無限ループ防止を内蔵し、長時間タスクが途切れない |
| **手を動かす、雑談だけではない** | ファイル・実行・検索・ブラウザ・MCP・マルチエージェント・Workflow——単一 IDE に縛られず実際に作業する |
| **複数セッション協調** | 端末間 peer メッセージ；`delegate` は独立プロセス（独自 context / cwd）；`task` は安価なプロセス内 subagent——役割が分かれている |
| **覚える、そして忘れる** | 記憶はエントリ単位で保存・関連性想起・drift 防御。常駐ルールは `users/{user_id}/AGENTS.md`（CLI default は `~/.agentica/AGENTS.md` symlink からも編集可能） |
| **使うほど強くなる** | ツール失敗 / ユーザー修正 / 成功シーケンスが経験カードになり、再利用可能な `SKILL.md` へ自動コンパイル、セッションをまたいで有効 |
| **すべて交換可能、ロックインしない** | モデル・ツール・記憶・Skill・Guardrails・MCP はすべて置換可能な部品。閉鎖的な SaaS ブラックボックスではない |

## インストール

```bash
pip install -U agentica
```

## 設定

API キーは 3 つの方法のいずれかで設定します（優先順位：シェル環境変数 > `.env` > `config.yaml`）：

```bash
export OPENAI_API_KEY="sk-xxx"
# 無料で始められる ZhipuAI の場合：export ZAI_API_KEY="your-api-key"
```

`~/.agentica/.env` に書き込むか、`agentica setup` を実行して `~/.agentica/config.yaml` を生成することもできます（CLI 内では `/model` でいつでもモデルを切り替え可能）。詳細は [インストールドキュメント](https://shibing624.github.io/agentica/getting-started/installation) を参照してください。

## クイックスタート

### CLI（まずはこれから）

```bash
agentica
```

対話ターミナルが起動したら、そのまま指示を入力してください（例：「このリポジトリのテストが落ちる原因を調べて」）。長時間タスクには `/goal`、複数セッション協調には `delegate` / peer メッセージを使います。詳細は後述の [CLI](#cli) セクションを参照してください。

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/cli_snap.png" width="800" alt="Agentica CLI スクリーンショット" />

### Python SDK

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

Agent に実際に「作業」させる——Web 検索してファイルに書き出す、`run_sync` 一発：

```python
from agentica import Agent, OpenAIChat, BuiltinWebSearchTool, BuiltinFileTool, BuiltinExecuteTool

agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[BuiltinWebSearchTool(), BuiltinFileTool(work_dir="./workspace"), BuiltinExecuteTool(work_dir="./workspace")],
)
agent.run_sync("Python 3.13 の新機能を調べて features.md に書いてください")
```

全部入りのフルパワー版（40+ 組み込みツール + 圧縮 + 長期記憶 + skills + MCP）はこちら：

```python
from agentica import DeepAgent
agent = DeepAgent()
```

## 機能

**コアエンジン**

- **Async-First** — ネイティブ async API、`asyncio.gather()` による並列ツール実行、同期アダプター対応
- **40以上の組み込みツール** — 検索、コード実行、ファイル操作、ブラウザ、OCR、画像生成
- **20以上のモデル** — OpenAI Chat Completions / [Responses API](https://shibing624.github.io/agentica/guides/openai-responses)、DeepSeek、Claude、ZhipuAI、Qwen、Moonshot、Ollama、LiteLLM など
- **ガードレール** — 入力 / 出力 / ツールレベルのガードレール、ストリーミングリアルタイム検出
- **マルチモーダル** — テキスト、画像、音声、動画の理解

**長時間タスクと協調**

- **`/goal` 長時間タスク** — `await agent.run_goal("xxx")` で目標に向けて継続的に推進、完了・再開・一時停止を自動判定；token / wall-clock / turn の 3 種ハードキャップ対応；CLI の `/goal /subgoal` はそのまま使えます。詳細は [ドキュメント](https://shibing624.github.io/agentica/advanced/goals)
- **マルチエージェント** — SDK：`Agent.as_tool()`、Workflow、Swarm、[Markdown Subagent](https://shibing624.github.io/agentica/multi-agent/subagent)；CLI：プロセス内 `task`、プロセス級 `delegate`、端末間 peer メッセージ（[端末ドキュメント](https://shibing624.github.io/agentica/getting-started/terminal)）
- **Actor-Critic 精錬** — `refine()` による複数 Critic 並列レビュー、`SchemaCritic` のゼロコストプログラム検証 / `AgentCritic` の異種強モデル監査、ループ検出による自動早期停止

**記憶と進化**

- **永続メモリ** — インデックス / コンテンツ分離、関連性ベースの想起、4タイプ分類、drift 防御；常駐ルールは `users/{user_id}/AGENTS.md`（CLI default は `~/.agentica/AGENTS.md` symlink からも編集可能）
- **スキルシステム** — Markdown ベースのスキル注入、プロジェクト級・ユーザー級・外部ホストの skill ディレクトリに対応
- **自己進化** — 経験カードがセッションをまたいで再利用できる `SKILL.md` に自動コンパイル（フローは下図）

**統合**

- **MCP / ACP** — Model Context Protocol と Agent Communication Protocol のサポート
- **RAG** — ナレッジベース管理、ハイブリッド検索、Rerank、LangChain / LlamaIndex 統合

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/evo_pipeline.png" width="900" alt="Agentica Self-Evolution Pipeline" />
</div>

## アーキテクチャ

Agentica は、低レベルのモデルルーティングから高レベルのマルチエージェントオーケストレーションまで、完全な抽象化スタックを提供します：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/architecturev2.jpg" width="800" alt="Agentica Architecture" />
</div>

### コア実行エンジン (Agentic Loop)

Agent のコアは、ツール呼び出しによって厳密に駆動される決定論的な `while(true)` エンジン内で実行され、無限ループ防止、コスト追跡、[2 層のコンテキスト圧縮](https://github.com/shibing624/agentica/blob/main/docs/advanced/compression.md)（無料の追い出し → LLM 要約）、および4層のガードレールシステムが組み込まれています：

<div align="center">
  <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agent_loop.png" width="800" alt="Agentica Loop Architecture" />
</div>

## CLI

```bash
agentica
```

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

### 協調：`task` / `delegate` / peer

| 仕組み | 何をするか | いつ使うか |
|------|--------|--------|
| `task` | プロセス内 subagent（デフォルト auxiliary モデル、読み取り専用） | コード検索・情報収集などの軽い作業 |
| `delegate` | 完全な `agentica --query --print` プロセスを別起動 | 独立した context / 別ディレクトリが必要な大きな作業；`/ps`、`wait`、`/stop` で管理可能 |
| peer | 2 つの対話ターミナル間でプレーンテキストを送受信（`list_agents` / `send_message`） | 別セッションに「こちらの変更内容」を伝えるためのもの。作業を丸投げする道具ではない |

選び方の詳細：[Choosing](https://shibing624.github.io/agentica/multi-agent/choosing) · [端末ドキュメント](https://shibing624.github.io/agentica/getting-started/terminal)。

## Web UI / IM Integration

```bash
pip install -U "agentica[gateway]"
```

起動：

```bash
agentica-gateway
```

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/agentica-web.png" width="800" alt="Agentica Web UI スクリーンショット" />

デフォルトでは `http://127.0.0.1:8881/chat` で起動します。

モバイル IM (QQ / Feishu / WeChat / WeCom / Telegram / Discord / Slack) への接続をサポートし、スケジュールタスク機能を内蔵しています。

IM 連携の詳細（スキャンコードバインディング、チャネル設定、環境変数）：[Gateway ドキュメント](https://github.com/shibing624/agentica/blob/main/docs/advanced/gateway.md)。

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

## 他の Agent CLI との比較

| | Agentica | Claude Code | Codex CLI | Gemini CLI |
|---|---|---|---|---|
| モデル選択 | ✅ 20+ プロバイダーを自由に切替 | Claude モデルのみ | OpenAI モデルのみ | Gemini モデルのみ |
| 端末間の複数セッション協調 | ✅ peer + `delegate` / `task` | ❌ | ❌ | ❌ |
| `/goal` 長時間タスクループ | ✅ 予算管理 + 完了自動判定 + 再開 | ❌ | ❌ | ❌ |
| Web UI + IM Gateway | ✅ WeChat / WeCom / Feishu / Telegram など本機に直結 | ❌ | ❌ | ❌ |
| 自己進化 Skill | ✅ 経験が `SKILL.md` に自動コンパイル | ❌ | ❌ | ❌ |
| Python SDK | ✅ 完全な SDK、任意のコードに組み込み可能 | 部分（Claude 限定） | ❌ | ❌ |
| オープンソース | ✅ Apache 2.0 | ❌ | ✅ | ✅ |

## 🔥 News

- [2026/08/10] **v1.4.12**：コンテキスト圧縮を二層に整理（約 70%→50% で古い tool result を淘汰 → LLM/native 要約）；読み直し無限ループと Anthropic 経路の圧縮未発火を修正。Layer 0 は回収可能かで落盤か切り詰めかを決め、圧縮回数は `RunResponse` に記録；CLI 前提を SDK に押し付けない。端末間 peer メッセージ（`list_agents` / `send_message`）とプロセス級 `delegate`（独立した `agentica --query --print`、`/ps` `/stop` `wait` で管理）を追加し、安価なプロセス内 `task` と役割を分離。詳細は [Release-v1.4.12](https://github.com/shibing624/agentica/releases/tag/v1.4.12)
- [2026/08/04] **v1.4.11**：OpenAI Responses API（ネイティブ compaction 含む）、Markdown 設定可能なサブエージェント、複数ファイル `apply_patch` を追加；CLI の resume/ステータス/圧縮フィードバックを改善；prompt と grep/glob schema のコストを削減；Learned Experiences の汚染と `write_todos` の全リストエコーを修正。詳細は [Release-v1.4.11](https://github.com/shibing624/agentica/releases/tag/v1.4.11)
- [2026/07/24] **v1.4.10**：カタログ駆動のモデル能力判定によるネイティブ画像入力を追加；`/rename` と名前指定の `/resume` を追加；Pillow コア依存関係のメタデータを修正。詳細は [Release-v1.4.10](https://github.com/shibing624/agentica/releases/tag/v1.4.10)
- [2026/07/21] **v1.4.9**：SDK/CLI/Web の権限を 3 階層（`ask`/`auto`/`allow-all`、yolo/full/strict 廃止）に統一；内蔵サブエージェントは読み取り専用化（`task` のデフォルトを `explore` に、edit/execute を禁止し aux モデルの低品質コード生成を修正）；`OpenAIChat` が OpenAI 互換プロキシから漏れた Claude `<invoke>` テキストツール呼び出しを解析；`edit_file` を硬拒否から tip 提示に変更；`ask_user_question` の CLI フリーズを修正。詳細は [Release-v1.4.9](https://github.com/shibing624/agentica/releases/tag/v1.4.9)

<details>
<summary>過去のバージョン</summary>

- [2026/07/05] **v1.4.7**：CLI に統一 braille スピナー（thinking/tool/answering 全フェーズで回転し、稼働中とハングを目視で判別可能）を追加；`ask_user_input` の入力フリーズと `/btw` が主モデルを汚染するバグを修正；cron ランタイム（`/cron` コマンド + デーモン）、自己管理（`/upgrade`、`/config set|env`）を追加；設定を `~/.agentica/config.yaml` に統一（main + aux model、`cli_config.json`/`task_model` を削除、コメントを保持）；`/resume` が完全/プレフィックス/省略 session id に対応。stream upload の OOM と `/api/upload` のパストラバーサル（CWE-22）も修正。詳細は [Release-v1.4.7](https://github.com/shibing624/agentica/releases/tag/v1.4.7)
- [2026/06/03] **v1.4.6**：クロスプロバイダー fallback がツール呼び出しターンに対応——fallback モデルがツールを呼び出して最終回答を生成でき、そのプロバイダー固有の履歴は圧縮され、主モデルへのリプレイがクリーンに保たれます。fallback モデルは run ごとにクローンされ並行安全性を確保。編集時 LSP 診断 CLI フラグ（`--enable-diagnostics`/`--diagnostics-server`）、強化版 `agentica doctor`、`/checkpoint restore --yes` 確認、`/goal` 予算フラグを追加。詳細は [Release-v1.4.6](https://github.com/shibing624/agentica/releases/tag/v1.4.6)
- [2026/05/11] **v1.4.4**：MemoryExtractHooks の最適化——新しい `auto_extract_memory_background` がメモリ抽出をバックグラウンドで実行（`on_agent_end` をブロックしなくなりました）、抽出は高速・低コストな `auxiliary_model` を優先。詳細は [Release-v1.4.4](https://github.com/shibing624/agentica/releases/tag/v1.4.4)
- [2026/05/10] **v1.4.3**：Skill ライフサイクルのリファクタリング + VaG の分離——VaG 実験コードは `evaluation/vag/` 研究モジュールへ移動、統一された `SkillLifecycleHooks` 拡張ポイントを追加。詳細は [Release-v1.4.3](https://github.com/shibing624/agentica/releases/tag/v1.4.3)

</details>

## ドキュメント

完全なドキュメント：**https://shibing624.github.io/agentica**

## コミュニティとサポート

> Agentica がお役に立てば、ぜひ ⭐ Star をお願いします——より多くの人に届きます！

- **GitHub Issues** — [issue を開く](https://github.com/shibing624/agentica/issues)
- **WeChat Group** — WeChat で `xuming624` を追加し、「llm」と伝えて開発者グループに参加

<img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/assets/wechat.jpeg" width="200" alt="WeChat グループ QR コード" />

## 引用

研究で Agentica を使用する場合は、以下を引用してください：

> Xu, M. (2026). Agentica: A Human-Centric Framework for Large Language Model Agent Workflows. GitHub. https://github.com/shibing624/agentica

BibTeX：

```bibtex
@misc{xu2026agentica,
  author    = {Xu, Ming},
  title     = {Agentica: A Human-Centric Framework for Large Language Model Agent Workflows},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/shibing624/agentica}
}
```

リポジトリのルートに [CITATION.cff](https://github.com/shibing624/agentica/blob/main/CITATION.cff) も用意しています。

## ライセンス

[Apache License 2.0](https://github.com/shibing624/agentica/blob/main/LICENSE)

## 貢献

貢献を歓迎します！[CONTRIBUTING.md](https://github.com/shibing624/agentica/blob/main/CONTRIBUTING.md) をご覧ください。

## 謝辞

- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [openai/openai-agents-python](https://github.com/openai/openai-agents-python)
