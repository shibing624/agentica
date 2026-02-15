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
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#コミュニティとサポート)

**Agentica** は軽量な Python フレームワークで、AIエージェントの構築に使用します。Async-First アーキテクチャで、ツール呼び出し、RAG、マルチエージェントチーム、ワークフローオーケストレーション、MCP プロトコルをサポートします。

## インストール

```bash
pip install -U agentica
```

## クイックスタート

```python
import asyncio
from agentica import Agent, ZhipuAI

async def main():
    agent = Agent(model=ZhipuAI())
    result = await agent.run("北京を一文で紹介してください")
    print(result.content)

asyncio.run(main())
```

```
北京は中国の首都であり、三千年以上の歴史を持つ文化都市で、政治・文化・国際交流の中心地です。
```

まず API キーを設定してください：

```bash
export ZHIPUAI_API_KEY="your-api-key"      # ZhipuAI（glm-4.7-flash は無料）
export OPENAI_API_KEY="sk-xxx"              # OpenAI
export DEEPSEEK_API_KEY="your-api-key"      # DeepSeek
```

## 機能

- **Async-First** — ネイティブ async API、`asyncio.gather()` による並列ツール実行、同期アダプター対応
- **20以上のモデル** — OpenAI / DeepSeek / Claude / ZhipuAI / Qwen / Moonshot / Ollama / LiteLLM など
- **40以上の組み込みツール** — 検索、コード実行、ファイル操作、ブラウザ、OCR、画像生成
- **RAG** — ナレッジベース管理、ハイブリッド検索、Rerank、LangChain / LlamaIndex 統合
- **マルチエージェント** — Team（動的委任）と Workflow（確定的オーケストレーション）
- **ガードレール** — 入力 / 出力 / ツールレベルのガードレール
- **MCP / ACP** — Model Context Protocol と Agent Communication Protocol のサポート
- **スキルシステム** — Markdown ベースのスキル注入、モデル非依存
- **マルチモーダル** — テキスト、画像、音声、動画の理解

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
| **基本** | Hello World、ストリーミング、構造化出力、マルチターン、マルチモーダル |
| **ツール** | カスタムツール、Async ツール、検索、コード実行、ファイル操作 |
| **エージェントパターン** | Agent-as-Tool、並列実行、チームコラボレーション、ディベート |
| **RAG** | PDF Q&A、高度な RAG、LangChain / LlamaIndex 統合 |
| **ワークフロー** | データパイプライン、投資リサーチ、ニュースレポート、コードレビュー |
| **MCP** | Stdio / SSE / HTTP トランスポート、JSON 設定 |
| **アプリケーション** | LLM OS、ディープリサーチ、カスタマーサービス |

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
