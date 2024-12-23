[**🇨🇳中文**](https://github.com/shibing624/agentica/blob/main/README.md) | [**🌐English**](https://github.com/shibing624/agentica/blob/main/README_EN.md) | [**🇯🇵日本語**](https://github.com/shibing624/agentica/blob/main/README_JP.md)

<div align="center">
  <a href="https://github.com/shibing624/agentica">
    <img src="https://raw.githubusercontent.com/shibing624/agentica/main/docs/logo.png" height="150" alt="Logo">
  </a>
</div>

-----------------

# Agentica: AIエージェントの構築
[![PyPI version](https://badge.fury.io/py/agentica.svg)](https://badge.fury.io/py/agentica)
[![Downloads](https://static.pepy.tech/badge/agentica)](https://pepy.tech/project/agentica)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.5%2B-green.svg)](requirements.txt)
[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#Contact)


**agentica**: 大規模言語モデルエージェントワークフローのための人間中心のフレームワーク、エージェントワークフローを迅速に構築

**agentica**: あなた自身のエージェントを迅速に構築

## 概要

#### LLMエージェント
![llm_agnet](https://github.com/shibing624/agentica/blob/main/docs/llm_agent.png)

- **計画（Planning）**：タスクの分解、計画の生成、反省
- **記憶（Memory）**：短期記憶（プロンプト実装）、長期記憶（RAG実装）
- **ツール使用（Tool use）**：function call能力、外部APIの呼び出し、外部情報の取得、現在の日付、カレンダー、コード実行能力、専用情報源へのアクセスなど

#### agenticaアーキテクチャ
![agentica_arch](https://github.com/shibing624/agentica/blob/main/docs/agent_arch.png)

- **Planner**：LLMが複雑なタスクを完了するための多段階計画を生成し、相互依存する「チェーン計画」を生成し、各ステップが前のステップの出力に依存することを定義
- **Worker**：チェーン計画を受け取り、計画内の各サブタスクをループで処理し、ツールを呼び出してタスクを完了し、自動的に反省して修正しタスクを完了
- **Solver**：すべての出力を統合して最終的な答えを提供

## 特徴
`agentica`はエージェントワークフロー構築ツールであり、以下の機能を提供：

- 簡単なコードで複雑なワークフローを迅速に編成
- ワークフローの編成はプロンプトコマンドだけでなく、ツール呼び出し（tool_calls）もサポート
- OpenAI APIおよびMoonshot API(kimi)の呼び出しをサポート

## インストール

```bash
pip install -U agentica
```

または

```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## はじめに

1. [example.env](https://github.com/shibing624/agentica/blob/main/example.env)ファイルをコピーして`.env`にし、OpenAI APIキーまたはMoonshoot APIキーを貼り付けます。
    ```shell
    export OPENAI_API_KEY=your_openai_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. `agentica`を使用してエージェントを構築し、タスクを分解して実行：

自動的にGoogle検索ツールを呼び出す例：[examples/web_search_demo.py](https://github.com/shibing624/agentica/blob/main/examples/web_search_demo.py)

```python
from agentica import Agent, OpenAIChat, SearchSerperTool

m = Agent(model=OpenAIChat(id='gpt-4o'), tools=[SearchSerperTool()], add_datetime_to_instructions=True)
r = m.run("Where will the next Olympics be held?")
print(r)
```


## 例

| 例                                                                                                                                  | 説明                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| [examples/naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/naive_rag_demo.py)                         | 基本的RAGを実装し、Txtドキュメントに基づいて質問に回答                                                                                                           |
| [examples/advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/advanced_rag_demo.py)                   | 高度なRAGを実装し、PDFドキュメントに基づいて質問に回答、新機能：pdfファイル解析、クエリの改訂、文字+意味の多重リコール、リコールの再ランク付け（rerank）                                                               |
| [examples/python_assistant_demo.py](https://github.com/shibing624/agentica/blob/main/examples/python_assistant_demo.py)           | Code Interpreter機能を実装し、自動的にpythonコードを生成して実行                                                                                          |
| [examples/research_demo.py](https://github.com/shibing624/agentica/blob/main/examples/research_demo.py)                           | Research機能を実装し、自動的に検索ツールを呼び出し、情報をまとめて科学レポートを作成                                                                                              |
| [examples/run_flow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/run_flow_news_article_demo.py) | ニュース記事の作成ワークフローを実装し、multi-agentの実装、複数のAssistantとTaskを定義し、検索ツールを複数回呼び出し、高度なレイアウトのニュース記事を生成                                                            |
| [examples/run_flow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/run_flow_investment_demo.py)     | 投資研究のワークフローを実装：株式情報収集 - 株式分析 - 分析レポート作成 - レポートの再確認など複数のTask                                                                                |
| [examples/crawl_webpage.py](https://github.com/shibing624/agentica/blob/main/examples/crawl_webpage.py)                           | ウェブページ分析ワークフローを実装：Urlから資金調達ニュースをクロール - ウェブページの内容と形式を分析 - 主要情報を抽出 - mdファイルとして保存                                                                          |
| [examples/find_paper_from_arxiv.py](https://github.com/shibing624/agentica/blob/main/examples/find_paper_from_arxiv.py)           | 論文推薦ワークフローを実装：arxivから複数の論文を自動検索 - 類似論文の重複を排除 - 主要論文情報を抽出 - csvファイルとして保存                                                                        |
| [examples/remove_image_background.py](https://github.com/shibing624/agentica/blob/main/examples/remove_image_background.py)       | 画像の背景を自動的に削除する機能を実装し、pipを通じてライブラリを自動的にインストールし、ライブラリを呼び出して画像の背景を削除                                                                                          |
| [examples/text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/text_classification_demo.py)     | 分類モデルの自動トレーニングワークフローを実装：トレーニングセットファイルを読み取り形式を理解 - Googleでpytextclassifierライブラリを検索 - githubページをクロールしてpytextclassifierの呼び出し方法を理解 - コードを書いてfasttextモデルをトレーニング - トレーニング済みモデルの予測結果をチェック |


## 連絡先

- Issue(提案)
  ：[![GitHub issues](https://img.shields.io/github/issues/shibing624/agentica.svg)](https://github.com/shibing624/agentica/issues)
- メール：xuming: xuming624@qq.com
- WeChat：*WeChat ID：xuming624, メモ：名前-会社-NLP* でNLPグループに参加。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 引用

研究で`agentica`を使用した場合は、以下の形式で引用してください：

APA:

```
Xu, M. agentica: A Human-Centric Framework for Large Language Model Agent Workflows (Version 0.0.2) [Computer software]. https://github.com/shibing624/agentica
```

BibTeX:

```
@misc{Xu_agentica,
  title={agentica: A Human-Centric Framework for Large Language Model Agent Workflows},
  author={Xu Ming},
  year={2024},
  howpublished={\url{https://github.com/shibing624/agentica}},
}
```

## ライセンス

ライセンスは [The Apache License 2.0](/LICENSE) であり、商用利用が無料です。製品説明に`agentica`のリンクとライセンスを追加してください。
## 貢献

プロジェクトのコードはまだ粗削りです。コードの改善がある場合は、このプロジェクトに戻して提出してください。提出前に以下の2点に注意してください：

- `tests`に対応する単体テストを追加
- `python -m pytest`を使用してすべての単体テストを実行し、すべてのテストが通過することを確認

その後、PRを提出できます。

## 謝辞 

- [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [https://github.com/simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
- [https://github.com/phidatahq/phidata](https://github.com/phidatahq/phidata)


彼らの素晴らしい仕事に感謝します！
