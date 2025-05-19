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
[![License Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![python_version](https://img.shields.io/badge/Python-3.10%2B-green.svg)](requirements.txt)
[![MseeP.ai](https://img.shields.io/badge/mseep.ai-agentica-blue)](https://mseep.ai/app/shibing624-agentica)
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

1. [example.env](https://github.com/shibing624/agentica/blob/main/example.env)ファイルをコピーして`~/.agentica/.env`にし、OpenAI APIキーまたはOPENAI APIキーを貼り付けます。
    ```shell
    export OPENAI_API_KEY=your_openai_api_key
    export SERPER_API_KEY=your_serper_api_key
    ```

2. `agentica`を使用してエージェントを構築し、タスクを分解して実行：

自動的にGoogle検索ツールを呼び出す例：[examples/11_web_search_openai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/11_web_search_openai_demo.py)

```python
from agentica import Agent, OpenAIChat, SearchSerperTool

m = Agent(model=OpenAIChat(id='gpt-4o'), tools=[SearchSerperTool()], add_datetime_to_instructions=True)
r = m.run("Where will the next Olympics be held?")
print(r)
```


## 例

| 例                                                                                                                                                    | 説明                                                                                                                                                                                    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [examples/01_llm_demo.py](https://github.com/shibing624/agentica/blob/main/examples/01_llm_demo.py)                                                   | LLM Q&A デモ                                                                                                                                                                            |
| [examples/02_user_prompt_demo.py](https://github.com/shibing624/agentica/blob/main/examples/02_user_prompt_demo.py)                                   | カスタムユーザープロンプトデモ                                                                                                                                                                       |
| [examples/03_user_messages_demo.py](https://github.com/shibing624/agentica/blob/main/examples/03_user_messages_demo.py)                               | カスタム入力ユーザーメッセージデモ                                                                                                                                                                     |
| [examples/04_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/04_memory_demo.py)                                             | エージェントのメモリーデモ                                                                                                                                                                         |
| [examples/05_response_model_demo.py](https://github.com/shibing624/agentica/blob/main/examples/05_response_model_demo.py)                             | 指定された形式（pydanticのBaseModel）で応答するデモ                                                                                                                                                    |
| [examples/06_calc_with_csv_file_demo.py](https://github.com/shibing624/agentica/blob/main/examples/06_calc_with_csv_file_demo.py)                     | LLMがCSVファイルを読み込み、計算を実行して質問に答えるデモ                                                                                                                                                      |
| [examples/07_create_image_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/07_create_image_tool_demo.py)                       | 画像ツールを作成するデモ                                                                                                                                                                          |
| [examples/08_ocr_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/08_ocr_tool_demo.py)                                         | OCRツー��を実装するデモ                                                                                                                                                                        |
| [examples/09_remove_image_background_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/09_remove_image_background_tool_demo.py) | 画像の背景を自動的に削除する機能を実装するデモ。ライブラリを自動的にインストールし、画像の背景を削除するためにライブラリを呼び出す機能を含む                                                                                                                |
| [examples/10_vision_demo.py](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                             | ビジョン理解デモ                                                                                                                                                                              |
| [examples/11_web_search_openai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/11_web_search_openai_demo.py)                       | OpenAIのfunction callに基づくウェブ検索デモ                                                                                                                                                       |
| [examples/12_web_search_moonshot_demo.py](https://github.com/shibing624/agentica/blob/main/examples/12_web_search_moonshot_demo.py)                   | Moonshotのfunction callに基づくウェブ検索デモ                                                                                                                                                     |
| [examples/13_storage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/13_storage_demo.py)                                           | エージェントのストレージデモ                                                                                                                                                                        |
| [examples/14_custom_tool_demo.py](https://github.com/shibing624/agentica/blob/main/examples/14_custom_tool_demo.py)                                   | カスタムツールを実装し、大規模モデルが自律的に選択して呼び出すデモ                                                                                                                                                     |
| [examples/15_crawl_webpage_demo.py](https://github.com/shibing624/agentica/blob/main/examples/15_crawl_webpage_demo.py)                               | ウェブページ分析ワークフローを実装するデモ：URLから資金調達ニュースをクロールし、ウェブページの内容と形式を分析し、主要情報を抽出し、mdファイルとして保存する                                                                                                     |
| [examples/16_get_top_papers_demo.py](https://github.com/shibing624/agentica/blob/main/examples/16_get_top_papers_demo.py)                             | 毎日の論文を解析し、JSON形式で保存するデモ                                                                                                                                                               |
| [examples/17_find_paper_from_arxiv_demo.py](https://github.com/shibing624/agentica/blob/main/examples/17_find_paper_from_arxiv_demo.py)               | 論文推薦のデモ：arxivから複数の論文を自動検索し、類似論文を重複排除し、主要論文情報を抽出し、CSVファイルとして保存する                                                                                                                       |
| [examples/18_agent_input_is_list.py](https://github.com/shibing624/agentica/blob/main/examples/18_agent_input_is_list.py)                             | エージェントのメッセージがリストであることを示すデモ                                                                                                                                                            |
| [examples/19_naive_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/19_naive_rag_demo.py)                                       | 基本的なRAGを実装し、テキストドキュメントに基づいて質問に答えるデモ                                                                                                                                                   |
| [examples/20_advanced_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | 高度なRAGを実装し、PDFドキュメントに基づいて質問に答えるデモ。新機能：PDFファイル解析、クエリの改訂、文字+意味の多重リコール、リコールの再ランク付け（rerank）                                                                                               |
| [examples/21_memorydb_rag_demo.py](https://github.com/shibing624/agentica/blob/main/examples/21_reference_in_prompt_rag_demo.py)                      | プロンプトに参考資料を含める従来のRAGのデモ                                                                                                                                                               |
| [examples/22_chat_pdf_app_demo.py](https://github.com/shibing624/agentica/blob/main/examples/22_chat_pdf_app_demo.py)                                 | PDFドキュメントとの深い対話のデモ                                                                                                                                                                    |
| [examples/23_python_agent_memory_demo.py](https://github.com/shibing624/agentica/blob/main/examples/23_python_agent_memory_demo.py)                   | メモリを持つコードインタープリタ機能を実装し、Pythonコードを自動生成して実行し、次回の実行時にメモリから結果を取得するデモ                                                                                                                      |
| [examples/24_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/24_context_demo.py)                                           | コンテキストを持つ対話のデモ                                                                                                                                                                        |
| [examples/25_tools_with_context_demo.py](https://github.com/shibing624/agentica/blob/main/examples/25_tools_with_context_demo.py)                     | コンテキストパラメータを持つツールのデモ                                                                                                                                                                  |
| [examples/26_complex_translate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/26_complex_translate_demo.py)                       | 複雑な翻訳のデモ                                                                                                                                                                              |
| [examples/27_research_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/27_research_agent_demo.py)                             | Research機能を実装し、検索ツールを自動的に呼��出し、情報をまとめて科学レポートを作成するデモ                                                                                                                                   |
| [examples/28_rag_integrated_langchain_demo.py](https://github.com/shibing624/agentica/blob/main/examples/28_rag_integrated_langchain_demo.py)         | LangChainと統合されたRAGデモ                                                                                                                                                                  |
| [examples/29_rag_integrated_llamaindex_demo.py](https://github.com/shibing624/agentica/blob/main/examples/29_rag_integrated_llamaindex_demo.py)       | LlamaIndexと統合されたRAGデモ                                                                                                                                                                 |
| [examples/30_text_classification_demo.py](https://github.com/shibing624/agentica/blob/main/examples/30_text_classification_demo.py)                   | 分類モデルを自動的にトレーニングするエージェントのデモ：トレーニングセットファイルを読み取り形式を理解し、Googleでpytextclassifierライブラリを検索し、GitHubページをクロールしてpytextclassifierの呼び出し方法を理解し、コードを書いてfasttextモデルをトレーニングし、トレーニング済みモデルの予測結果をチェックす�� |
| [examples/31_team_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | Team実装：ニュース記事を書くた���のチーム協力、マルチロール実装、各自のタスクを完了するために異なる役割を委任：研究��が記事を検索して分析し、ライターがレイアウトに従って記事を書き、複数の役割の成果をまとめる                                                                          |
| [examples/32_team_debate_demo.py](https://github.com/shibing624/agentica/blob/main/examples/32_team_debate_demo.py)                                   | Team実装：委任に基づく二人の討論デモ、トランプとバイデンの討論                                                                                                                                                     |
| [examples/33_self_evolving_agent_demo.py](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | 自己進化エージェントのデモ                                                                                                                                                                         |
| [examples/34_llm_os_demo.py](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | LLM OSの初期設計、LLM設計のオペレーティングシステムに基づき、LLMを通じてRAG、コードエグゼキュータ、シェルなどを呼び出し、コードインタープリタ、研究アシスタント、投資アシスタントなどと協力して問題を解決する。                                                                       |
| [examples/35_workflow_investment_demo.py](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                   | 投資研究のワークフローを実装：株式情報収集、株式分析、分析レポート作成、レポートの再確認など複数のタスク                                                                                                                                  |
| [examples/36_workflow_news_article_demo.py](https://github.com/shibing624/agentica/blob/main/examples/36_workflow_news_article_demo.py)               | ニュース記事を書くためのワークフローを実装、マルチエージェント実装、検索ツールを複数回呼び出し、高度なレイアウトのニュース記事を生成                                                                                                                    |
| [examples/37_workflow_write_novel_demo.py](https://github.com/shibing624/agentica/blob/main/examples/37_workflow_write_novel_demo.py)                 | 小説を書くためのワークフローを実装：小説のアウトラインを設定し、Googleでアウトラインを反映し、小説の内容を書き、mdファイルとして保存する                                                                                                              |
| [examples/38_workflow_write_tutorial_demo.py](https://github.com/shibing624/agentica/blob/main/examples/38_workflow_write_tutorial_demo.py)           | 技術チュートリアルを書くためのワークフローを実装：チュートリアルディレクトリを設定し、ディレクトリ内容を反映し、チュートリアル内容を書き、mdファイルとして保存する                                                                                                    |
| [examples/39_audio_multi_turn_demo.py](https://github.com/shibing624/agentica/blob/main/examples/39_audio_multi_turn_demo.py)                         | OpenAIの音声APIに基づくマルチターン音声対話のデモ                                                                                                                                                         |
| [examples/40_weather_zhipuai_demo.py](https://github.com/shibing624/agentica/blob/main/examples/40_web_search_zhipuai_demo.py)                        | 基于智谱AI的api做天气查询的Demo                                                                                                              |
| [examples/41_mcp_stdio_demo.py](https://github.com/shibing624/agentica/blob/main/examples/41_mcp_stdio_demo.py)                                       | Stdio的MCP Server调用的Demo                                                                                                           |
| [examples/42_mcp_sse_server.py](https://github.com/shibing624/agentica/blob/main/examples/42_mcp_sse_server.py)                                       | SSE的MCP Server调用的Demo                                                                                                             |
| [examples/42_mcp_sse_client.py](https://github.com/shibing624/agentica/blob/main/examples/42_mcp_sse_client.py)                                       | SSE的MCP Client调用的Demo                                                                                                             |
| [examples/43_minimax_mcp_demo.py](https://github.com/shibing624/agentica/blob/main/examples/43_minimax_mcp_demo.py)                                   | Minimax语音生成调用的Demo                                                                                                                |
| [examples/44_mcp_streamable_http_server.py](https://github.com/shibing624/agentica/blob/main/examples/44_mcp_streamable_http_server.py)                           | Streamable Http的MCP Server调用的Demo                                                                                                 |
| [examples/44_mcp_streamable_http_client.py](https://github.com/shibing624/agentica/blob/main/examples/44_mcp_streamable_http_client.py)                           | Streamable Http的MCP Client调用的Demo                                                                                                 |

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
