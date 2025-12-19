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
[![Wechat Group](https://img.shields.io/badge/wechat-group-green.svg?logo=wechat)](#%E3%82%B3%E3%83%9F%E3%83%A5%E3%83%8B%E3%83%86%E3%82%A3%E3%81%A8%E3%82%B5%E3%83%9D%E3%83%BC%E3%83%88)

**Agenticaは、自律型AIエージェントを構築、管理、展開するための軽量で強力なPythonフレームワークです。**

シンプルなチャットボット、複雑なリサーチアシスタント、または専門エージェントの協力チームを作成する場合でも、Agenticaは目標をより速く達成するためのツールと抽象化を提供します。私たちの開発者第一のアプローチは、RAG、マルチエージェントワークフロー、長期記憶などの高度な機能を簡素化し、誰もが利用できるようにします。

## 🚀 なぜAgenticaを選ぶのか？

*   **開発者第一のAPI**：シンプルで直感的、オブジェクト指向のインターフェースで、習得が容易で使いやすい。
*   **モジュール式で拡張可能**：LLM、メモリバックエンド、ベクトルストアなどのコンポーネントを簡単に交換可能。
*   **豊富な機能**：豊富な組み込みツール（Web検索、コードインタプリタ、ファイルI/O）、メモリタイプ、高度なRAG機能を標準で提供。
*   **高度な機能を簡素化**：マルチエージェントコラボレーション（チーム）、タスク分解（ワークフロー）、自己反省などの複雑なパターンを簡単に実装。
*   **本番環境に対応**：コマンドラインインターフェース、Web UI、またはサービスとしてエージェントを展開。**モデルコンテキストプロトコル（MCP）**による標準化されたツール統合もサポート。
*   **Agent Skillサポート**：プロンプトベースのスキルシステムで、スキル指示をSystem Promptに注入し、tool callingをサポートする任意のモデルで使用可能。

## ✨ 主な機能

*   **🤖 コアエージェント機能**：高度な計画、反省、短期および長期記憶、堅牢なツール使用能力を持つエージェントを構築。
*   **🧩 高度なオーケストレーション**：
    *   **マルチエージェントチーム**：問題を解決するために協力する専門エージェントのチームを作成。
    *   **ワークフロー**：複雑なタスクを異なるエージェントやツールが実行する一連のステップに分解。
*   **🛡️ ガードレール（Guardrails）**：
    *   **入出力ガードレール**：エージェント処理前にユーザー入力を検証し、返却前にエージェント出力をチェック。
    *   **ツールガードレール**：実行前にツール引数を検証し、結果から機密データをフィルタリング。
    *   **3つの動作モード**：allow（許可）、reject_content（拒否して続行）、raise_exception（実行停止）。
*   **🎯 Agent Skillシステム**：
    *   **Prompt Engineering技術**：Skillはテキスト指示であり、コードレベルの能力拡張ではありません。
    *   **実装方法**：SKILL.mdのメタデータを解析し、スキル説明をSystem Promptに注入。
    *   **実行フロー**：LLMがスキル説明を読んだ後、基本ツール（shell、python、file viewer）を使用してタスクを実行。
    *   **モデル非依存**：tool callingをサポートする任意のモデルがスキルを使用可能（スキルはテキスト指示のため）。
    *   **利点**：拡張可能、モデル非依存、メンテナンスが容易（Markdownドキュメントを更新するだけ）。
*   **🛠️ 豊富なツールエコシステム**：
    *   広範な組み込みツール（Web検索、OCR、画像生成、シェルコマンド）。
    *   独自のカスタムツールを簡単に作成。
    *   標準化されたツール統合のための**モデルコンテキストプロトコル（MCP）**の第一級サポート。
*   **📚 柔軟なRAGパイプライン**：
    *   組み込みの知識ベース管理とドキュメント解析（PDF、テキスト）。
    *   最高の精度を実現するためのハイブリッド検索戦略と結果の再ランキング。
    *   LangChainやLlamaIndexなどの人気ライブラリとの統合。
*   **🌌 マルチモーダルサポート**：テキスト、画像、音声、ビデオを理解し生成できるエージェントを構築。
*   **🧠 幅広いLLM互換性**：OpenAI、Azure、Deepseek、Moonshot、Anthropic、ZhipuAI、Ollama、Togetherなどのプロバイダーからの数十のモデルと連携。
*   **💡 自己進化エージェント**：自己反省と記憶増強能力を持ち、自己進化するエージェント。

## 🏗️ システムアーキテクチャ

<div align="center">
    <img src="https://github.com/shibing624/agentica/blob/main/docs/agentica_architecture.png" alt="Agentica Architecture" width="800"/>
</div>

Agenticaのモジュール設計は、最大限の柔軟性とスケーラビリティを可能にします。その中心には `Agent`、`Model`、`Tool`、`Memory` コンポーネントがあり、これらを簡単に組み合わせて拡張することで、強力なアプリケーションを作成できます。

## 💾 インストール

```bash
pip install -U agentica
```

ソースからインストールする場合：
```bash
git clone https://github.com/shibing624/agentica.git
cd agentica
pip install .
```

## ⚡ クイックスタート

1.  **APIキーを設定します。** `~/.agentica/.env` にファイルを作成するか、環境変数を設定します。

    ```shell
    # ZhipuAI用
    export ZHIPUAI_API_KEY="your-api-key"
    ```

2.  **最初のエージェントを実行しましょう！** この例では、天気をチェックできるエージェントを作成します。

```python
    from agentica import Agent, ZhipuAI, WeatherTool

    # モデルと天気ツールでエージェントを初期化
    agent = Agent(
        model=ZhipuAI(), 
        tools=[WeatherTool()],
        # 「明日」のような質問のためにエージェントに時間感覚を与える
        add_datetime_to_instructions=True  
    )

    # エージェントに質問する
    agent.print_response("明日の北京の天気はどうですか？")
    ```

    **出力：**
    ```markdown
    明日の北京の天気予報は以下の通りです：

    - 朝：晴れ、気温約18℃、風速3km/hの弱風。
    - 昼：晴れ、気温は23℃に上昇、風速6-7km/h。
    - 夕方：晴れ、気温はわずかに21℃に低下、風速35-44km/hの強風。
    - 夜：晴れ、気温は15℃に低下、風速32-39km/h。

    一日を通して降水はなく、視界は良好です。夕方の強風にご注意ください。
    ```

## 📖 コアコンセプト

*   **Agent**：思考し、決定を下し、行動を実行する中心的なコンポーネント。モデル、ツール、メモリを結びつけます。
*   **Model**：エージェントの「脳」。通常はエージェントの推論能力を支える大規模言語モデル（LLM）です。
*   **Tool**：エージェントが外部世界と対話するために使用できる機能や能力（例：Web検索、コード実行、データベースアクセス）。
*   **Memory**：エージェントが過去の対話を記憶する（短期）および重要な情報を後で思い出すために保存する（長期）ことを可能にします。
*   **Knowledge**：エージェントが検索拡張生成（RAG）を使用してクエリできる外部の知識源（ドキュメントのコレクションなど）。
*   **Workflow/Team**：複雑な多段階のタスクを調整したり、複数のエージェント間の協力を管理したりするための高レベルの構成要素。

## 🚀 ショーケース：構築可能なもの

Agenticaで何が可能か、包括的な例をご覧ください：

| 例                                                                                                                                                    | 説明                                                                                                                                |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| [**高度なRAGエージェント**](https://github.com/shibing624/agentica/blob/main/examples/20_advanced_rag_demo.py)                                 | クエリ書き換え、ハイブリッド検索、再ランキング機能を備えたPDFドキュメント上の強力なQ&Aシステムを構築。                                                               |
| [**マルチエージェントチーム**](https://github.com/shibing624/agentica/blob/main/examples/31_team_news_article_demo.py)                       | ニュース記事の執筆に協力するためにエージェントのチーム（例：研究者とライター）を編成。                                                     |
| [**自己進化エージェント**](https://github.com/shibing624/agentica/blob/main/examples/33_self_evolving_agent_demo.py)                   | 対話から学び、時間とともに知識ベースを向上させるエージェントを作成。                                                                                                                 |
| [**LLM OS**](https://github.com/shibing624/agentica/blob/main/examples/34_llm_os_demo.py)                                             | LLMを搭載した対話型オペレーティングシステムを構築するという魅力的な実験。                                                  |
| [**投資調査ワークフロー**](https://github.com/shibing624/agentica/blob/main/examples/35_workflow_investment_demo.py)                   | データ収集と分析からレポート生成まで、投資調査プロセス全体を自動化。                                                                                                   |
| [**ビジョンエージェント**](https://github.com/shibing624/agentica/blob/main/examples/10_vision_demo.py)                                             | 画像を理解し推論できるエージェントを構築。                                                                                                                          |
| [**ガードレール**](https://github.com/shibing624/agentica/blob/main/examples/52_guardrails_demo.py)                                             | 入出力ガードレールを使用してエージェントとツールのI/Oを検証し、機密データをフィルタリングする方法をデモ。                                                                                                                          |

[➡️ **すべての例を見る**](https://github.com/shibing624/agentica/tree/main/examples)

## 🖥️ 展開

### コマンドラインインターフェース (CLI)

ターミナルから直接エージェントと対話します。

```shell
# agenticaをインストール
pip install -U agentica

# 単一のクエリを実行
agentica --query "次のオリンピックはどこで開催されますか？" --model_provider zhipuai --model_name glm-4.6v-flash --tools baidu_search

# 対話型チャットセッションを開始
agentica --model_provider zhipuai --model_name glm-4.6v-flash
```

CLI show case (Like ClaudeCode):
<img src="https://github.com/shibing624/agentica/blob/main/docs/cli_snap.png" width="800" />

### Web UI

Agenticaは[ChatPilot](https://github.com/shibing624/ChatPilot)と完全に互換性があり、エージェント用の機能豊富なGradioベースのWebインターフェースを提供します。

<div align="center">
    <img src="https://github.com/shibing624/ChatPilot/blob/main/docs/shot.png" width="800" />
</div>

設定手順については、[ChatPilotリポジトリ](https://github.com/shibing624/ChatPilot)をご覧ください。

## 🤝 他のフレームワークとの比較

| 機能                | Agentica                                   | LangChain                                 | AutoGen                             | CrewAI                             |
|------------------------|--------------------------------------------|-------------------------------------------|-------------------------------------|------------------------------------|
| **コア設計**        | エージェント中心、モジュール式、直感的      | チェーン中心、複雑なコンポーネントグラフ    | マルチエージェント対話に焦点    | 役割ベースのマルチエージェントに焦点       |
| **使いやすさ**        | 高（シンプルさを追求）             | 中（学習曲線が急）           | 中                            | 高                               |
| **マルチエージェント**        | ネイティブな`Team`と`Workflow`サポート         | カスタム実装が必要            | コア機能                        | コア機能                       |
| **RAG**                | 組み込みの高度なパイプライン                | コンポーネントの手動組み立てが必要    | 外部統合が必要       | 外部統合が必要      |
| **ツール**            | 豊富な組み込みツール + MCPサポート          | 大規模なエコシステム、複雑な場合がある         | 基本的なツールサポート                  | 基本的なツールサポート                 |
| **マルチモーダル**        | ✅ 対応（テキスト、画像、音声、ビデオ）         | ✅ 対応（ただし統合が複雑な場合がある）  | ❌ 非対応（主にテキストベース）      | ❌ 非対応（主にテキストベース）     |


## 💬 コミュニティとサポート

*   **GitHub Issues**：質問や機能リクエストがありますか？[issueを開いてください](https://github.com/shibing624/agentica/issues)。
*   **WeChat**：開発者コミュニティに参加しましょう！WeChatで`xuming624`を追加し、「agentica」と伝えてグループに追加してもらってください。

<img src="https://github.com/shibing624/agentica/blob/main/docs/wechat.jpeg" width="200" />

## 📜 引用

研究でAgenticaを使用する場合は、以下の形式で引用してください：

```bibtex
@misc{agentica,
  author = {Ming Xu},
  title = {Agentica: Effortlessly Build Intelligent, Reflective, and Collaborative Multimodal AI Agents},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/shibing624/agentica}},
}
```

## 📄 ライセンス

Agenticaは[Apache License 2.0](LICENSE)の下でライセンスされています。

## ❤️ 貢献

あらゆる種類の貢献を歓迎します！始めるには[貢献ガイドライン](CONTRIBUTING.md)をご覧ください。

## 🙏 謝辞

私たちの仕事は、偉大なプロジェクトの肩の上に成り立っています。以下の素晴らしいプロジェクトのチームに感謝します：
- [langchain-ai/langchain](https://github.com/langchain-ai/langchain)
- [phidatahq/phidata](https://github.com/phidatahq/phidata)
- [simonmesmith/agentflow](https://github.com/simonmesmith/agentflow)
