# Deep Research Agent Evaluation

本目录包含用于评测 Agentica Agent 多轮深度搜索研究能力的脚本和数据集。

## 概述

评测基于 `enable_multi_round=True` 的多轮策略，Agent 会自动进行多轮搜索、访问网页、分析信息，直到找到答案。

### 多轮策略工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Round Strategy                      │
├─────────────────────────────────────────────────────────────┤
│  1. 用户提问                                                  │
│  2. Agent 调用搜索工具获取相关网页                              │
│  3. Agent 访问网页获取详细信息                                  │
│  4. Agent 分析信息，判断是否需要继续搜索                         │
│  5. 重复 2-4 直到无工具调用（任务完成）                          │
│  6. 返回最终答案                                               │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

```bash
# 基础评测（3个样本）
python run.py --model gpt-4o --dataset browsecomp_zh_small --eval_n_limit 3

# 完整评测
python run.py --model gpt-4o --dataset browsecomp_zh_small --eval_n_limit 0

# 使用不同模型
python run.py --model deepseek-reasoner --dataset browsecomp_zh_small

# 调整多轮参数
python run.py --model gpt-4o --max_rounds 100 --max_tokens 128000
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | `gpt-4o` | 模型 ID |
| `--dataset` | str | `browsecomp_zh_small` | 评测数据集 |
| `--eval_n_limit` | int | `3` | 评测样本数（0=全部） |
| `--max_rounds` | int | `20` | 最大轮次 |
| `--max_tokens` | int | `40000` | Token 上限 |
| `--tools` | str | `baidu` | 搜索工具（baidu/serper/jina/all） |
| `--debug` | int | `0` | 调试模式（0=关，1=开） |
| `--output_dir` | str | `outputs` | 输出目录 |
| `--skip_judge` | flag | - | 跳过 LLM 评判 |

## 支持的数据集

| 数据集 | 描述 | 样本数 |
|--------|------|--------|
| `browsecomp_zh_small` | BrowseComp 中文小规模 | ~10 |
| `browsecomp_zh` | BrowseComp 中文完整 | ~100 |
| `browsecomp_en_small` | BrowseComp 英文小规模 | ~10 |
| `browsecomp_en` | BrowseComp 英文完整 | ~100 |
| `simple_qa` | SimpleQA 问答 | - |
| `gaia_2023_all_validation` | GAIA 2023 验证集 | ~165 |
| `xbench_deepsearch` | XBench 深度搜索 | - |
| `sailorfog-QA` | SailorFog QA | - |

## 输出文件

评测完成后会在 `outputs/` 目录生成：

1. **predictions-{dataset}.jsonl** - 预测结果
   ```json
   {
     "question": "问题内容",
     "answer": "标准答案",
     "prediction": "模型预测",
     "messages": [...],
     "tool_calls": [...],
     "full_response": "完整响应"
   }
   ```

2. **summary-{dataset}.json** - 评测摘要
   ```json
   {
     "dataset": "browsecomp_zh_small",
     "model": "gpt-4o",
     "accuracy": 66.67,
     "correct": 2,
     "total": 3,
     "statistics": {
       "avg_tool_calls": 8.5,
       "avg_rounds": 4.2,
       ...
     }
   }
   ```

## 评测指标

- **Accuracy**: 正确率（由 LLM Judge 判断）
- **Avg Tool Calls**: 平均工具调用次数
- **Avg Rounds**: 平均对话轮次
- **Avg Answer Length**: 平均答案长度
- **Avg Reasoning Length**: 平均推理长度

## 搜索工具配置

| 配置 | 工具 | 说明 |
|------|------|------|
| `baidu` | BaiduSearchTool + UrlCrawlerTool | 百度搜索（中文推荐） |
| `serper` | SearchSerperTool + UrlCrawlerTool | Serper API（需要 API Key） |
| `jina` | JinaTool + UrlCrawlerTool | Jina AI（需要 API Key） |
| `all` | 全部工具 | 所有搜索工具 |

## 示例输出

```
============================================================
📊 EVALUATION RESULTS
============================================================
Dataset: browsecomp_zh_small
Model: gpt-4o
Instances: 3
----------------------------------------
✅ Accuracy: 66.67% (2/3)
📈 Avg Tool Calls: 8.5
   - Search: 3.2
   - Visit: 5.3
   - Other: 0.0
📝 Avg Rounds: 4.2 (max: 8)
📄 Avg Answer Length: 156
🧠 Avg Reasoning Length: 2340
============================================================
```

## 目录结构

```
evaluation/
├── README.md           # 本文档
├── run.py              # 评测脚本
├── prompt.py           # Judge 提示词
├── data/               # 评测数据集
│   ├── browsecomp_zh_small.jsonl
│   ├── browsecomp_zh.jsonl
│   ├── browsecomp_en_small.jsonl
│   ├── gaia_2023_all_validation.jsonl
│   └── ...
└── outputs/            # 评测结果
    ├── predictions-*.jsonl
    └── summary-*.json
```

## 相关文档

- [Multi-Round Deep Research Agent](../docs/multi_round_deep_research_agent.md) - 多轮策略实现原理
