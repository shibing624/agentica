# 模型提供商

Agentica 支持 20+ 模型提供商，所有模型类遵循统一接口，可即插即用。

## 支持的模型

| 类名 | 提供商 | 默认模型 | 环境变量 |
|------|--------|---------|----------|
| `OpenAIChat` | OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| `DeepSeek` | DeepSeek | deepseek-chat | `DEEPSEEK_API_KEY` |
| `ZhipuAI` | 智谱 AI | glm-4.7-flash | `ZHIPUAI_API_KEY` |
| `Claude` | Anthropic | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| `Qwen` | 阿里通义 | qwen-plus | `DASHSCOPE_API_KEY` |
| `Moonshot` | Moonshot | moonshot-v1-128k | `MOONSHOT_API_KEY` |
| `Doubao` | 字节豆包 | doubao-pro-32k | `DOUBAO_API_KEY` |
| `KimiChat` | Kimi Coding | k2p5 | `KIMI_API_KEY` |
| `Yi` | 零一万物 | yi-lightning | `YI_API_KEY` |
| `XAI` | xAI | grok-3 | `XAI_API_KEY` |
| `Ollama` | Ollama (本地) | llama3.1 | — |
| `Together` | Together AI | — | `TOGETHER_API_KEY` |
| `LitellmChat` | LiteLLM | — | 按模型而定 |

## 基本用法

### 通过环境变量配置 API Key（推荐）

```bash
export OPENAI_API_KEY="sk-xxx"
export ZHIPUAI_API_KEY="your-key"
```

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o-mini"))
```

### 直接传入 API Key

```python
from agentica import Agent, DeepSeek

agent = Agent(model=DeepSeek(id="deepseek-chat", api_key="your-key"))
```

## 模型示例

### OpenAI

```python
from agentica import Agent, OpenAIChat

agent = Agent(model=OpenAIChat(id="gpt-4o"))
result = agent.run_sync("Hello!")
print(result.content)
```

### DeepSeek

```python
from agentica import Agent, DeepSeek

agent = Agent(model=DeepSeek(id="deepseek-chat"))
result = agent.run_sync("写一个快速排序")
print(result.content)
```

### 智谱 AI（免费模型）

```python
from agentica import Agent, ZhipuAI

# glm-4.7-flash 免费，支持工具调用，128k 上下文
agent = Agent(model=ZhipuAI(id="glm-4.7-flash"))
result = agent.run_sync("一句话介绍北京")
print(result.content)
```

### Claude (Anthropic)

```python
from agentica import Agent, Claude

agent = Agent(model=Claude(id="claude-sonnet-4-20250514"))
result = agent.run_sync("Explain quantum computing")
print(result.content)
```

### Kimi Coding

```python
from agentica import Agent, KimiChat

agent = Agent(model=KimiChat(id="k2p5"))
result = agent.run_sync("写一个二分查找")
print(result.content)
```

### Ollama（本地模型）

```python
from agentica import Agent, Ollama

agent = Agent(model=Ollama(id="llama3.1"))
result = agent.run_sync("What is Python?")
print(result.content)
```

### LiteLLM（统一接口）

通过 LiteLLM 可以用统一接口调用 100+ 模型：

```python
from agentica import Agent, LitellmChat

# 使用 Bedrock
agent = Agent(model=LitellmChat(id="bedrock/anthropic.claude-v2"))

# 使用 Azure OpenAI
agent = Agent(model=LitellmChat(id="azure/gpt-4"))
```

## 切换模型

所有模型类遵循统一接口，切换模型只需更改一行：

```python
from agentica import Agent, OpenAIChat, DeepSeek, ZhipuAI

# 开发阶段用免费模型
agent = Agent(model=ZhipuAI())

# 生产环境切换为更强模型
agent = Agent(model=OpenAIChat(id="gpt-4o"))

# 或使用 DeepSeek
agent = Agent(model=DeepSeek())
```
