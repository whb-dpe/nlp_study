# 多代理并行工作系统 (Multi-Agent Parallel System)

一个基于 DeepSeek 大模型的多 Agent 协作系统，支持主 Agent (Orchestrator) 将复杂任务拆解为多个子任务，分配给多个 SubAgent **并行执行**，最后汇总结果。

## 架构概览

```
┌───────────────────────────────────────────────────────────┐
│                     用户 (User)                            │
│                       │                                    │
│                       ▼                                    │
│           ┌─────────────────────┐                         │
│           │   Orchestrator      │  主调度Agent             │
│           │  (任务拆解/调度/汇总)│  - LLM 拆解任务          │
│           └─────────┬───────────┘  - 并发控制              │
│                     │              - 结果整合              │
│     ┌───────────────┼───────────────┐                      │
│     ▼               ▼               ▼                      │
│ ┌─────────┐   ┌─────────┐   ┌─────────┐  并行执行          │
│ │SubAgent │   │SubAgent │   │SubAgent │  - Researcher      │
│ │Research.│   │Analyst  │   │Coder    │  - Analyst         │
│ └────┬────┘   └────┬────┘   └────┬────┘  - Writer          │
│      │             │             │        - Coder 等        │
│      └─────────────┼─────────────┘                         │
│                    ▼                                       │
│           ┌─────────────────────┐                         │
│           │   结果汇总 & 输出    │                         │
│           └─────────────────────┘                         │
└───────────────────────────────────────────────────────────┘
```

## 模块说明

| 文件 | 说明 |
|------|------|
| `config.py` | 配置管理（API Key、模型参数、并发数等） |
| `llm_client.py` | DeepSeek API 异步客户端封装（aiohttp） |
| `task.py` | 任务数据结构与状态管理 (Task/TaskStatus) |
| `base_agent.py` | Agent 抽象基类，封装通用 LLM 调用能力 |
| `sub_agent.py` | 子 Agent，内置多种角色 (研究员/分析师/程序员/写手等) |
| `orchestrator.py` | **主调度 Agent**：任务拆解 → 并行调度 → 结果汇总 |
| `main.py` | 使用示例（内置3种场景 + 交互模式） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```bash
cp .env.example .env
# 然后编辑 .env
```

`.env` 文件内容：
```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
MAX_CONCURRENT_AGENTS=5
```

### 3. 运行

```bash
python main.py
```

然后选择运行模式：
- **1** - 示例1：自动任务拆解（LLM调研）
- **2** - 示例2：自定义子任务（Python学习）
- **3** - 交互式模式（自己输入任务）
- **4** - 全部示例

## 使用方式

### 方式一：自动任务拆解（推荐）

Orchestrator 会用 LLM 自动分析目标，拆解成可并行子任务：

```python
import asyncio
from orchestrator import Orchestrator

async def main():
    async with Orchestrator(max_concurrent=3) as agent:
        result = await agent.run("写一份关于新能源汽车行业的研究报告")
        print(result["final_result"])

asyncio.run(main())
```

### 方式二：手动指定子任务

精确控制每个子任务的描述和角色：

```python
subtasks = [
    {"description": "调研新能源汽车市场规模", "role": "researcher"},
    {"description": "分析头部企业竞争格局", "role": "analyst"},
    {"description": "撰写市场分析报告", "role": "writer"},
]

result = await agent.run(
    user_objective="新能源汽车行业研究",
    custom_subtasks=subtasks,
)
```

### SubAgent 可用角色

| 角色 | 适用场景 |
|------|----------|
| `researcher` | 信息调研、资料收集、数据查找 |
| `analyst` | 数据分析、对比分析、逻辑推理 |
| `writer` | 撰写报告、文档、文案 |
| `coder` | 代码编写、技术方案设计 |
| `reviewer` | 审校内容、发现问题、提出建议 |
| `planner` | 制定计划、任务拆解、时间安排 |
| `translator` | 多语言翻译 |
| `general` | 通用任务（默认） |

## 核心特性

1. **异步并行执行**：基于 `asyncio` + `aiohttp`，真正的并发 I/O
2. **并发数控制**：通过 `Semaphore` 限制最大并行 Agent 数，避免 API 限流
3. **LLM 任务拆解**：用大模型自动分析任务并拆解成最优子任务集
4. **结果智能汇总**：汇总阶段再次调用 LLM 对多子任务结果进行整合、去重、梳理
5. **状态管理**：每个任务有完整生命周期 (PENDING → RUNNING → COMPLETED/FAILED)
6. **耗时统计**：记录每个子任务的开始/结束时间，便于性能分析
7. **灵活扩展**：可自定义角色、自定义子任务、自定义回调