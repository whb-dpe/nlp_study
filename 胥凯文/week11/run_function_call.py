"""
run_function_call.py — 方式一：Function Call（模型原生函数调用）

教学重点：
  1. 手写 JSON Schema：每个工具的 name/description/parameters 都要开发者自己写
     ——这是 Function Call 的"接入成本"，schema 写得越清楚，模型调用越准
  2. 单轮闭环三步：模型输出 tool_call → 宿主执行工具 → 结果以 role=tool 回填 → 模型生成最终回答
  3. 并行工具调用：模型一次输出多个 tool_call（如同时查年报+查天气），宿主逐个执行后一并回填
  4. 工具名 → 后端函数的 dispatch 表：业务逻辑（src/）与协议层（本文件）彻底分离

使用方式：
  # 配置环境变量
  #   Windows:  set DEEPSEEK_API_KEY=sk-xxx & set DASHSCOPE_API_KEY=sk-xxx
  #   Linux:    export DEEPSEEK_API_KEY=sk-xxx; export DASHSCOPE_API_KEY=sk-xxx

  # 单个问题
  python mode_function_call/run_function_call.py --question "宁德时代2023年营收和净利润？"

  # 内置示例问题（演示并行工具调用）
  python mode_function_call/run_function_call.py --demo

依赖：
  pip install openai
  环境变量：DASHSCOPE_API_KEY（Embedding，rag_backend 内部用）
            DEEPSEEK_API_KEY（默认 LLM；可在 --provider dashscope 切到 qwen-plus）

与其它方式的关系：
  本文件的 LLM 单轮循环代码，和 mode_mcp/run_mcp.py、mode_cli/run_cli.py 几乎一样，
  差异只在"工具从哪来"和"调用怎么执行"——这正是三者对比的教学点。
"""

import json
import os
import sys
import time
from pathlib import Path

from openai import OpenAI

from weather_backend import get_coordinates, get_weather_by_coords  # noqa: E402

# ── LLM 配置 ───────────────────────────────────────────────────────────────

PROVIDERS = {
    "deepseek": {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",  # 即 deepseek-v4-flash
    },
}


def build_client(provider: str):
    cfg = PROVIDERS[provider]
    if not cfg["api_key"]:
        print(f"错误：未设置 {provider.upper()}_API_KEY", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"]), cfg["model"]


# ── 【教学时刻 1】：手写工具的 JSON Schema ──────────────────────────────────
# Function Call 的核心接入成本：每个工具的参数 schema 必须开发者手写。
# description 直接决定模型"什么时候调这个工具、传什么参数"——写得越具体越准。

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": "查询指定城市的经纬度坐标。输入城市中文名，返回该城市的纬度和经度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市中文名，如 '宁德'、'北京'、'上海'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_by_coords",
            "description": "通过经纬度查询当前天气及未来3天预报。需要先通过 get_coordinates 获取纬度和经度。",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number", "description": "纬度，例如 31.23"},
                    "longitude": {"type": "number", "description": "经度，例如 121.47"},
                    "location_name": {"type": "string", "description": "可选，城市名称，用于显示"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]

# ── 【教学时刻 2】：工具名 → 后端函数的 dispatch 表 ─────────────────────────
# 业务逻辑在 src/，本文件只负责"协议层"——把模型生成的 tool_call 派发给后端函数。
# 新增工具只需：1) 在上面写 schema；2) 在这里加一行映射。这是 Function Call 的扩展方式。

TOOL_DISPATCH = {
    "get_coordinates": get_coordinates,
    "get_weather_by_coords": get_weather_by_coords,
}


# ── 单轮闭环 ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一名天气查询助手。你有两个工具可以使用：\n"
    "1. get_coordinates(city): 查询城市的经纬度坐标。\n"
    "2. get_weather_by_coords(latitude, longitude, location_name): 通过经纬度查询天气。\n\n"
    "当用户询问经纬度时，只需调用 get_coordinates 并返回结果。\n"
    "当用户询问天气时，必须先调用 get_coordinates 获取经纬度，然后调用 get_weather_by_coords 查询天气，最后返回结果。\n"
    "每轮可以调用多个工具。只依据工具返回的结果作答，不要编造数据。"
)


def run(client, model: str, question: str, verbose: bool = True) -> dict:
    """
    多轮闭环：提问 → 模型输出 tool_call → 执行 → 回填 → 再请求（可能继续调用工具）→ 最终回答。
    使用 while 循环，允许模型在一次对话中调用多个工具。
    返回 {answer, tool_calls, elapsed} 用于对比器汇总。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    t0 = time.time()
    tool_call_log = []
    max_rounds = 5  # 防止无限循环

    # 【核心改动】：使用 for/while 循环，支持模型多轮调用工具
    for round_num in range(max_rounds):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        # 如果模型没有工具调用请求，说明已有最终答案
        if not msg.tool_calls:
            if verbose and round_num > 0:
                print(f"  → [round {round_num}] 模型已收集足够信息，生成最终回答")
            break

        # 有工具调用：把 assistant 的 tool_calls 消息回填
        messages.append(msg)

        if verbose:
            print(f"  → [round {round_num}] 模型调用 {len(msg.tool_calls)} 个工具：")

        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            tool_call_log.append({"name": name, "args": args})
            if verbose:
                print(f"      · {name}({args})")
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result = f"未知工具：{name}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"参数错误：{e}"
                except Exception as e:
                    result = f"工具执行失败：{e}"
            preview = (result or "")[:120].replace("\n", " ")
            if verbose:
                print(f"        ↩ {preview}{'...' if len(result or '') > 120 else ''}")
            # 以 role=tool 把结果回填，tool_call_id 必须对上
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    answer = msg.content or ""
    elapsed = time.time() - t0
    if verbose:
        print(f"  → [llm] 最终回答（{elapsed:.1f}s，共 {len(tool_call_log)} 次工具调用）")
    return {"answer": answer, "tool_calls": tool_call_log, "elapsed": elapsed}


# ── 入口 ───────────────────────────────────────────────────────────────────

DEMO_QUESTIONS = [
    "北京的经纬度是多少？",
    "北京的天气如何？",
    "上海今天的天气怎么样？",
    "宁德的经纬度和天气分别是什么？",
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="方式一：Function Call")
    parser.add_argument("--question", "-q", help="单个问题")
    parser.add_argument("--demo", action="store_true", help="跑内置示例问题集")

    parser.add_argument("--quiet", action="store_true", help="少输出（被 compare.py 调用时用）")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供 compare.py 解析）")
    args = parser.parse_args()

    client, model = build_client("deepseek")
    if not args.json:
        print(f"[Function Call] provider=deepseek model={model}\n")

    questions = DEMO_QUESTIONS if args.demo else ([args.question] if args.question else [DEMO_QUESTIONS[0]])
    results = []
    for i, q in enumerate(questions, 1):
        if not args.json:
            print("=" * 60)
            print(f"Q{i}：{q}")
            print("=" * 60)
        result = run(client, model, q, verbose=not (args.quiet or args.json))
        result["question"] = q
        results.append(result)
        if not args.json:
            print("\n最终回答：")
            print(result["answer"])
            print()

    if args.json:
        # 单问题输出单对象；demo 输出数组
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))


if __name__ == "__main__":
    main()