"""
Function Calling API 版 ReAct Agent（带记忆 · 类实现）

教学重点：
  1. 类封装 + self.messages 实例属性实现跨轮次记忆持久化
  2. 上下文传递：每次 API 调用都传入完整 self.messages
  3. 结果累积：用户问题 / 工具调用 / 最终回答全部写回 self.messages
  4. while True REPL 循环：复用同一个 agent 实例反复提问

使用方式：
  python react_function_calling.py --repl                 # REPL 多轮（带记忆）
  python react_function_calling.py --question "问题内容"   # 单轮
  python react_function_calling.py --question "..." --max_steps 8
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
- 保留对话上下文，用户可能会追问上一轮的问题
"""


# ──────────────────────────────────────────────────────────────────────────────
# ReActAgentFC 类：self.messages 记忆存储 + 上下文传递 + 结果累积
# ──────────────────────────────────────────────────────────────────────────────
class ReActAgentFC:
    """Function Calling 版 ReAct Agent，支持跨轮次记忆持久化"""

    def __init__(self):
        from tools import TOOLS_MAP, TOOLS_SCHEMA
        self.TOOLS_MAP = TOOLS_MAP
        self.TOOLS_SCHEMA = TOOLS_SCHEMA
        # ✅ 记忆存储：self.messages 作为实例属性，跨轮次持久化
        self.messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
        ]
        self.step_counter = 0  # 全局累计步数

    def reset(self):
        """清空对话记忆，重开新对话"""
        self.messages = [
            {"role": "system", "content": FC_SYSTEM_PROMPT},
        ]
        self.step_counter = 0

    def run(self, question: str, max_steps: int = 10) -> Generator[dict, None, None]:
        """
        执行一轮 ReAct 推理：
        ✅ 上下文传递：每次 LLM API 调用都传入完整 self.messages
        ✅ 结果累积：用户问题、工具调用结果、最终回答全部写回 self.messages
        """
        # ✅ 结果累积 1/4：新问题写入 messages
        self.messages.append({"role": "user", "content": question})
        step_in_round = 0

        while True:
            step_in_round += 1
            self.step_counter += 1

            if step_in_round > max_steps:
                msg = f"已达本轮最大步数 {max_steps}，未能得出最终答案"
                self.messages.append({"role": "assistant", "content": msg})
                yield {"step": self.step_counter, "type": "max_steps", "answer": msg}
                return

            # ✅ 上下文传递：每次 API 调用都传入完整的 self.messages
            response = client.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=self.TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0,
            )
            msg    = response.choices[0].message
            reason = response.choices[0].finish_reason

            # 模型决定直接回答（无工具调用）
            if reason == "stop" or not msg.tool_calls:
                final_answer = msg.content or "（模型返回空内容）"
                # ✅ 结果累积 2/4：最终回答写入 messages
                self.messages.append({"role": "assistant", "content": final_answer})
                yield {
                    "step":   self.step_counter,
                    "type":   "final",
                    "thought": "",
                    "answer": final_answer,
                }
                return

            # ✅ 结果累积 3/4：assistant tool_calls 消息写入 messages
            # 注意：SDK 返回的 msg 是 ChatCompletionMessage 对象（非 dict），
            #       必须转为标准 dict，否则后续用 m["role"] 遍历会报
            #       TypeError: 'ChatCompletionMessage' object is not subscriptable
            if hasattr(msg, "model_dump"):
                msg_dict = msg.model_dump()
            elif hasattr(msg, "dict"):
                msg_dict = msg.dict()
            else:
                msg_dict = {"role": msg.role, "content": msg.content}
                if getattr(msg, "tool_calls", None):
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
            self.messages.append(msg_dict)

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                tool_fn = self.TOOLS_MAP.get(tool_name)
                if tool_fn is None:
                    observation = f"未知工具 '{tool_name}'"
                else:
                    try:
                        observation = tool_fn(**tool_args)
                    except TypeError as e:
                        observation = f"工具参数错误: {e}"

                yield {
                    "step":         self.step_counter,
                    "type":         "action",
                    "thought":      "",
                    "action":       tool_name,
                    "action_input": tool_args,
                    "observation":  str(observation),
                }

                # ✅ 结果累积 4/4：工具调用结果写入 messages
                self.messages.append({
                    "role":         "tool",
                    "tool_call_id": tool_call.id,
                    "content":      str(observation),
                })

    @staticmethod
    def _msg_role(m) -> str:
        """兼容 dict / ChatCompletionMessage 对象两种格式，返回 role"""
        if isinstance(m, dict):
            return m.get("role", "")
        return getattr(m, "role", "")

    def run_and_print(self, question: str, max_steps: int = 10):
        """执行一轮推理并彩色打印"""
        user_count = len([m for m in self.messages if self._msg_role(m) == "user"])
        print(f"\n{'─'*60}")
        print(f"问题: {question}")
        print(f"模型: {MODEL}  实现: Function Calling (带记忆)  对话轮次: {user_count}")
        print('─'*60)
        start = time.time()
        step_start = self.step_counter

        for step_data in self.run(question, max_steps=max_steps):
            stype = step_data["type"]
            if stype == "action":
                print(f"\n[Step {step_data['step']}]")
                print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
                print(_c("action",  f"🔧 Action:  {step_data['action']}"))
                print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
                obs = step_data['observation']
                print(_c("obs", f"👁  Obs:     {obs[:300]}{'…' if len(obs) > 300 else ''}"))
            elif stype == "final":
                elapsed = time.time() - start
                print(f"\n{'─'*60}")
                print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
                print(f"\n本轮步数: {self.step_counter - step_start}，累计步数: {self.step_counter}，耗时 {elapsed:.1f}s")
            elif stype in ("error", "max_steps"):
                print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


# ── CLI 彩色输出辅助 ─────────────────────────────────────────────────────────
COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


# ── 兼容性函数（保持旧模块级接口，无跨轮记忆）─────────────────────────────────
def run_and_print(question: str, max_steps: int = 10):
    """兼容旧接口：创建临时 agent 执行单轮（无记忆）"""
    ReActAgentFC().run_and_print(question, max_steps)

def run(question: str, max_steps: int = 10) -> Generator[dict, None, None]:
    """兼容旧接口：创建临时 agent 执行单轮（无记忆）"""
    yield from ReActAgentFC().run(question, max_steps)


# ── 脚本入口：支持 --repl 进入 while True REPL 循环 ──────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",  default=None,
                        help="单轮问题；不指定则进入 REPL 多轮模式")
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--repl",      action="store_true",
                        help="强制进入 REPL 多轮对话模式（带记忆）")
    args = parser.parse_args()

    if args.repl or args.question is None:
        # ✅ while True REPL 循环：复用同一个 agent 实例，跨轮次保持记忆
        agent = ReActAgentFC()
        print(f"{'='*60}")
        print(f"  ReAct Agent REPL 模式 (Function Calling 版 · 带记忆)")
        print(f"  输入问题进行多轮对话 | /reset 清空记忆 | exit 退出")
        print(f"{'='*60}")
        while True:
            try:
                question = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("👋 再见！")
                break
            if question == "/reset":
                agent.reset()
                print("🧹 对话记忆已清空")
                continue
            agent.run_and_print(question, args.max_steps)
    else:
        run_and_print(args.question, args.max_steps)