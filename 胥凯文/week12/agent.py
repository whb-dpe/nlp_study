"""
统一入口：Function Calling 版 ReAct Agent（带记忆 + REPL 模式）

使用方式：
  # 单轮模式
  python agent.py --question "茅台2023年毛利率是多少？"

  # REPL 多轮对话模式（推荐，带跨轮次记忆）
  python agent.py --repl
  python agent.py --repl --max_steps 8

  # REPL 内命令：
  #   /reset   清空对话记忆
  #   exit     退出程序

环境变量：
  AGENT_MODEL        默认 deepseek-v4-flash，可换其他模型
"""

import os
import argparse
import traceback

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"


# ──────────────────────────────────────────────────────────────────────────────
# AgentRunner 类：同一个 agent 实例复用 + while True REPL 循环
# ──────────────────────────────────────────────────────────────────────────────
class AgentRunner:
    """
    Agent 运行器（Function Calling 版）

    核心特性：
    ✅ 实例复用：同一个 agent 实例全程复用 → 跨轮次记忆持久化
    ✅ REPL 循环：while True 持续接收输入，反复提问
    ✅ 记忆管理：支持 /reset 清空记忆
    """

    def __init__(self, max_steps: int = 10):
        self.max_steps = max_steps
        from react_function_calling import ReActAgentFC
        self.agent = ReActAgentFC()   # ✅ 单例 agent 实例，跨轮次复用

    def reset_memory(self):
        """清空对话记忆"""
        if self.agent is not None and hasattr(self.agent, "reset"):
            self.agent.reset()

    def run_one(self, question: str):
        """执行单轮推理，复用 agent 实例，保持上下文"""
        self.agent.run_and_print(question, self.max_steps)

    def run_repl(self):
        """
        ✅ while True REPL 主循环：
        复用同一个 agent 实例，跨轮次持久化记忆
        """
        print(f"{'='*60}")
        print(f"  🤖 ReAct Financial Agent  (Function Calling · 带记忆)")
        print(f"  ──────────────────────────────────────────────────────")
        print(f"  输入问题 → 多轮对话，自动保留上下文")
        print(f"  /reset       → 清空对话记忆")
        print(f"  exit / Ctrl+C → 退出")
        print(f"{'='*60}")

        while True:
            try:
                raw = input("\n👤 你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 再见！")
                break

            if not raw:
                continue

            # ── 内置命令 ────────────────────────────────────
            if raw.lower() in ("exit", "quit", "q"):
                print("👋 再见！")
                break
            if raw == "/reset":
                self.reset_memory()
                print("🧹 对话记忆已清空，开始新对话")
                continue
            if raw.startswith("/"):
                print(f"❌ 未知命令 '{raw}'，支持: /reset  exit")
                continue

            # ── 正常推理（复用同一个 agent 实例） ────────────
            try:
                self.run_one(raw)
            except Exception as e:
                print(f"❌ 执行出错: {e}")
                traceback.print_exc()


# ── 脚本入口 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent (带记忆)")
    parser.add_argument("--question",  default=None,
                        help="单轮模式：指定问题后执行一次即退出")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="单轮最大推理步数 (默认: 10)")
    parser.add_argument("--repl",      action="store_true",
                        help="进入 REPL 多轮对话模式（带记忆）")
    args = parser.parse_args()

    runner = AgentRunner(max_steps=args.max_steps)

    if args.repl or args.question is None:
        # REPL 模式：显式 --repl 或未指定 --question 时进入多轮对话
        runner.run_repl()
    else:
        # 单轮模式
        runner.run_one(args.question)