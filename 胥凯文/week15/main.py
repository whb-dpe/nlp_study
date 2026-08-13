import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import Orchestrator


EXAMPLE_OBJECTIVES = {
    1: (
        "分析人工智能大语言模型在2024年的发展现状，包括：主要的技术突破、"
        "代表性的产品、应用领域、以及面临的挑战和未来趋势。最后给出一份综述报告。"
    ),
    2: (
        "我想学习Python编程。请帮我：1) 调研Python的学习路径和最佳资源；"
        "2) 分析Python与其他主流语言的优劣对比；3) 制定一份为期3个月的学习计划；"
        "4) 写一个包含数据结构、面向对象、文件操作的综合示例代码。"
    ),
    3: (
        "为一家初创科技公司撰写一份完整的商业计划书概要，"
        "包括市场分析、产品定位、商业模式、团队介绍、财务预测、风险分析等模块。"
    ),
}


def print_banner():
    banner = r"""
  ____                      _           _       
 / ___| _   _  ___ ___  ___| |_ __ _  _| |_ ___ 
 \___ \| | | |/ __/ _ \/ __| __/ _` |/ / __/ _ \
  ___) | |_| | (_|  __/ (__| || (_|   <| ||  __/
 |____/ \__,_|\___\___|\___|\__\__,_|\_\\__\___|
    Multi-Agent Parallel System  |  DeepSeek LLM
"""
    print(banner)


async def example_1_auto_decompose():
    """示例1：自动任务拆解模式"""
    print("\n" + "=" * 70)
    print(" 示例1：自动任务拆解 —— AI大模型发展现状调研")
    print("=" * 70)

    async with Orchestrator(max_concurrent=10) as orchestrator:
        result = await orchestrator.run(
            user_objective=EXAMPLE_OBJECTIVES[1],
        )

        print("\n" + "=" * 70)
        print(" 📋 最终结果：")
        print("=" * 70)
        print(result["final_result"])

        print("\n" + "=" * 70)
        print(" 📊 子任务执行情况：")
        print("=" * 70)
        for t in result["tasks_summary"]:
            status = "✅" if t["status"] == "completed" else "❌"
            duration = f"{t['duration']:.1f}s" if t["duration"] else "N/A"
            print(f" {status} [{t['status']:9s}] ({duration}) {t['description'][:60]}...")

    return result


async def example_2_custom_subtasks():
    """示例2：手动指定子任务"""
    print("\n" + "=" * 70)
    print(" 示例2：自定义子任务 —— Python学习方案")
    print("=" * 70)

    custom_subtasks = [
        {
            "description": (
                "调研Python的最佳学习路径：从零基础到进阶，分阶段列出学习内容、推荐教材、"
                "在线课程、练习网站、实战项目等资源。"
            ),
            "role": "researcher",
        },
        {
            "description": (
                "对比分析Python与JavaScript、Java、Go、Rust这4门语言的优缺点，"
                "从语法简洁性、性能、生态系统、就业市场、适用场景等维度进行对比，"
                "用表格和文字说明。"
            ),
            "role": "analyst",
        },
        {
            "description": (
                "制定一份为期3个月的Python学习计划，按周分解：每周学习目标、"
                "学习内容、预计学时、检验成果的小项目。"
            ),
            "role": "planner",
        },
        {
            "description": (
                "编写一个Python综合示例代码，包含：1) 自定义数据结构(链表/树二选一)；"
                "2) 面向对象编程(继承、多态)；3) 文件读写操作(JSON/TXT)；"
                "4) 异常处理；5) 类型注解。代码要有注释和使用示例。"
            ),
            "role": "coder",
        },
    ]

    async with Orchestrator(max_concurrent=10) as orchestrator:
        result = await orchestrator.run(
            user_objective=EXAMPLE_OBJECTIVES[2],
            custom_subtasks=custom_subtasks,
        )

        print("\n" + "=" * 70)
        print(" 📋 最终结果：")
        print("=" * 70)
        print(result["final_result"])

    return result


async def example_interactive():
    """交互式模式：用户输入自定义目标"""
    print("\n" + "=" * 70)
    print(" 交互式模式")
    print("=" * 70)

    user_objective = input("\n请输入你的任务目标 (直接回车退出): ").strip()
    if not user_objective:
        print("已退出。")
        return

    concurrent_input = input("最大并行数 (默认10): ").strip()
    max_concurrent = int(concurrent_input) if concurrent_input.isdigit() else 10

    async with Orchestrator(max_concurrent=max_concurrent) as orchestrator:
        result = await orchestrator.run(user_objective=user_objective)

        output_file = "agent_output.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"任务目标:\n{user_objective}\n\n")
            f.write("=" * 70 + "\n")
            f.write("最终结果:\n")
            f.write("=" * 70 + "\n")
            f.write(result["final_result"] + "\n\n")
            f.write("=" * 70 + "\n")
            f.write("子任务详情:\n")
            f.write("=" * 70 + "\n")
            for t in result["tasks_summary"]:
                f.write(f"\n[{t['status']}] {t['description']}\n")
                if t["result"]:
                    f.write(f"结果: {t['result']}\n")
                if t["error"]:
                    f.write(f"错误: {t['error']}\n")

        print(f"\n完整结果已保存到: {output_file}")
        print("\n" + "=" * 70)
        print(" 📋 最终结果摘要：")
        print("=" * 70)
        preview = result["final_result"][:1500]
        print(preview + ("\n... (内容已截断，完整内容请查看输出文件)" if len(result["final_result"]) > 1500 else ""))


async def main():
    print_banner()

    print("\n请选择运行模式：")
    print("  1. 示例1：自动任务拆解（AI大模型调研）")
    print("  2. 示例2：自定义子任务（Python学习方案）")
    print("  3. 交互式模式（自己输入任务）")
    print("  4. 运行全部示例")
    print("  0. 退出")

    choice = input("\n请输入选项 (0-4): ").strip()

    if choice == "1":
        await example_1_auto_decompose()
    elif choice == "2":
        await example_2_custom_subtasks()
    elif choice == "3":
        await example_interactive()
    elif choice == "4":
        await example_1_auto_decompose()
        await example_2_custom_subtasks()
    elif choice == "0":
        print("再见！")
        return
    else:
        print("无效选项，默认运行示例1...")
        await example_1_auto_decompose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出。")
    except Exception as e:
        print(f"\n程序出错: {e}")
        import traceback
        traceback.print_exc()