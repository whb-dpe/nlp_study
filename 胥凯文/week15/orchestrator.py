import asyncio
from typing import List, Dict, Optional, Callable
from collections import defaultdict

from base_agent import BaseAgent
from sub_agent import SubAgent
from task import Task, TaskStatus
from llm_client import DeepSeekClient
from config import Config


class Orchestrator(BaseAgent):
    def __init__(
        self,
        name: Optional[str] = None,
        llm_client: Optional[DeepSeekClient] = None,
        max_concurrent: Optional[int] = None,
    ):
        system_prompt = (
            "你是一个智能任务调度专家（Orchestrator）。"
            "你的职责是接收用户的复杂任务，将其拆解为多个可以并行执行的子任务，"
            "为每个子任务分配合适类型的子Agent，汇总所有子Agent的结果，"
            "最后给出完整的、高质量的最终答案。\n\n"
            "你需要：\n"
            "1. 准确理解用户的目标和需求\n"
            "2. 合理拆解为可独立执行的子任务（任务之间尽量解耦，便于并行）\n"
            "3. 为每个子任务选择最合适的角色（researcher/analyst/writer/coder/reviewer等）\n"
            "4. 汇总结果时要去重、整合、梳理逻辑，而不是简单拼接\n"
            "5. 最终输出要结构清晰、内容完整、符合用户预期"
        )
        super().__init__(name=name, llm_client=llm_client, system_prompt=system_prompt)
        self.max_concurrent = max_concurrent or Config.MAX_CONCURRENT_AGENTS
        self.sub_agents: Dict[str, SubAgent] = {}
        self.tasks: List[Task] = []
        self.on_task_complete: Optional[Callable[[Task], None]] = None

    def _default_system_prompt(self) -> str:
        return ""

    async def decompose_task(self, user_objective: str) -> List[Dict]:
        """使用LLM将用户目标拆解为多个子任务"""
        prompt = f"""
用户的总体目标是：
\"\"\"
{user_objective}
\"\"\"

请将这个目标拆解为多个可以并行执行的子任务。每个子任务需要指定：
1. description: 子任务的具体描述（要明确、可执行）
2. role: 负责该子任务的Agent角色，从以下角色中选择一个最匹配的：
   - researcher: 信息调研、资料收集
   - analyst: 数据分析、逻辑分析
   - writer: 文档撰写、内容整理
   - coder: 代码编写、技术方案
   - reviewer: 内容审核、质量检查
   - planner: 计划制定、任务拆解
   - translator: 翻译工作
   - general: 通用任务

请以JSON数组格式输出，例如：
[
  {{"description": "调研...", "role": "researcher"}},
  {{"description": "分析...", "role": "analyst"}}
]

只输出JSON，不要输出其他任何内容。不要包含markdown代码块标记。
"""
        print(f"[Orchestrator] 正在使用LLM进行任务拆解...")
        response = await self.call_llm(prompt, temperature=0.3)

        response = response.strip()
        if response.startswith("```"):
            lines = response.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            response = "\n".join(lines)
        response = response.strip()

        import json

        try:
            subtasks = json.loads(response)
            if not isinstance(subtasks, list):
                raise ValueError("Response is not a list")
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[Orchestrator] 任务拆解解析失败: {e}，使用通用单任务模式")
            subtasks = [{"description": user_objective, "role": "general"}]

        valid_roles = {"researcher", "analyst", "writer", "coder", "reviewer", "planner", "translator", "general"}
        normalized = []
        for idx, st in enumerate(subtasks):
            desc = st.get("description", "").strip()
            role = st.get("role", "general").lower()
            if role not in valid_roles:
                role = "general"
            if desc:
                normalized.append({"description": desc, "role": role, "index": idx})

        print(f"[Orchestrator] 任务拆解完成，共 {len(normalized)} 个子任务")
        return normalized

    def _get_or_create_agent(self, role: str, index: int) -> SubAgent:
        agent_key = f"{role}_{index}"
        if agent_key not in self.sub_agents:
            self.sub_agents[agent_key] = SubAgent(
                name=f"SubAgent-{role}-{index+1}",
                role=role,
                llm_client=self.llm_client,
            )
        return self.sub_agents[agent_key]

    async def _run_with_semaphore(
        self,
        task: Task,
        agent: SubAgent,
        semaphore: asyncio.Semaphore,
    ) -> Task:
        async with semaphore:
            result_task = await agent.run(task)
            if self.on_task_complete:
                self.on_task_complete(result_task)
            return result_task

    async def execute_subtasks(
        self,
        subtask_defs: List[Dict],
    ) -> List[Task]:
        """并行执行所有子任务，带并发数控制"""
        self.tasks = []
        for st in subtask_defs:
            task = Task(description=st["description"], metadata={"role": st["role"]})
            self.tasks.append(task)

        print(f"\n[Orchestrator] 开始并行执行 {len(self.tasks)} 个子任务 (最大并发: {self.max_concurrent})")
        print("=" * 70)

        semaphore = asyncio.Semaphore(self.max_concurrent)
        coroutines = []

        for idx, task in enumerate(self.tasks):
            role = subtask_defs[idx]["role"]
            agent = self._get_or_create_agent(role, idx)
            coroutines.append(self._run_with_semaphore(task, agent, semaphore))

        completed_tasks = await asyncio.gather(*coroutines)
        print("=" * 70)
        return list(completed_tasks)

    async def synthesize_results(
        self,
        user_objective: str,
        completed_tasks: List[Task],
    ) -> str:
        """汇总所有子任务的结果，生成最终答案"""
        completed = [t for t in completed_tasks if t.status == TaskStatus.COMPLETED]
        failed = [t for t in completed_tasks if t.status == TaskStatus.FAILED]

        print(f"\n[Orchestrator] 正在汇总结果 (成功: {len(completed)}, 失败: {len(failed)})")

        results_text = ""
        for i, task in enumerate(completed):
            results_text += f"\n--- 子任务 {i+1}: {task.description} ---\n"
            results_text += f"负责Agent: {task.assigned_agent[:8]}...\n"
            results_text += f"结果:\n{task.result}\n"

        if failed:
            results_text += f"\n--- 以下子任务执行失败 ---\n"
            for task in failed:
                results_text += f"  - {task.description}: {task.error}\n"

        prompt = f"""
用户的总体目标是：
\"\"\"
{user_objective}
\"\"\"

以下是多个子Agent并行工作后产出的结果（按子任务划分）：
\"\"\"
{results_text}
\"\"\"

请你作为总调度师，整合、梳理、去重、完善这些结果，输出一份结构清晰、内容完整、逻辑连贯的最终回答。
如果有子任务失败，请在最终结果中说明并尽可能从其他结果中弥补相关信息。
输出要专业、有组织，适合直接呈现给用户。
"""
        final_result = await self.call_llm(prompt, temperature=0.5)
        return final_result

    async def run(
        self,
        user_objective: str,
        custom_subtasks: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        执行完整的工作流：任务拆解 -> 并行执行 -> 结果汇总

        Args:
            user_objective: 用户的总体目标/任务描述
            custom_subtasks: 可选，手动指定子任务列表（跳过LLM拆解）。
                           每项需包含 description 和 role 字段。

        Returns:
            Dict: 包含 final_result, tasks_summary 等信息
        """
        if not user_objective.strip():
            raise ValueError("user_objective cannot be empty")

        print(f"\n{'='*70}")
        print(f"[Orchestrator] 收到新任务")
        print(f"  目标: {user_objective[:80]}{'...' if len(user_objective) > 80 else ''}")
        print(f"{'='*70}")

        if custom_subtasks is not None:
            print(f"[Orchestrator] 使用自定义子任务 ({len(custom_subtasks)} 个)")
            subtask_defs = []
            for idx, st in enumerate(custom_subtasks):
                subtask_defs.append({
                    "description": st["description"],
                    "role": st.get("role", "general"),
                    "index": idx,
                })
        else:
            subtask_defs = await self.decompose_task(user_objective)

        for i, st in enumerate(subtask_defs):
            print(f"  [{i+1}] ({st['role']:12s}) {st['description'][:60]}...")

        completed_tasks = await self.execute_subtasks(subtask_defs)

        final_result = await self.synthesize_results(user_objective, completed_tasks)

        tasks_summary = [t.to_dict() for t in completed_tasks]
        success_count = sum(1 for t in completed_tasks if t.status == TaskStatus.COMPLETED)

        print(f"\n{'='*70}")
        print(f"[Orchestrator] 全部工作完成！")
        print(f"  子任务总数: {len(completed_tasks)}")
        print(f"  成功: {success_count} | 失败: {len(completed_tasks) - success_count}")
        print(f"{'='*70}\n")

        return {
            "agent_id": self.agent_id,
            "user_objective": user_objective,
            "final_result": final_result,
            "tasks_summary": tasks_summary,
            "success_count": success_count,
            "total_count": len(completed_tasks),
        }

    async def close(self):
        await self.llm_client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()