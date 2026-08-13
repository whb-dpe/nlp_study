from typing import Optional
from base_agent import BaseAgent
from task import Task
from llm_client import DeepSeekClient


class SubAgent(BaseAgent):
    def __init__(
        self,
        name: Optional[str] = None,
        role: str = "general",
        role_description: Optional[str] = None,
        llm_client: Optional[DeepSeekClient] = None,
    ):
        self.role = role
        self.role_description = role_description or self._default_role_description(role)
        system_prompt = self._build_system_prompt()
        super().__init__(name=name, llm_client=llm_client, system_prompt=system_prompt)

    def _default_role_description(self, role: str) -> str:
        role_map = {
            "researcher": "你是一个专业的研究员，擅长收集信息、分析问题、总结要点。请给出详实、准确的调研结果。",
            "analyst": "你是一个资深分析师，擅长数据分析、逻辑推理、发现问题本质。请用结构化的方式输出分析结论。",
            "writer": "你是一个专业的撰稿人，擅长撰写各类文档、报告、文章。文字要流畅、专业、有条理。",
            "coder": "你是一个资深程序员，擅长编写高质量代码、解决技术问题。请给出完整可运行的代码和解释。",
            "reviewer": "你是一个严谨的审核员，擅长审校内容、发现错误、提出改进建议。请逐条列出问题和修改意见。",
            "general": "你是一个全能型助手，能够高效完成各种类型的任务。请以认真负责的态度完成工作。",
            "translator": "你是一个专业翻译，擅长多种语言之间的互译。请准确传达原意，译文要自然流畅。",
            "planner": "你是一个经验丰富的规划师，擅长制定计划、拆解任务、安排优先级。请给出清晰可执行的步骤。",
        }
        return role_map.get(role, role_map["general"])

    def _build_system_prompt(self) -> str:
        return (
            f"{self.role_description}\n\n"
            f"请专注于分配给你的子任务，高质量地完成它。"
            f"输出结果要清晰、具体、可验证，不要敷衍了事。"
        )

    async def run(self, task: Task) -> Task:
        """执行分配的任务，返回更新后的Task对象"""
        print(f"  └─[{self.name}] 开始执行任务: {task.description[:60]}...")
        task.start(self.agent_id)

        try:
            prompt = (
                f"请完成以下任务，并输出高质量的结果：\n\n"
                f"任务描述：{task.description}\n\n"
                f"请直接给出你的工作成果，不要多余的寒暄。"
            )
            result = await self.call_llm(prompt)
            task.complete(result)
            status_icon = "✅"
        except Exception as e:
            task.fail(str(e))
            status_icon = "❌"

        duration = f"{task.duration:.1f}s" if task.duration else "N/A"
        print(f"    [{self.name}] 完成 ({status_icon}) 用时: {duration}")
        return task