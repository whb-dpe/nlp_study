from abc import ABC, abstractmethod
from typing import Optional
import uuid
from llm_client import DeepSeekClient


class BaseAgent(ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        llm_client: Optional[DeepSeekClient] = None,
        system_prompt: Optional[str] = None,
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name or f"{self.__class__.__name__}_{self.agent_id[:8]}"
        self.llm_client = llm_client or DeepSeekClient()
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return "你是一个有帮助的AI助手，请认真、准确地完成用户的任务。"

    @abstractmethod
    async def run(self, *args, **kwargs):
        """执行Agent的核心逻辑，子类必须实现"""
        pass

    async def call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        actual_system = system_prompt if system_prompt is not None else self.system_prompt
        return await self.llm_client.generate(
            user_prompt=user_prompt,
            system_prompt=actual_system,
            temperature=temperature,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} id={self.agent_id[:8]}>"