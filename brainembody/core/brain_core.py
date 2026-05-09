"""
具身智能核心大脑
整合 LLM、记忆系统和具身感知
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from .llm_adapter import LLMAdapter, LLMConfig, Message


@dataclass
class Perception:
    """感知数据"""
    visual: np.ndarray = None
    proprioception: np.ndarray = None
    touch: np.ndarray = None
    audio: np.ndarray = None
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().timestamp()


@dataclass
class Action:
    """动作指令"""
    action_type: str
    parameters: Dict[str, Any]
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class BrainState:
    """大脑状态"""
    attention_focus: str = "general"
    emotional_state: str = "neutral"
    cognitive_load: float = 0.0
    working_memory_active: int = 0


class BrainCore:
    """
    具身智能核心大脑

    创新点：
    1. 多模态感知融合
    2. LLM 驱动的推理
    3. 实时状态追踪
    4. 动作规划与执行
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None):
        self.llm = LLMAdapter(llm_config)
        self.state = BrainState()
        self.perception_buffer = []
        self.action_history = []

        self.system_prompt = """你是具身智能机器人的核心大脑。

你的能力：
1. 理解多模态感知（视觉、触觉、位置）
2. 推理当前环境和状态
3. 生成安全有效的动作计划
4. 从经验中学习和反思

当前场景：仿真机器人环境
你的目标：帮助智能体完成具身任务

重要原则：
- 安全第一
- 效率优先
- 持续学习
"""

    def perceive(self, perception: Perception) -> Dict:
        """
        处理感知输入

        Args:
            perception: 感知数据

        Returns:
            感知分析结果
        """
        self.perception_buffer.append(perception)

        if len(self.perception_buffer) > 100:
            self.perception_buffer.pop(0)

        analysis = {
            "timestamp": perception.timestamp,
            "visual_shape": perception.visual.shape if perception.visual is not None else None,
            "has_proprioception": perception.proprioception is not None,
            "attention": self.state.attention_focus
        }

        return analysis

    def think(self, task: str, context: Optional[Dict] = None) -> str:
        """
        思考和推理

        Args:
            task: 当前任务
            context: 额外上下文

        Returns:
            思考结果
        """
        context_str = ""
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

        prompt = f"""任务：{task}

上下文：
{context_str}

请进行推理并给出回答。"""

        return self.llm.chat(prompt, system_prompt=self.system_prompt)

    def plan_action(self, goal: str, current_state: Dict) -> Action:
        """
        规划动作

        Args:
            goal: 目标描述
            current_state: 当前状态

        Returns:
            动作指令
        """
        state_desc = "\n".join([f"- {k}: {v}" for k, v in current_state.items()])

        prompt = f"""目标：{goal}

当前状态：
{state_desc}

请生成具体的动作计划。

输出格式：
动作类型：[具体类型]
参数：[动作参数]
理由：[为什么这个动作]
置信度：[0-1之间]"""

        response = self.llm.chat(prompt, system_prompt=self.system_prompt)

        action = self._parse_action_response(response)
        self.action_history.append(action)

        return action

    def _parse_action_response(self, response: str) -> Action:
        """解析动作响应"""
        lines = response.split('\n')

        action_type = "move"
        parameters = {}
        confidence = 0.8
        reasoning = ""

        for line in lines:
            line_lower = line.lower()
            if '动作类型' in line or 'action_type' in line_lower:
                parts = line.split('：') or line.split(':')
                if len(parts) > 1:
                    action_type = parts[1].strip().lower()
            elif '参数' in line or 'parameters' in line_lower:
                parts = line.split('：') or line.split(':')
                if len(parts) > 1:
                    parameters = {"description": parts[1].strip()}
            elif '置信度' in line or 'confidence' in line_lower:
                parts = line.split('：') or line.split(':')
                if len(parts) > 1:
                    try:
                        confidence = float(parts[1].strip())
                    except:
                        confidence = 0.8
            elif '理由' in line or 'reasoning' in line_lower:
                parts = line.split('：') or line.split(':')
                if len(parts) > 1:
                    reasoning = parts[1].strip()

        return Action(
            action_type=action_type,
            parameters=parameters,
            confidence=confidence,
            reasoning=reasoning
        )

    def reflect(self, experience: str) -> str:
        """反思经验"""
        return self.llm.reflect(experience)

    def update_state(self, **kwargs):
        """更新大脑状态"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def get_state_summary(self) -> Dict:
        """获取状态摘要"""
        return {
            "attention": self.state.attention_focus,
            "emotion": self.state.emotional_state,
            "cognitive_load": self.state.cognitive_load,
            "working_memory": self.state.working_memory_active,
            "perception_buffer_size": len(self.perception_buffer),
            "action_history_size": len(self.action_history),
            "llm_stats": self.llm.get_stats()
        }

    def reset(self):
        """重置大脑"""
        self.state = BrainState()
        self.perception_buffer.clear()
        self.action_history.clear()
        self.llm.reset_conversation()
