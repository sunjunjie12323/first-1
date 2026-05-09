"""
DeepSeek LLM 适配器
连接 DeepSeek API 作为大脑核心
"""

import os
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM 配置"""
    api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 60


@dataclass
class Message:
    """对话消息"""
    role: str
    content: str
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict:
        return {"role": self.role, "content": self.content}


class LLMAdapter:
    """
    DeepSeek LLM 适配器

    核心功能：
    - 文本生成
    - 对话管理
    - 具身指令生成
    - 思维推理
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

        if not self.config.api_key:
            print("⚠️ 警告：未设置 DEEPSEEK_API_KEY，将使用模拟模式")
            self.mock_mode = True
        else:
            self.mock_mode = False
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url
                )
            except ImportError:
                print("⚠️ 未安装 openai 库，将使用模拟模式")
                self.mock_mode = True

        self.conversation_history: List[Message] = []
        self.total_tokens_used = 0

    def chat(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        发送对话请求

        Args:
            prompt: 用户输入
            system_prompt: 系统提示

        Returns:
            LLM 回复
        """
        messages = []

        if system_prompt:
            messages.append(Message("system", system_prompt).to_dict())

        for msg in self.conversation_history[-10:]:
            messages.append(msg.to_dict())

        messages.append(Message("user", prompt).to_dict())

        if self.mock_mode:
            response = self._mock_response(prompt)
        else:
            response = self._real_request(messages)

        self.conversation_history.append(Message("user", prompt))
        self.conversation_history.append(Message("assistant", response))

        return response

    def _real_request(self, messages: List[Dict]) -> str:
        """真实 API 请求"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout
            )

            self.total_tokens_used += response.usage.total_tokens

            return response.choices[0].message.content

        except Exception as e:
            print(f"❌ API 请求失败: {e}")
            return self._mock_response(messages[-1]["content"])

    def _mock_response(self, prompt: str) -> str:
        """模拟响应（用于测试）"""
        time.sleep(0.1)

        prompt_lower = prompt.lower()

        if "具身" in prompt or "机器人" in prompt or "action" in prompt_lower:
            return """我会执行以下动作：
1. 观察环境 - 获取周围空间信息
2. 规划路径 - 避开障碍物
3. 移动执行 - 朝目标位置移动
4. 反馈确认 - 验证动作完成状态

基于当前环境，我会采取最安全高效的移动策略。"""

        elif "记忆" in prompt or "memory" in prompt_lower:
            return """记忆系统状态：
- 工作记忆：5 个活跃概念
- 情景记忆：最近 10 次交互
- 语义记忆：领域知识已加载
- 记忆巩固：已触发 2 次

建议：当前记忆负荷正常，可以继续学习新任务。"""

        elif "目标" in prompt or "goal" in prompt_lower:
            return """目标分析：
1. 主目标：到达指定位置
2. 子目标：
   - 识别当前位置
   - 检测路径障碍
   - 规划最优路径
   - 执行移动动作
3. 奖励信号：成功 +1.0，失败 -0.5

我会逐步完成这些目标。"""

        else:
            return f"""我理解你的输入："{prompt[:50]}..."

作为具身智能大脑，我会：
1. 处理感知信息
2. 推理当前状态
3. 生成动作计划
4. 评估执行结果

请提供具体的具身任务。"""

    def generate_action_plan(self, state_description: str) -> List[str]:
        """生成具身动作计划"""
        prompt = f"""给定当前状态：
{state_description}

生成 5 个具体的具身动作序列（用中文描述）：
1. 动作名称
2. 动作参数
3. 预期结果
4. 风险评估

只输出动作序列，不要其他解释。"""

        response = self.chat(prompt)

        actions = []
        for line in response.split('\n'):
            if line.strip() and (line[0].isdigit() or '•' in line):
                actions.append(line.strip())

        return actions[:5]

    def reflect(self, experience: str) -> str:
        """生成反思"""
        prompt = f"""基于以下经验进行反思：
{experience}

反思要点：
1. 做得好的地方
2. 需要改进的地方
3. 学到的教训
4. 下次行动的调整

用中文输出。"""

        return self.chat(prompt)

    def reset_conversation(self):
        """重置对话历史"""
        self.conversation_history.clear()

    def get_stats(self) -> Dict:
        """获取使用统计"""
        return {
            "total_messages": len(self.conversation_history),
            "total_tokens_used": self.total_tokens_used,
            "mock_mode": self.mock_mode
        }
