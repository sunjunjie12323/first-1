"""
NeuroBrain Conversation Memory
创新性类脑对话记忆系统

核心创新点：
1. 动态记忆图谱 (Dynamic Memory Graph)
2. 概念关联网络 (Concept Association Network)
3. 注意力引导的记忆检索 (Attention-Guided Recall)
4. 记忆衰减与巩固机制
5. 情感加权记忆
"""

import numpy as np
import json
import hashlib
from collections import defaultdict
from datetime import datetime


class ConceptNode:
    """概念节点 - 存储语义信息"""
    def __init__(self, text, importance=1.0):
        self.text = text
        self.embedding = self._generate_embedding(text)
        self.importance = importance
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.connections = {}
        self.emotional_value = 0.0

    def _generate_embedding(self, text):
        """简单的语义向量生成（实际应用中应使用预训练模型）"""
        words = text.lower().split()
        vector = np.zeros(32)
        for i, word in enumerate(words[:32]):
            vector[i] = hash(word) % 100 / 100
        return vector

    def connect(self, other, strength=0.5):
        """建立概念之间的连接"""
        if other not in self.connections:
            self.connections[other] = {'strength': 0.0, 'count': 0}
        self.connections[other]['strength'] = min(1.0, self.connections[other]['strength'] + strength)
        self.connections[other]['count'] += 1

    def update_importance(self, delta):
        """更新重要性"""
        self.importance = max(0.1, min(10.0, self.importance + delta))

    def decay(self, rate=0.99):
        """记忆衰减"""
        self.importance *= rate
        for conn in self.connections.values():
            conn['strength'] *= rate


class MemoryGraph:
    """动态记忆图谱 - 核心创新架构"""
    def __init__(self):
        self.nodes = {}
        self.context_history = []
        self.max_context_length = 10
        self.decay_rate = 0.98

    def _get_node_id(self, text):
        """生成唯一节点ID"""
        return hashlib.md5(text.encode()).hexdigest()[:8]

    def add_concept(self, text, importance=1.0, emotional_value=0.0):
        """添加概念到记忆图谱"""
        node_id = self._get_node_id(text)
        
        if node_id in self.nodes:
            self.nodes[node_id].update_importance(importance * 0.5)
            self.nodes[node_id].emotional_value = max(self.nodes[node_id].emotional_value, emotional_value)
        else:
            self.nodes[node_id] = ConceptNode(text, importance)
            self.nodes[node_id].emotional_value = emotional_value

        # 与最近的上下文建立连接
        for ctx in self.context_history[-3:]:
            ctx_node = self.nodes.get(self._get_node_id(ctx))
            if ctx_node:
                self.nodes[node_id].connect(ctx_node, strength=0.3)

        # 更新上下文
        self.context_history.append(text)
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)

        return self.nodes[node_id]

    def recall(self, query, top_k=5):
        """基于查询检索记忆"""
        query_embedding = self._text_to_embedding(query)
        scores = []

        for node_id, node in self.nodes.items():
            # 语义相似度
            similarity = np.dot(query_embedding, node.embedding) / \
                        (np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding) + 1e-8)
            
            # 综合得分：相似度 + 重要性 + 情感值
            score = similarity * 0.5 + node.importance * 0.3 + node.emotional_value * 0.2
            scores.append((score, node))

        # 按得分排序
        scores.sort(reverse=True, key=lambda x: x[0])
        return [node for _, node in scores[:top_k]]

    def _text_to_embedding(self, text):
        """文本转向量"""
        words = text.lower().split()
        vector = np.zeros(32)
        for i, word in enumerate(words[:32]):
            vector[i] = hash(word) % 100 / 100
        return vector

    def consolidate(self):
        """记忆巩固 - 模拟睡眠重放"""
        # 找到最重要的节点
        important_nodes = sorted(self.nodes.values(), 
                                key=lambda x: x.importance + x.emotional_value, 
                                reverse=True)[:10]

        # 强化重要节点之间的连接
        for i, node1 in enumerate(important_nodes):
            for j, node2 in enumerate(important_nodes[i+1:]):
                if node2 in node1.connections:
                    node1.connections[node2]['strength'] *= 1.1

        # 衰减所有记忆
        for node in self.nodes.values():
            node.decay(self.decay_rate)

    def get_context_summary(self):
        """获取上下文摘要"""
        if not self.context_history:
            return "无上下文"
        return "；".join(self.context_history[-5:])


class DialogMemory:
    """对话记忆系统"""
    def __init__(self):
        self.memory_graph = MemoryGraph()
        self.conversation_history = []
        self.persona = {
            'name': 'NeuroBrain',
            'role': '类脑AI助手',
            'traits': ['好奇', '专注', '善于学习'],
            'goals': '帮助用户进行类脑AI研究'
        }

    def process_message(self, user_message):
        """处理用户消息"""
        # 分析情感
        emotional_value = self._analyze_emotion(user_message)
        
        # 添加到记忆图谱
        self.memory_graph.add_concept(user_message, importance=1.0, emotional_value=emotional_value)
        
        # 检索相关记忆
        related_memories = self.memory_graph.recall(user_message, top_k=3)
        
        # 添加到对话历史
        self.conversation_history.append({
            'user': user_message,
            'timestamp': datetime.now().isoformat(),
            'related_memories': [node.text for node in related_memories]
        })

        return related_memories

    def _analyze_emotion(self, text):
        """简单情感分析"""
        positive_words = ['好', '棒', '喜欢', '爱', '感谢', '赞', '强', '厉害']
        negative_words = ['坏', '差', '讨厌', '恨', '不行', '糟糕', '垃圾']
        
        score = 0
        for word in positive_words:
            if word in text:
                score += 0.3
        for word in negative_words:
            if word in text:
                score -= 0.3
        return score

    def generate_response(self, user_message):
        """生成响应"""
        related_memories = self.process_message(user_message)
        
        # 构建响应
        response_parts = []
        
        # 如果有相关记忆
        if related_memories:
            response_parts.append(f"我记得你之前提到过：")
            for i, memory in enumerate(related_memories[:2], 1):
                response_parts.append(f"  {i}. {memory.text}")
        
        # 添加个性化元素
        response_parts.append("\n基于我们的对话，")
        
        # 简单的思维模拟
        thinking = self._simulate_thinking(user_message, related_memories)
        response_parts.append(thinking)
        
        return "\n".join(response_parts)

    def _simulate_thinking(self, message, memories):
        """模拟思维过程"""
        if any(m.text for m in memories if '研究' in m.text):
            return "你一直在进行类脑AI研究，这很有趣！我们可以一起探索记忆巩固机制。"
        elif any(m.text for m in memories if '记忆' in m.text):
            return "关于记忆系统，我可以帮你测试不同的巩固策略。"
        elif any(m.text for m in memories if '数据' in m.text):
            return "大数据处理需要分布式架构，我们可以讨论如何扩展当前框架。"
        else:
            return "我正在学习你的需求，让我们一起探索类脑AI的可能性。"

    def daily_consolidation(self):
        """每日记忆巩固"""
        self.memory_graph.consolidate()
        print("记忆巩固完成")


def demo_conversation():
    """演示对话记忆功能"""
    print("=" * 70)
    print("NeuroBrain 对话记忆演示")
    print("=" * 70)
    print("这个系统可以记住你之前说过的话！\n")

    dialog_memory = DialogMemory()

    # 模拟对话
    messages = [
        "我想研究类脑记忆系统",
        "你能帮我设计一个记忆框架吗？",
        "我们可以测试不同的记忆巩固策略",
        "我需要处理大规模的数据",
        "你还记得我之前说过什么吗？"
    ]

    for i, message in enumerate(messages):
        print(f"用户: {message}")
        response = dialog_memory.generate_response(message)
        print(f"NeuroBrain: {response}")
        print()

    # 记忆巩固
    dialog_memory.daily_consolidation()

    print("\n" + "=" * 70)
    print("对话记忆系统工作正常！")
    print("=" * 70)


if __name__ == "__main__":
    demo_conversation()
