"""
Amygdala - 杏仁核模块
负责情感处理、威胁检测和记忆的情感增强
模拟人脑杏仁核的核心功能
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class EmotionType(Enum):
    FEAR = "fear"
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


@dataclass
class EmotionalState:
    """情感状态"""
    valence: float = 0.0
    arousal: float = 0.5
    dominant_emotion: EmotionType = EmotionType.NEUTRAL
    emotion_intensities: Dict[EmotionType, float] = field(default_factory=dict)


class Amygdala:
    """
    杏仁核模块
    
    功能：
    1. 情感评估：评估输入的情感价值
    2. 威胁检测：识别潜在威胁
    3. 记忆增强：情感增强记忆编码
    4. 情感学习：情感条件反射
    """
    
    def __init__(
        self,
        emotional_weight: float = 0.3,
        threat_threshold: float = 0.7
    ):
        self.emotional_weight = emotional_weight
        self.threat_threshold = threat_threshold
        
        self._emotion_prototypes = self._init_emotion_prototypes()
        self._threat_patterns = self._init_threat_patterns()
        
        self._current_state = EmotionalState()
        self._emotion_history = deque(maxlen=100)
        
        self._conditioned_responses: Dict[str, float] = {}
        self._arousal_baseline = 0.5
        self._valence_baseline = 0.0
        
    def process(
        self, 
        input_data: np.ndarray,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        处理输入并生成情感响应
        
        Args:
            input_data: 输入数据
            context: 上下文信息
            
        Returns:
            情感响应字典
        """
        emotion_scores = self._evaluate_emotions(input_data)
        
        threat_level = self._detect_threat(input_data)
        
        valence, arousal = self._compute_valence_arousal(emotion_scores, threat_level)
        
        dominant_emotion = self._determine_dominant_emotion(emotion_scores)
        
        self._update_emotional_state(valence, arousal, dominant_emotion, emotion_scores)
        
        emotional_weight = self._compute_emotional_weight(
            valence, arousal, threat_level
        )
        
        self._emotion_history.append({
            'valence': valence,
            'arousal': arousal,
            'dominant_emotion': dominant_emotion.value,
            'threat_level': threat_level
        })
        
        return {
            'emotion_scores': emotion_scores,
            'valence': valence,
            'arousal': arousal,
            'dominant_emotion': dominant_emotion.value,
            'threat_level': threat_level,
            'weight': emotional_weight,
            'should_enhance_memory': arousal > 0.6 or threat_level > self.threat_threshold
        }
    
    def update_emotional_state(self, reward: float):
        """
        根据奖励更新情感状态
        
        Args:
            reward: 奖励信号
        """
        if reward > 0:
            self._current_state.valence = min(1.0, self._current_state.valence + 0.1)
            self._current_state.emotion_intensities[EmotionType.JOY] = min(
                1.0, 
                self._current_state.emotion_intensities.get(EmotionType.JOY, 0) + 0.1
            )
        elif reward < 0:
            self._current_state.valence = max(-1.0, self._current_state.valence - 0.1)
            self._current_state.emotion_intensities[EmotionType.FEAR] = min(
                1.0, 
                self._current_state.emotion_intensities.get(EmotionType.FEAR, 0) + 0.1
            )
        
        self._current_state.arousal = np.clip(
            self._current_state.arousal + 0.1 * abs(reward),
            0.0, 1.0
        )
    
    def create_conditioned_response(
        self, 
        pattern_id: str, 
        response_strength: float
    ):
        """
        创建条件反射
        
        Args:
            pattern_id: 模式ID
            response_strength: 响应强度
        """
        self._conditioned_responses[pattern_id] = response_strength
    
    def get_conditioned_response(self, pattern_id: str) -> float:
        """获取条件反射强度"""
        return self._conditioned_responses.get(pattern_id, 0.0)
    
    def get_current_state(self) -> EmotionalState:
        """获取当前情感状态"""
        return self._current_state
    
    def get_emotion_history(self, n: int = 10) -> List[Dict]:
        """获取情感历史"""
        return list(self._emotion_history)[-n:]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'current_valence': self._current_state.valence,
            'current_arousal': self._current_state.arousal,
            'dominant_emotion': self._current_state.dominant_emotion.value,
            'conditioned_responses': len(self._conditioned_responses),
            'emotion_history_length': len(self._emotion_history)
        }
    
    def get_state(self) -> Dict:
        """获取状态"""
        return {
            'current_state': {
                'valence': self._current_state.valence,
                'arousal': self._current_state.arousal,
                'dominant_emotion': self._current_state.dominant_emotion.value,
                'emotion_intensities': {
                    k.value: v for k, v in self._current_state.emotion_intensities.items()
                }
            },
            'conditioned_responses': self._conditioned_responses.copy(),
            'arousal_baseline': self._arousal_baseline,
            'valence_baseline': self._valence_baseline
        }
    
    def set_state(self, state: Dict):
        """设置状态"""
        cs = state['current_state']
        self._current_state = EmotionalState(
            valence=cs['valence'],
            arousal=cs['arousal'],
            dominant_emotion=EmotionType(cs['dominant_emotion']),
            emotion_intensities={
                EmotionType(k): v for k, v in cs['emotion_intensities'].items()
            }
        )
        self._conditioned_responses = state['conditioned_responses']
        self._arousal_baseline = state['arousal_baseline']
        self._valence_baseline = state['valence_baseline']
    
    def _init_emotion_prototypes(self) -> Dict[EmotionType, np.ndarray]:
        """初始化情感原型"""
        np.random.seed(42)
        prototype_dim = 50
        
        prototypes = {}
        for emotion in EmotionType:
            if emotion == EmotionType.NEUTRAL:
                prototypes[emotion] = np.zeros(prototype_dim)
            else:
                prototypes[emotion] = np.random.randn(prototype_dim) * 0.5
                if emotion in [EmotionType.JOY, EmotionType.SURPRISE]:
                    prototypes[emotion] = np.abs(prototypes[emotion])
                elif emotion in [EmotionType.FEAR, EmotionType.SADNESS, EmotionType.ANGER]:
                    prototypes[emotion] = -np.abs(prototypes[emotion])
        
        return prototypes
    
    def _init_threat_patterns(self) -> List[np.ndarray]:
        """初始化威胁模式"""
        np.random.seed(43)
        threat_patterns = []
        
        for _ in range(5):
            pattern = np.random.randn(50)
            pattern = np.abs(pattern) * 2
            threat_patterns.append(pattern)
        
        return threat_patterns
    
    def _evaluate_emotions(self, input_data: np.ndarray) -> Dict[EmotionType, float]:
        """评估情感"""
        scores = {}
        
        for emotion, prototype in self._emotion_prototypes.items():
            min_len = min(len(input_data), len(prototype))
            similarity = np.abs(np.corrcoef(
                input_data[:min_len], 
                prototype[:min_len]
            )[0, 1])
            
            if np.isnan(similarity):
                similarity = 0.0
            
            scores[emotion] = similarity
        
        total = sum(scores.values()) + 1e-8
        scores = {k: v / total for k, v in scores.items()}
        
        return scores
    
    def _detect_threat(self, input_data: np.ndarray) -> float:
        """检测威胁"""
        max_threat = 0.0
        
        for threat_pattern in self._threat_patterns:
            min_len = min(len(input_data), len(threat_pattern))
            correlation = np.abs(np.corrcoef(
                input_data[:min_len],
                threat_pattern[:min_len]
            )[0, 1])
            
            if np.isnan(correlation):
                correlation = 0.0
            
            max_threat = max(max_threat, correlation)
        
        return max_threat
    
    def _compute_valence_arousal(
        self, 
        emotion_scores: Dict[EmotionType, float],
        threat_level: float
    ) -> Tuple[float, float]:
        """计算效价和唤醒度"""
        positive_emotions = [EmotionType.JOY, EmotionType.SURPRISE]
        negative_emotions = [EmotionType.FEAR, EmotionType.SADNESS, EmotionType.ANGER, EmotionType.DISGUST]
        
        positive_score = sum(emotion_scores.get(e, 0) for e in positive_emotions)
        negative_score = sum(emotion_scores.get(e, 0) for e in negative_emotions)
        
        valence = positive_score - negative_score - threat_level * 0.3
        valence = np.clip(valence, -1.0, 1.0)
        
        arousal = sum(emotion_scores.values()) + threat_level * 0.5
        arousal = np.clip(arousal, 0.0, 1.0)
        
        return valence, arousal
    
    def _determine_dominant_emotion(
        self, 
        emotion_scores: Dict[EmotionType, float]
    ) -> EmotionType:
        """确定主导情感"""
        max_score = 0.0
        dominant = EmotionType.NEUTRAL
        
        for emotion, score in emotion_scores.items():
            if score > max_score:
                max_score = score
                dominant = emotion
        
        return dominant
    
    def _update_emotional_state(
        self, 
        valence: float, 
        arousal: float,
        dominant_emotion: EmotionType,
        emotion_scores: Dict[EmotionType, float]
    ):
        """更新情感状态"""
        alpha = 0.3
        
        self._current_state.valence = (1 - alpha) * self._current_state.valence + alpha * valence
        self._current_state.arousal = (1 - alpha) * self._current_state.arousal + alpha * arousal
        self._current_state.dominant_emotion = dominant_emotion
        self._current_state.emotion_intensities = emotion_scores.copy()
    
    def _compute_emotional_weight(
        self, 
        valence: float, 
        arousal: float,
        threat_level: float
    ) -> float:
        """计算情感权重"""
        base_weight = self.emotional_weight
        
        arousal_factor = arousal * 0.5
        
        valence_factor = abs(valence) * 0.3
        
        threat_factor = threat_level * 0.4 if threat_level > self.threat_threshold else 0
        
        total_weight = base_weight + arousal_factor + valence_factor + threat_factor
        
        return np.clip(total_weight, 0.0, 1.0)
