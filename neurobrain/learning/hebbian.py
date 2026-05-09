"""
HebbianLearning - 赫布学习规则
"一起发放的神经元连接在一起"
实现经典和现代赫布学习变体
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HebbianVariant(Enum):
    CLASSIC = "classic"
    OJA = "oja"
    SANGER = "sanger"
    BCM = "bcm"
    COVARIANCE = "covariance"


@dataclass
class HebbianConfig:
    """赫布学习配置"""
    learning_rate: float = 0.01
    variant: HebbianVariant = HebbianVariant.CLASSIC
    decay_rate: float = 0.001
    weight_decay: float = 0.0001
    bcm_threshold: float = 0.5
    oja_beta: float = 0.1


class HebbianLearning:
    """
    赫布学习规则
    
    实现多种赫布学习变体：
    - 经典赫布：Δw = η * x * y
    - Oja规则：Δw = η * y * (x - w * y)
    - Sanger规则：用于PCA
    - BCM规则：滑动阈值
    - 协方差规则：基于均值
    """
    
    def __init__(self, config: Optional[HebbianConfig] = None):
        self.config = config or HebbianConfig()
        
        self._mean_pre: Optional[np.ndarray] = None
        self._mean_post: Optional[np.ndarray] = None
        self._bcm_threshold: Optional[np.ndarray] = None
        self._update_count = 0
        
    def compute_weight_update(
        self,
        weights: np.ndarray,
        pre_activity: np.ndarray,
        post_activity: np.ndarray,
        modulatory_signal: float = 1.0
    ) -> np.ndarray:
        """
        计算权重更新
        
        Args:
            weights: 当前权重矩阵
            pre_activity: 突触前活动
            post_activity: 突触后活动
            modulatory_signal: 调制信号（如多巴胺）
            
        Returns:
            权重更新量
        """
        self._update_count += 1
        
        if self._mean_pre is None:
            self._mean_pre = np.zeros_like(pre_activity)
        if self._mean_post is None:
            self._mean_post = np.zeros_like(post_activity)
        
        alpha = 0.01
        self._mean_pre = (1 - alpha) * self._mean_pre + alpha * pre_activity
        self._mean_post = (1 - alpha) * self._mean_post + alpha * post_activity
        
        if self.config.variant == HebbianVariant.CLASSIC:
            delta_w = self._classic_hebbian(pre_activity, post_activity)
        
        elif self.config.variant == HebbianVariant.OJA:
            delta_w = self._oja_rule(weights, pre_activity, post_activity)
        
        elif self.config.variant == HebbianVariant.SANGER:
            delta_w = self._sanger_rule(weights, pre_activity, post_activity)
        
        elif self.config.variant == HebbianVariant.BCM:
            delta_w = self._bcm_rule(weights, pre_activity, post_activity)
        
        elif self.config.variant == HebbianVariant.COVARIANCE:
            delta_w = self._covariance_rule(pre_activity, post_activity)
        
        else:
            delta_w = self._classic_hebbian(pre_activity, post_activity)
        
        delta_w *= modulatory_signal
        
        delta_w -= self.config.weight_decay * weights
        
        return delta_w
    
    def _classic_hebbian(
        self, 
        pre: np.ndarray, 
        post: np.ndarray
    ) -> np.ndarray:
        """经典赫布规则"""
        return self.config.learning_rate * np.outer(post, pre)
    
    def _oja_rule(
        self, 
        weights: np.ndarray,
        pre: np.ndarray, 
        post: np.ndarray
    ) -> np.ndarray:
        """Oja规则（带权重归一化）"""
        hebbian = np.outer(post, pre)
        
        normalization = self.config.oja_beta * np.outer(post ** 2, weights.sum(axis=1))
        
        return self.config.learning_rate * (hebbian - normalization)
    
    def _sanger_rule(
        self, 
        weights: np.ndarray,
        pre: np.ndarray, 
        post: np.ndarray
    ) -> np.ndarray:
        """Sanger规则（用于PCA）"""
        delta_w = np.zeros_like(weights)
        
        for j in range(len(post)):
            hebbian = post[j] * pre
            
            sum_term = np.zeros_like(pre)
            for k in range(j + 1):
                sum_term += weights[k] * post[k]
            
            delta_w[j] = self.config.learning_rate * (hebbian - post[j] * sum_term)
        
        return delta_w
    
    def _bcm_rule(
        self, 
        weights: np.ndarray,
        pre: np.ndarray, 
        post: np.ndarray
    ) -> np.ndarray:
        """BCM规则（滑动阈值）"""
        if self._bcm_threshold is None:
            self._bcm_threshold = np.ones(len(post)) * self.config.bcm_threshold
        
        threshold_update = 0.01 * (post ** 2 - self._bcm_threshold)
        self._bcm_threshold = np.maximum(0.1, self._bcm_threshold + threshold_update)
        
        phi = post * (post - self._bcm_threshold)
        
        return self.config.learning_rate * np.outer(phi, pre)
    
    def _covariance_rule(
        self, 
        pre: np.ndarray, 
        post: np.ndarray
    ) -> np.ndarray:
        """协方差规则"""
        pre_centered = pre - self._mean_pre
        post_centered = post - self._mean_post
        
        return self.config.learning_rate * np.outer(post_centered, pre_centered)
    
    def apply_update(
        self, 
        weights: np.ndarray,
        delta_w: np.ndarray,
        w_min: float = 0.0,
        w_max: float = 1.0
    ) -> np.ndarray:
        """
        应用权重更新
        
        Args:
            weights: 当前权重
            delta_w: 权重更新
            w_min: 最小权重
            w_max: 最大权重
            
        Returns:
            更新后的权重
        """
        new_weights = weights + delta_w
        return np.clip(new_weights, w_min, w_max)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'update_count': self._update_count,
            'variant': self.config.variant.value,
            'learning_rate': self.config.learning_rate
        }
    
    def reset(self):
        """重置学习状态"""
        self._mean_pre = None
        self._mean_post = None
        self._bcm_threshold = None
        self._update_count = 0
