"""
具身智能仿真环境
支持机器人导航、操作等任务的仿真
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class TaskType(Enum):
    """任务类型"""
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    EXPLORATION = "exploration"
    OBJECT_RECOGNITION = "object_recognition"


@dataclass
class RobotState:
    """机器人状态"""
    position: np.ndarray
    orientation: float
    velocity: np.ndarray
    sensors: Dict[str, np.ndarray]


@dataclass
class EnvironmentObject:
    """环境物体"""
    id: str
    type: str
    position: np.ndarray
    size: Tuple[float, float]
    color: str


class EmbodiedSimulator:
    """
    具身智能仿真环境

    支持：
    1. 2D/3D 网格世界
    2. 机器人导航
    3. 物体操作
    4. 传感器模拟
    """

    def __init__(self, world_size: Tuple[int, int] = (100, 100)):
        self.world_size = world_size
        self.grid = np.zeros(world_size)

        self.robot_state = RobotState(
            position=np.array([50.0, 50.0]),
            orientation=0.0,
            velocity=np.array([0.0, 0.0]),
            sensors={}
        )

        self.objects: List[EnvironmentObject] = []
        self.target_position: Optional[np.ndarray] = None
        self.obstacles: List[EnvironmentObject] = []

        self.step_count = 0
        self.max_steps = 1000

        self._init_environment()

    def _init_environment(self):
        """初始化环境"""
        self._generate_obstacles(10)
        self._generate_targets(3)
        self._update_sensors()

    def _generate_obstacles(self, num_obstacles: int):
        """生成障碍物"""
        for i in range(num_obstacles):
            pos = np.array([
                random.uniform(10, self.world_size[0] - 10),
                random.uniform(10, self.world_size[1] - 10)
            ])

            size = (random.uniform(5, 15), random.uniform(5, 15))

            self.obstacles.append(EnvironmentObject(
                id=f"obstacle_{i}",
                type="wall",
                position=pos,
                size=size,
                color="gray"
            ))

    def _generate_targets(self, num_targets: int):
        """生成目标"""
        for i in range(num_targets):
            pos = np.array([
                random.uniform(5, self.world_size[0] - 5),
                random.uniform(5, self.world_size[1] - 5)
            ])

            if self.target_position is None:
                self.target_position = pos

            self.objects.append(EnvironmentObject(
                id=f"target_{i}",
                type="target",
                position=pos,
                size=(3, 3),
                color="green"
            ))

    def _update_sensors(self):
        """更新传感器数据"""
        pos = self.robot_state.position

        lidar = np.zeros(360)
        for angle in range(360):
            distance = self._raycast(pos, np.radians(angle))
            lidar[angle] = distance

        self.robot_state.sensors = {
            "lidar": lidar,
            "position": pos.copy(),
            "orientation": self.robot_state.orientation,
            "velocity": self.robot_state.velocity.copy()
        }

    def _raycast(self, start: np.ndarray, angle: float, max_distance: float = 100.0) -> float:
        """射线投射检测障碍物"""
        direction = np.array([np.cos(angle), np.sin(angle)])

        for obstacle in self.obstacles:
            obstacle_pos = obstacle.position
            obstacle_size = obstacle.size

            t_min = (obstacle_pos - start - np.array(obstacle_size) / 2) / direction
            t_max = (obstacle_pos - start + np.array(obstacle_size) / 2) / direction

            t1 = np.minimum(t_min, t_max)
            t2 = np.maximum(t_min, t_max)

            t_enter = np.max(t1)
            t_exit = np.min(t2)

            if t_enter < t_exit and t_exit > 0:
                return max(0, t_enter)

        return max_distance

    def step(self, action: str, params: Optional[Dict] = None) -> Tuple[Dict, float, bool]:
        """
        执行动作

        Args:
            action: 动作类型
            params: 动作参数

        Returns:
            (下一状态, 奖励, 是否结束)
        """
        self.step_count += 1

        params = params or {}

        if action == "move_forward":
            distance = params.get("distance", 1.0)
            self._move_forward(distance)
        elif action == "turn":
            angle = params.get("angle", 0.0)
            self._turn(angle)
        elif action == "move_to":
            target = params.get("target", self.target_position)
            self._move_towards(target)
        elif action == "explore":
            self._random_exploration()

        self._update_sensors()

        reward = self._calculate_reward()
        done = self._check_done()

        observation = self.get_observation()

        return observation, reward, done

    def _move_forward(self, distance: float):
        """向前移动"""
        direction = np.array([
            np.cos(self.robot_state.orientation),
            np.sin(self.robot_state.orientation)
        ])

        new_pos = self.robot_state.position + direction * distance

        if not self._check_collision(new_pos):
            self.robot_state.position = new_pos

        self.robot_state.velocity = direction * distance

    def _turn(self, angle: float):
        """转向"""
        self.robot_state.orientation += angle

        while self.robot_state.orientation > np.pi:
            self.robot_state.orientation -= 2 * np.pi
        while self.robot_state.orientation < -np.pi:
            self.robot_state.orientation += 2 * np.pi

    def _move_towards(self, target: np.ndarray):
        """向目标移动"""
        direction = target - self.robot_state.position
        distance = np.linalg.norm(direction)

        if distance > 0.1:
            direction = direction / distance
            self.robot_state.orientation = np.arctan2(direction[1], direction[0])

            step_size = min(1.0, distance)
            new_pos = self.robot_state.position + direction * step_size

            if not self._check_collision(new_pos):
                self.robot_state.position = new_pos

    def _random_exploration(self):
        """随机探索"""
        angle_change = random.uniform(-0.5, 0.5)
        self._turn(angle_change)
        self._move_forward(random.uniform(0.5, 1.5))

    def _check_collision(self, position: np.ndarray) -> bool:
        """检查碰撞"""
        for obstacle in self.obstacles:
            obstacle_pos = obstacle.position
            obstacle_size = obstacle.size

            if (abs(position[0] - obstacle_pos[0]) < obstacle_size[0] / 2 and
                abs(position[1] - obstacle_pos[1]) < obstacle_size[1] / 2):
                return True

        return False

    def _calculate_reward(self) -> float:
        """计算奖励"""
        distance_to_target = np.linalg.norm(
            self.robot_state.position - self.target_position
        )

        reward = 0.0

        if distance_to_target < 5.0:
            reward += 10.0
        else:
            reward -= distance_to_target * 0.01

        if self._check_collision(self.robot_state.position):
            reward -= 5.0

        return reward

    def _check_done(self) -> bool:
        """检查是否结束"""
        distance_to_target = np.linalg.norm(
            self.robot_state.position - self.target_position
        )

        if distance_to_target < 5.0:
            return True

        if self.step_count >= self.max_steps:
            return True

        if self._check_collision(self.robot_state.position):
            return True

        return False

    def get_observation(self) -> Dict:
        """获取观测"""
        return {
            "position": self.robot_state.position.tolist(),
            "orientation": self.robot_state.orientation,
            "lidar": self.robot_state.sensors["lidar"].tolist(),
            "target": self.target_position.tolist() if self.target_position is not None else None,
            "step": self.step_count
        }

    def reset(self) -> Dict:
        """重置环境"""
        self.robot_state.position = np.array([50.0, 50.0])
        self.robot_state.orientation = 0.0
        self.robot_state.velocity = np.array([0.0, 0.0])

        self.step_count = 0

        self._update_sensors()

        return self.get_observation()

    def render_text(self) -> str:
        """文本渲染"""
        pos = self.robot_state.position
        target = self.target_position

        output = []
        output.append(f"步数: {self.step_count}")
        output.append(f"位置: ({pos[0]:.1f}, {pos[1]:.1f})")
        output.append(f"朝向: {np.degrees(self.robot_state.orientation):.1f}°")

        if target is not None:
            distance = np.linalg.norm(pos - target)
            output.append(f"目标距离: {distance:.1f}")

        output.append(f"障碍物数量: {len(self.obstacles)}")

        return "\n".join(output)
