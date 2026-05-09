"""
具身智能训练环境
提供真实的机器人控制任务，用于测试NeuroBrain框架
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import math


class TaskType(Enum):
    NAVIGATION = "navigation"
    MAZE = "maze"
    OBJECT_MANIPULATION = "object_manipulation"


@dataclass
class RobotState:
    """机器人状态"""
    position: np.ndarray
    velocity: np.ndarray
    orientation: float
    sensors: np.ndarray


@dataclass
class EnvironmentState:
    """环境状态"""
    robot: RobotState
    target_position: np.ndarray
    obstacles: List[np.ndarray]
    reward: float
    done: bool
    info: Dict


class EmbodiedEnvironment:
    """
    具身智能仿真环境

    支持多种任务：
    1. 导航任务：移动到目标位置
    2. 迷宫任务：在迷宫中找到出口
    3. 物体操控：推动物体到目标位置
    """

    def __init__(
        self,
        task: TaskType = TaskType.NAVIGATION,
        world_size: Tuple[float, float] = (10.0, 10.0),
        sensor_range: float = 3.0,
        max_steps: int = 200
    ):
        self.task = task
        self.world_size = world_size
        self.sensor_range = sensor_range
        self.max_steps = max_steps

        self.robot = None
        self.target = None
        self.obstacles = []
        self.step_count = 0

        self._initialize()

    def _initialize(self):
        """初始化环境"""
        if self.task == TaskType.NAVIGATION:
            self._init_navigation()
        elif self.task == TaskType.MAZE:
            self._init_maze()
        elif self.task == TaskType.OBJECT_MANIPULATION:
            self._init_manipulation()

    def _init_navigation(self):
        """初始化导航任务"""
        self.robot = RobotState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            orientation=0.0,
            sensors=np.zeros(8)
        )
        self.target = np.array([
            np.random.uniform(2.0, self.world_size[0] - 2.0),
            np.random.uniform(2.0, self.world_size[1] - 2.0)
        ])
        self.obstacles = self._generate_obstacles(3)

    def _init_maze(self):
        """初始化迷宫任务"""
        self.robot = RobotState(
            position=np.array([0.5, 0.5]),
            velocity=np.array([0.0, 0.0]),
            orientation=0.0,
            sensors=np.zeros(16)
        )
        self.target = np.array([self.world_size[0] - 0.5, self.world_size[1] - 0.5])
        self.obstacles = self._generate_maze_walls()

    def _init_manipulation(self):
        """初始化物体操控任务"""
        self.robot = RobotState(
            position=np.array([5.0, 5.0]),
            velocity=np.array([0.0, 0.0]),
            orientation=0.0,
            sensors=np.zeros(8)
        )
        self.target = np.array([8.0, 8.0])
        self.movable_object = np.array([3.0, 3.0])
        self.obstacles = []

    def _generate_obstacles(self, count: int) -> List[np.ndarray]:
        """生成随机障碍物"""
        obstacles = []
        for _ in range(count):
            obs = np.array([
                np.random.uniform(2.0, self.world_size[0] - 2.0),
                np.random.uniform(2.0, self.world_size[1] - 2.0)
            ])
            if np.linalg.norm(obs - self.robot.position) > 2.0:
                obstacles.append(obs)
        return obstacles

    def _generate_maze_walls(self) -> List[np.ndarray]:
        """生成迷宫墙壁"""
        walls = []
        grid_size = 2.0

        for i in range(int(self.world_size[0] / grid_size)):
            for j in range(int(self.world_size[1] / grid_size)):
                if (i + j) % 3 == 0 and i > 0 and j > 0:
                    walls.append(np.array([i * grid_size, j * grid_size]))

        return walls

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.step_count = 0
        self._initialize()
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作

        Args:
            action: [linear_velocity, angular_velocity] 或 [dx, dy]

        Returns:
            observation, reward, done, info
        """
        self.step_count += 1

        if len(action) == 2:
            linear_vel = np.clip(action[0], -1.0, 1.0)
            angular_vel = np.clip(action[1], -1.0, 1.0)
            self._apply_action(linear_vel, angular_vel)
        else:
            dx = np.clip(action[0], -0.5, 0.5)
            dy = np.clip(action[1], -0.5, 0.5)
            self.robot.position[0] += dx
            self.robot.position[1] += dy

        self.robot.position = np.clip(
            self.robot.position,
            0,
            np.array(self.world_size)
        )

        self._update_sensors()

        reward, done, info = self._compute_reward()

        return self._get_observation(), reward, done, info

    def _apply_action(self, linear: float, angular: float):
        """应用动作"""
        dt = 0.1

        self.robot.orientation += angular * dt
        self.robot.orientation = self.robot.orientation % (2 * math.pi)

        self.robot.velocity[0] = linear * math.cos(self.robot.orientation)
        self.robot.velocity[1] = linear * math.sin(self.robot.orientation)

        new_pos = self.robot.position + self.robot.velocity * dt

        if not self._check_collision(new_pos):
            self.robot.position = new_pos

    def _check_collision(self, pos: np.ndarray) -> bool:
        """检查碰撞"""
        for obs in self.obstacles:
            if np.linalg.norm(pos - obs) < 0.5:
                return True

        if pos[0] < 0 or pos[0] > self.world_size[0]:
            return True
        if pos[1] < 0 or pos[1] > self.world_size[1]:
            return True

        return False

    def _update_sensors(self):
        """更新传感器"""
        if self.task == TaskType.NAVIGATION or self.task == TaskType.MAZE:
            self._update_proximity_sensors()
        elif self.task == TaskType.OBJECT_MANIPULATION:
            self._update_object_sensors()

    def _update_proximity_sensors(self):
        """更新接近传感器"""
        num_sensors = 8 if self.task == TaskType.NAVIGATION else 16
        angle_step = 2 * math.pi / num_sensors

        sensors = np.zeros(num_sensors)

        for i in range(num_sensors):
            angle = self.robot.orientation + i * angle_step
            direction = np.array([math.cos(angle), math.sin(angle)])

            distance = self.sensor_range
            for dist in np.linspace(0.1, self.sensor_range, 30):
                test_pos = self.robot.position + direction * dist

                if self._check_collision(test_pos):
                    distance = dist
                    break

                dist_to_target = np.linalg.norm(test_pos - self.target)
                if dist_to_target < 0.5:
                    sensors[i] = 1.0

            sensors[i] = 1.0 - distance / self.sensor_range

        self.robot.sensors = sensors

    def _update_object_sensors(self):
        """更新物体感知传感器"""
        dist_to_object = np.linalg.norm(self.robot.position - self.movable_object)
        dist_to_target = np.linalg.norm(self.movable_object - self.target)

        self.robot.sensors[0] = 1.0 - dist_to_object / self.sensor_range

        direction_to_object = (self.movable_object - self.robot.position) / (dist_to_object + 1e-6)
        self.robot.sensors[1] = np.dot(
            [math.cos(self.robot.orientation), math.sin(self.robot.orientation)],
            direction_to_object
        )

        for i in range(6):
            angle = self.robot.orientation + (i - 2) * math.pi / 6
            direction = np.array([math.cos(angle), math.sin(angle)])
            distance = self.sensor_range

            for dist in np.linspace(0.1, self.sensor_range, 20):
                test_pos = self.robot.position + direction * dist
                if self._check_collision(test_pos):
                    distance = dist
                    break

            self.robot.sensors[i + 2] = 1.0 - distance / self.sensor_range

    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """计算奖励"""
        done = self.step_count >= self.max_steps
        reward = -0.01
        info = {}

        if self.task == TaskType.NAVIGATION or self.task == TaskType.MAZE:
            dist_to_target = np.linalg.norm(self.robot.position - self.target)
            reward += -dist_to_target * 0.1

            if dist_to_target < 0.5:
                reward = 100.0
                done = True
                info['success'] = True

            min_obstacle_dist = min(
                np.linalg.norm(self.robot.position - obs) if len(self.obstacles) > 0 else 100
                for obs in self.obstacles
            ) if self.obstacles else 100

            if min_obstacle_dist < 0.5:
                reward = -10.0
                done = True
                info['collision'] = True

        elif self.task == TaskType.OBJECT_MANIPULATION:
            dist_object_target = np.linalg.norm(self.movable_object - self.target)
            reward += -dist_object_target * 0.1

            dist_robot_object = np.linalg.norm(self.robot.position - self.movable_object)
            if dist_robot_object < 1.0:
                direction = (self.target - self.movable_object) / (dist_object_target + 1e-6)
                self.movable_object += direction * 0.1

            if dist_object_target < 0.5:
                reward = 100.0
                done = True
                info['success'] = True

        info['distance_to_goal'] = np.linalg.norm(self.robot.position - self.target)
        info['steps'] = self.step_count

        return reward, done, info

    def _get_observation(self) -> np.ndarray:
        """获取观测"""
        obs = np.zeros(20)

        obs[0:2] = self.robot.position / np.array(self.world_size)
        obs[2] = self.robot.orientation / (2 * math.pi)

        min_len = min(len(self.robot.sensors), 8)
        obs[3:3+min_len] = self.robot.sensors[:min_len]

        obs[11:13] = self.target / np.array(self.world_size)

        if len(self.obstacles) > 0:
            nearest_obs = min(self.obstacles, key=lambda o: np.linalg.norm(self.robot.position - o))
            obs[13:15] = nearest_obs / np.array(self.world_size)

        return obs

    def render(self) -> np.ndarray:
        """渲染环境"""
        canvas = np.zeros((int(self.world_size[1] * 20), int(self.world_size[0] * 20), 3))

        robot_pix = (self.robot.position * 20).astype(int)
        canvas = self._draw_circle(canvas, robot_pix[0], robot_pix[1], 10, (0, 1, 0))

        target_pix = (self.target * 20).astype(int)
        canvas = self._draw_circle(canvas, target_pix[0], target_pix[1], 15, (1, 0, 0))

        for obs in self.obstacles:
            obs_pix = (obs * 20).astype(int)
            canvas = self._draw_circle(canvas, obs_pix[0], obs_pix[1], 8, (0.5, 0.5, 0.5))

        return canvas

    def _draw_circle(self, canvas: np.ndarray, x: int, y: int, r: int, color: Tuple) -> np.ndarray:
        """绘制圆形"""
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = x + dx, y + dy
                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                        canvas[py, px] = color
        return canvas


def create_environment(task: str = "navigation") -> EmbodiedEnvironment:
    """创建环境工厂函数"""
    task_map = {
        "navigation": TaskType.NAVIGATION,
        "maze": TaskType.MAZE,
        "manipulation": TaskType.OBJECT_MANIPULATION
    }
    task_type = task_map.get(task.lower(), TaskType.NAVIGATION)
    return EmbodiedEnvironment(task=task_type)
