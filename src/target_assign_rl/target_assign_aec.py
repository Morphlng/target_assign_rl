import time

import gymnasium as gym
import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

from target_assign_rl.viz import create_visualizer


class TaskAllocationAEC(AECEnv):
    metadata = {
        "render_modes": ["human", "building", "grid", None],
        "name": "task_allocation_v0",
    }

    def __init__(self, config: dict = None):
        """Task allocation environment modelled in Agent-Environment-Cycle (AEC) paradigm.

        Args:
            config (dict, optional): Configuration dictionary. Available keys are as follows.

            - min_drones (int) : The minimal amount of drones per episode. Defaults to 20.
            - max_drones (int) : The maximal amount of drones per episode. Defaults to 20.
            - num_threats (int) : The maximal amount of threats per episode. Defaults to 20.
            - attack_prob (float) : The probability of a drone successfully attacking a threat. Defaults to 0.7.
            - dict_obs (bool) : Type of observation. Defaults to False (array).
            - possible_level (list) : List of possible threat probabilities. Defaults to [0, 0.2, 0.4, 0.6, 0.8].
            - threat_dist (list) : List of threat distribution. Defaults to random.
            - render_mode (str) : Rendering mode.

        Raises:
            ValueError: Length of threat_probs must be equal to length of threat_levels
        """
        super().__init__()
        self.timestamps = time.strftime("%y%m%d_%H%M", time.localtime())
        self.config = config or {}

        # Configurable parameters
        self.max_drones = self.config.get("max_drones", 20)
        self.min_drones = self.config.get("min_drones", 20)
        self.num_threats = self.config.get("num_threats", 20)
        self.attack_prob = self.config.get("attack_prob", 0.7)
        self.dict_obs = self.config.get("dict_obs", False)
        self.possible_level = self.config.get("possible_level", [0, 0.2, 0.4, 0.6, 0.8])
        self.threat_dist = self.config.get("threat_dist", None)
        self.render_mode = self.config.get("render_mode", None)
        self.render_config = self.config.get("render_config", {})
        self._limit = self.config.get("_limit", self.max_drones)

        # Validation for threat distribution
        if self.threat_dist is not None and (
            len(self.threat_dist) != len(self.possible_level)
        ):
            raise ValueError(
                "Length of threat_probs must be equal to length of threat_levels"
            )

        self.possible_agents = [f"drone_{i}" for i in range(self.max_drones)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.max_drones)))
        )

        # Action and observation spaces
        self.action_spaces = {
            agent: gym.spaces.Discrete(self.num_threats)
            for agent in self.possible_agents
        }

        if self.dict_obs:
            self.observation_spaces = {
                agent: gym.spaces.Dict(
                    {
                        "threat_info": gym.spaces.Box(
                            low=0, high=1, shape=(self.num_threats,), dtype=np.float32
                        ),
                        "pre_allocation": gym.spaces.Box(
                            low=0,
                            high=self.max_drones,
                            shape=(self.num_threats,),
                            dtype=np.int32,
                        ),
                        "current_allocation": gym.spaces.Box(
                            low=0,
                            high=self.max_drones,
                            shape=(self.num_threats,),
                            dtype=np.int32,
                        ),
                    }
                )
                for agent in self.possible_agents
            }
        else:
            self.observation_spaces = {
                agent: gym.spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(self.num_threats * 3,),
                    dtype=np.float32,
                )
                for agent in self.possible_agents
            }

        # Internal state variables
        self.episode_count = 0
        self.num_drones = 0
        self.threat_positions = None
        self.visualizer = create_visualizer(self.render_mode, self.render_config)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Reset environment state
        self.episode_count += 1
        self.num_drones = np.random.randint(self.min_drones, self.max_drones + 1)
        self.agents = self.possible_agents[: self.num_drones]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.eliminated_threats = np.zeros(self.num_threats, dtype=bool)
        self.step_left = self.num_drones
        self.flying_drones = []

        # Random initialize threats
        self.threat_levels = np.zeros(self.num_threats)
        while not self.threat_levels.any():
            if self.threat_dist is None:
                raw_probs = np.random.random(len(self.possible_level))
                threat_probs = raw_probs / np.sum(raw_probs)
            else:
                threat_probs = self.threat_dist

            self.threat_levels = np.random.choice(
                self.possible_level,
                size=self.num_threats,
                p=threat_probs,
            )

        self.threat_levels = np.sort(self.threat_levels)[::-1]
        self.actual_threats = np.random.random(self.num_threats) < self.threat_levels
        self.num_actual_threat = np.sum(self.actual_threats)

        self.current_allocation = np.zeros(self.num_threats, dtype=int)
        self.pre_allocation = self.calculate_pre_allocation()
        self.actual_allocation = np.zeros(self.num_threats, dtype=int)
        self.allocation_map = {}

        self._initialize_threat_positions()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        # Get target position (center of the window)
        target_idx = action
        target_center = self.threat_positions[target_idx]

        # Calculate offset for the new drone based on current allocation
        drone_count = self.current_allocation[target_idx]
        offset = self._calculate_drone_offset(drone_count)
        target_pos = target_center + offset

        # Add new flying drone
        drone = {
            "id": self.agent_selection,
            "target_pos": target_pos,
            "current_pos": np.array([0, 0]),
            "arrived": False,
            "status": "active",
        }
        self.flying_drones.append(drone)
        self._update_flying_drones()

        # Record allocation
        self.step_left -= 1
        self.current_allocation[action] += 1
        self.allocation_map[self.agent_selection] = action

        # Check if all drones are assigned
        if self._agent_selector.is_last():
            self._simulate_engagement()
            self.rewards = self._calculate_rewards()
            for drone in self.flying_drones:
                agent = drone["id"]
                if self.terminations[agent]:
                    drone["status"] = "destroyed"
                elif self.truncations[agent]:
                    drone["status"] = "damaged"
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

        return (
            self.observe(self.agent_selection),
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def observe(self, agent):
        if self.dict_obs:
            return {
                "threat_info": self.threat_levels.copy(),
                "pre_allocation": self.pre_allocation.copy(),
                "current_allocation": self.current_allocation.copy(),
            }
        else:
            return self.state()

    def state(self):
        return np.concatenate(
            [self.threat_levels, self.pre_allocation, self.current_allocation],
            dtype=np.float32,
        )

    def action_space(self, agent):
        return self.action_spaces[agent]

    def action_mask(self, agent):
        threats = (self.threat_levels > 0).astype(np.bool_)
        redundant = (self.current_allocation > self._limit).astype(np.bool_)
        return (threats & ~redundant).astype(np.int8)

    def calculate_pre_allocation(self):
        pre_allocation = np.zeros(self.num_threats, dtype=int)
        remaining_drones = self.num_drones

        # Allocate one drone to each non-zero threat
        for i, threat in enumerate(self.threat_levels):
            if threat > 0 and remaining_drones > 0:
                pre_allocation[i] += 1
                remaining_drones -= 1

        # Allocate remaining drones to highest threats
        while remaining_drones > 0:
            for i in range(self.num_threats):
                if self.threat_levels[i] > 0 and remaining_drones > 0:
                    pre_allocation[i] += 1
                    remaining_drones -= 1
                if remaining_drones == 0:
                    break

        return pre_allocation

    def get_env_state(self):
        state = vars(self).copy()
        state["is_first_agent"] = self._agent_selector.is_first()
        return state

    def _simulate_engagement(self):
        for agent in self.allocation_map:
            if not (self.terminations[agent] or self.truncations[agent]):
                self.actual_allocation[self.allocation_map[agent]] += 1

        self.drone_cost = np.zeros(self.num_threats, dtype=int)
        for agent, pos in self.allocation_map.items():
            if not (self.terminations[agent] or self.truncations[agent]):
                if self.actual_threats[pos] and not self.eliminated_threats[pos]:
                    self.terminations[agent] = True
                    self.drone_cost[pos] += 1
                    if np.random.random() < self.attack_prob:
                        self.eliminated_threats[pos] = True

    def _calculate_rewards(self):
        covered_threats = (self.current_allocation > 0)[self.actual_threats]
        actual_coverage = (
            (np.sum(covered_threats) / self.num_actual_threat)
            if self.num_actual_threat > 0
            else 1
        )

        # coverage by threats
        weighted_coverage = np.sum(
            self.current_allocation * self.threat_levels
        ) / np.sum(self.threat_levels)

        # success rate
        threats_destroyed = np.sum(self.eliminated_threats)
        drones_lost = sum(self.terminations.values())
        success_rate = (
            (threats_destroyed / self.num_actual_threat)
            if self.num_actual_threat > 0
            else 0
        )

        # kill reward by threats
        destroy_reward = np.sum(self.eliminated_threats * self.threat_levels)

        # remain penalty
        remaining_penalty = np.sum(
            self.threat_levels[self.actual_threats & ~self.eliminated_threats]
        ) / (np.sum(self.threat_levels[self.actual_threats]) + 1e-8)

        # overall reward
        total_reward = (
            0.75 * weighted_coverage + success_rate + destroy_reward - remaining_penalty
        )

        self.infos = {
            agent: {
                "coverage": actual_coverage,
                "success_rate": success_rate,
                "threat_destroyed": threats_destroyed,
                "drone_lost": drones_lost,
                "kd_ratio": threats_destroyed / (drones_lost + 1e-8),
                "num_remaining_threat": self.num_actual_threat - threats_destroyed,
            }
            for agent in self.agents
        }

        return {agent: total_reward for agent in self.agents}

    def render(self):
        if self.visualizer is None:
            return

        self.visualizer.render(self.get_env_state())
        if self.render_config.get("save_screenshot", False):
            self.visualizer.save_screenshot(
                self.timestamps, self.episode_count, self.agent_selection
            )

    def close(self):
        if self.visualizer:
            self.visualizer.close()

    def _initialize_threat_positions(self):
        """Initialize the window positions and threat positions (centers)."""
        rows, cols = self.num_threats // 5, 5
        x_padding, y_padding = 250, 80  # Padding around the building
        grid_width = self.render_config.get("width", 1400) - 2 * x_padding
        grid_height = self.render_config.get("height", 800) - 2 * y_padding
        cell_width = grid_width // cols
        cell_height = grid_height // rows

        self.windows = []
        self.threat_positions = []

        for row in range(rows):
            for col in range(cols):
                # Calculate window position
                x = x_padding + col * cell_width
                y = y_padding + row * cell_height
                rect = pygame.Rect(x, y, cell_width, cell_height)
                self.windows.append(rect)

                # Calculate threat position (center of the window)
                center_x = x + cell_width // 2
                center_y = y + cell_height // 2
                self.threat_positions.append((center_x, center_y))

        # Convert to a NumPy array for easier indexing and operations
        self.threat_positions = np.array(self.threat_positions)

    def _calculate_drone_offset(self, drone_count):
        """Calculate offset for a drone based on the number of drones assigned to the same window."""
        base_radius = 30
        radius_increment = 20

        # Calculate which layer the drone is in
        layer = 0
        drones_in_layer = 6
        total_drones_in_previous_layers = 0

        while drone_count >= total_drones_in_previous_layers + drones_in_layer:
            layer += 1
            total_drones_in_previous_layers += drones_in_layer
            drones_in_layer = 6 + layer * 6

        # Determine the position of the drone within the current layer
        position_in_layer = drone_count - total_drones_in_previous_layers

        # Calculate the radius of the current layer
        radius = base_radius + layer * radius_increment

        # Calculate the angle for the drone's position within the layer
        angle = (2 * np.pi / drones_in_layer) * position_in_layer
        offset_x = int(radius * np.cos(angle))
        offset_y = int(radius * np.sin(angle))

        return np.array([offset_x, offset_y])

    def _update_flying_drones(self):
        for drone in self.flying_drones:
            if drone["arrived"]:
                continue

            target_pos = np.array(drone["target_pos"])
            current_pos = drone["current_pos"]
            direction = target_pos - current_pos
            dist = np.linalg.norm(direction)
            direction = direction / dist
            move_distance = dist / max(self.step_left, 1)
            drone["current_pos"] = drone["current_pos"] + direction * move_distance
            if np.allclose(drone["current_pos"], drone["target_pos"], atol=1):
                drone["arrived"] = True


def raw_env(config: dict = None) -> TaskAllocationAEC:
    """Create an environment with the given configuration.

    Args:
        config (dict, optional): Configuration dictionary. See environment documentation for details.

    Returns:
        TaskAllocationEnv: Wrapped environment.
    """
    env = TaskAllocationAEC(config)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


if __name__ == "__main__":
    env = raw_env(
        dict(
            render_mode="human",
            render_config={"pause_at_end": True, "save_screenshot": False},
        )
    )

    for _ in range(20):
        env.reset()
        for agent in env.agents:
            obs, reward, te, tr, info = env.last()
            if te or tr:
                action = None
            else:
                action = np.random.choice(np.where(env.action_mask(agent))[0])
            env.step(action)
            env.render()
            pygame.time.wait(33)

    env.close()
