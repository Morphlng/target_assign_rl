import colorsys
import os
import sys
import time

import gymnasium as gym
import numpy as np
import pygame
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


def get_threat_color(threat_level):
    if threat_level == 0:
        return (240, 240, 240)

    hue = 0.15 - (threat_level * 0.15)
    saturation = 0.5 + (threat_level * 0.5)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return True
    return False


class TaskAllocationEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "task_allocation_v0"}

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
            - render_mode (str) : Rendering mode. Not implemented yet.

        Raises:
            ValueError: Length of threat_probs must be equal to length of threat_levels
        """
        super().__init__()
        self.timestamps = time.strftime("%y%m%d_%H%M", time.localtime())
        self.config = config or {}

        self.max_drones = self.config.get("max_drones", 20)
        self.min_drones = self.config.get("min_drones", 20)
        self.num_threats = self.config.get("num_threats", 20)
        self.attack_prob = self.config.get("attack_prob", 0.7)
        self.render_config = self.config.get("render_config", {})
        self.dict_obs = self.config.get("dict_obs", False)
        self.possible_level = self.config.get("possible_level", [0, 0.2, 0.4, 0.6, 0.8])
        self.threat_dist = self.config.get("threat_dist", None)
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

        self.episode_count = 0
        self.num_drones = 0
        self.screen = None

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

        # Random initialize threats
        self.threat_levels = np.zeros(self.num_threats)
        while not self.threat_levels.any():
            if self.threat_dist is None:
                raw_probs = np.random.random(5)
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

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        # Update allocation
        self.current_allocation[action] += 1
        self.allocation_map[self.agent_selection] = action

        # Check if all drones have been allocated
        if self._agent_selector.is_last():
            self._simulate_engagement()
            self.rewards = self._calculate_rewards()
        else:
            self._clear_rewards()

        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()

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
        return (self.threat_levels > 0).astype(np.int8)

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
        if self.screen is None:
            self._init_rendering()

        self.screen.fill((240, 240, 240))
        font = pygame.font.Font(None, 24)

        # Draw threat grid
        grid_width = self.screen_width - 300
        grid_height = self.screen_height - 100
        cell_width = grid_width // self.num_threats
        cell_height = grid_height // 3

        for i in range(self.num_threats):
            x = i * cell_width + 250

            # Threat level
            color = get_threat_color(self.threat_levels[i])
            pygame.draw.rect(self.screen, color, (x, 50, cell_width, cell_height))

            # Write threat level in the cell
            if self.render_config.get("show_threat_level"):
                threat_text = font.render(
                    f"{self.threat_levels[i]:.2f}", True, (0, 0, 0)
                )
                text_rect = threat_text.get_rect(
                    center=(x + cell_width // 2, 50 + cell_height // 2)
                )
                self.screen.blit(threat_text, text_rect)

            # Actual threat
            if self.actual_threats[i]:
                threat_surface = pygame.Surface((cell_width, cell_height))
                threat_surface.set_alpha(128)
                threat_surface.fill((100, 200, 255))
                self.screen.blit(threat_surface, (x, 50 + cell_height))

            # Success
            if self.eliminated_threats[i]:
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (x + 5, 50 + cell_height + 5),
                    (x + cell_width - 5, 50 + cell_height * 2 - 5),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (x + cell_width - 5, 50 + cell_height + 5),
                    (x + 5, 50 + cell_height * 2 - 5),
                    3,
                )

        # Allocation
        counter = np.zeros(self.num_threats)
        for agent, pos in self.allocation_map.items():
            counter[pos] = min(counter[pos] + 1, 10)
            x = pos * cell_width + 250
            color = (0, 100, 255)  # Default blue for active drones
            if self.terminations.get(agent, False):
                color = (255, 0, 0)  # Red for destroyed drones
            elif self.truncations.get(agent, False):
                color = (255, 165, 0)  # Orange for damaged drones
            pygame.draw.circle(
                self.screen,
                color,
                (x + cell_width // 2, 50 + cell_height * 2 + counter[pos] * 20 + 10),
                5,
            )

        # Draw grid lines
        for i in range(self.num_threats + 1):
            x = i * cell_width + 250
            pygame.draw.line(
                self.screen, (0, 0, 0), (x, 50), (x, 50 + cell_height * 2), 2
            )

        for i in range(3):
            y = i * cell_height + 50
            pygame.draw.line(
                self.screen, (0, 0, 0), (250, y), (self.screen_width - 50, y), 2
            )

        # Draw info text
        font = pygame.font.Font(None, 24)
        texts = [
            f"Episode: {self.episode_count}",
            f"Drones: {self.num_drones}",
            f"Threats: {self.num_threats}",
            f"Actual Threats: {self.num_actual_threat}",
        ]

        info = next(iter(self.infos.values()))
        if info:
            texts.extend(
                [
                    f"Coverage: {info['coverage']:.2f}",
                    f"Success Rate: {info['success_rate']:.2f}",
                    f"Threats Destroyed: {info['threat_destroyed']}",
                    f"Drones Lost: {info['drone_lost']}",
                    f"K/D Ratio: {info['kd_ratio']:.2f}",
                    f"Remaining Threats: {info['num_remaining_threat']}",
                ]
            )

        # Create a semi-transparent surface for text background
        text_surface = pygame.Surface((210, self.screen_height))
        text_surface.set_alpha(200)
        text_surface.fill((220, 220, 220))
        self.screen.blit(text_surface, (0, 0))

        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))

        # Draw legend
        legend_items = [
            ("Threat Level", get_threat_color(0.5)),
            ("Actual Threat", (100, 200, 255)),
            ("Active Drone", (0, 100, 255)),
            ("Damaged Drone", (255, 165, 0)),
            ("Destroyed Drone", (255, 0, 0)),
            ("Eliminate", (255, 0, 0)),
        ]

        legend_x = 10
        legend_y = self.screen_height - 150
        for i, (text, color) in enumerate(legend_items):
            if text == "Threat Level":
                # Draw a gradient for threat level
                gradient_width = 40
                for j in range(gradient_width):
                    threat_level = j / (gradient_width - 1)
                    gradient_color = get_threat_color(threat_level)
                    pygame.draw.line(
                        self.screen,
                        gradient_color,
                        (legend_x + 10 + j, legend_y + 10 + i * 20),
                        (legend_x + 10 + j, legend_y + 25 + i * 20),
                    )
            elif text == "Eliminate":
                pygame.draw.line(
                    self.screen,
                    color,
                    (legend_x + 10, legend_y + 15 + i * 20),
                    (legend_x + 30, legend_y + 25 + i * 20),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    color,
                    (legend_x + 30, legend_y + 15 + i * 20),
                    (legend_x + 10, legend_y + 25 + i * 20),
                    3,
                )
            else:
                if text == "Actual Threat":
                    # Draw a semi-transparent rectangle for actual threat
                    threat_surface = pygame.Surface((15, 15))
                    threat_surface.set_alpha(128)
                    threat_surface.fill(color)
                    self.screen.blit(
                        threat_surface, (legend_x + 10, legend_y + 10 + i * 20)
                    )
                else:
                    # Draw a circle for drones
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (legend_x + 17, legend_y + 17 + i * 20),
                        5,
                    )

            legend_text = font.render(text, True, (0, 0, 0))
            self.screen.blit(legend_text, (legend_x + 55, legend_y + 10 + i * 20))

        pygame.display.flip()
        if self._agent_selector.is_first() and self.render_config.get(
            "pause_at_end", False
        ):
            self.screen.blit(
                font.render("Press SPACE to continue", True, (0, 0, 0)),
                (self.screen_width // 2 - 100, 10),
            )
            pygame.display.flip()

            while not handle_events():
                pygame.time.wait(10)
        else:
            handle_events()

        if self.render_config.get("save_screenshot"):
            self._save_screenshot()

    def close(self):
        pygame.quit()

    def _init_rendering(self):
        pygame.init()
        self.screen_width = self.render_config.get("width", 1000)
        self.screen_height = self.render_config.get("height", 600)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Drone Task Allocation")

    def _save_screenshot(self):
        folder = f"logs/{self.timestamps}_{self.episode_count}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        file_path = os.path.abspath(os.path.join(folder, f"{self.agent_selection}.png"))
        pygame.image.save(self.screen, file_path)


def raw_env(config: dict = None) -> TaskAllocationEnv:
    """Create an environment with the given configuration.

    Args:
        config (dict, optional): Configuration dictionary. See environment documentation for details.

    Returns:
        TaskAllocationEnv: Wrapped environment.
    """
    env = TaskAllocationEnv(config)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


if __name__ == "__main__":
    env = raw_env(dict(render_config={"pause_at_end": True, "save_screenshot": True}))

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
