import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers


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
        config = config or {}

        self.max_drones = config.get("max_drones", 20)
        self.min_drones = config.get("min_drones", 20)
        self.num_threats = config.get("num_threats", 20)
        self.attack_prob = config.get("attack_prob", 0.7)
        self.render_mode = config.get("render_mode", "human")
        self.dict_obs = config.get("dict_obs", False)
        self.possible_level = config.get("possible_level", [0, 0.2, 0.4, 0.6, 0.8])
        self.threat_dist = config.get("threat_dist", None)
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

        # Check if all drones have been allocated
        if self._agent_selector.is_last():
            self._simulate_engagement()
            self.rewards = self._calculate_rewards()
            self.truncations = {agent: True for agent in self.agents}
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
        return self.threat_levels > 0

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
        self.successful_engagements = np.zeros(self.num_threats, dtype=bool)
        self.drone_cost = np.zeros(self.num_threats, dtype=int)
        drone_iter = iter(self.agents)
        drone = next(drone_iter)

        for i in range(self.num_threats):
            if self.actual_threats[i]:
                for _ in range(self.current_allocation[i]):
                    while drone and (
                        self.truncations[drone] or self.terminations[drone]
                    ):
                        drone = next(drone_iter, None)

                    if drone is not None:
                        self.terminations[drone] = True
                        self.drone_cost[i] += 1
                        if np.random.random() < self.attack_prob:
                            self.successful_engagements[i] = True
                            break

    def _calculate_rewards(self):
        covered_threats = (self.current_allocation > 0)[self.actual_threats]
        coverage = (
            (np.sum(covered_threats) / self.num_actual_threat)
            if self.num_actual_threat > 0
            else 0
        )

        # 1. Weighted coverage
        weighted_coverage = np.sum(
            covered_threats * self.threat_levels[self.actual_threats]
        ) / (np.sum(self.threat_levels[self.actual_threats]) + 1e-8)

        # 2. Success rate
        threats_destroyed = np.sum(self.successful_engagements)
        drones_lost = sum(self.terminations.values())
        success_rate = (
            (threats_destroyed / self.num_actual_threat)
            if self.num_actual_threat > 0
            else 0
        )

        # 3. Remain punishment
        remaining_penalty = np.sum(
            self.threat_levels[self.actual_threats & ~self.successful_engagements]
        ) / (np.sum(self.threat_levels[self.actual_threats]) + 1e-8)

        # 4. Redundancy penalty
        redundancy = np.maximum(
            self.current_allocation[self.successful_engagements]
            - self.drone_cost[self.successful_engagements]
            - 1,
            0,
        )
        redundancy_penalty = np.sum(
            redundancy / (self.current_allocation[self.successful_engagements] + 1e-8)
        )

        # 5. Zero prob allocation penalty (enable only if no action_mask)
        # zero_penalty = np.sum(self.current_allocation * (self.threat_levels == 0))

        # overall reward
        total_reward = (
            weighted_coverage + success_rate - remaining_penalty - redundancy_penalty
        )

        self.infos = {
            agent: {
                "coverage": coverage,
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
        if self.render_mode == "human":
            print(f"Threat levels: {self.threat_levels}")
            print(f"Current allocation: {self.current_allocation}")
            print(f"Pre-allocation: {self.pre_allocation}")


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
    env = raw_env()

    for _ in range(20):
        env.reset()
        for agent in env.agents:
            obs, reward, te, tr, info = env.last()
            if te or tr:
                action = None
            else:
                action = np.random.choice(np.where(env.action_mask(agent))[0])
            env.step(action)
