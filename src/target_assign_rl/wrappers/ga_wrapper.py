import gymnasium as gym
import numpy as np

from target_assign_rl.examples.algorithm import EnhancedGeneticAlgorithm
from target_assign_rl.target_assign_env import TaskAllocationEnv


class GeneticAlgoWrapper(gym.Wrapper):
    env: TaskAllocationEnv

    def __init__(self, env: TaskAllocationEnv, config: dict = None):
        super().__init__(env)

        self.aec_env = env.aec_env
        self._config = config or {}

        self.high_level_layer = self._config.get("high_level_layer", None)
        self.high_level_window = self._config.get("high_level_window", None)
        self.high_level = self._config.get("high_level", 0.9)
        self.num_group = self._config.get("num_group", 4)
        self.ga_model = self._config.get("ga_model", None)
        self.ga_update = self._config.get("ga_update", False)
        self.ga_config = self._config.get("ga_config", {})
        self.ga_config["food_full"] = self.aec_env.attack_prob
        self.GA_result = np.full(self.num_group, self.aec_env.max_drones)

        obs_space = self.aec_env.observation_space(self.aec_env.possible_agents[0])
        if self.aec_env.dict_obs:
            obs_space["pre_allocation"] = gym.spaces.Box(
                low=0,
                high=self.aec_env.max_drones,
                shape=(self.num_group,),
                dtype=np.int32,
            )
        else:
            obs_space = gym.spaces.Box(
                low=0,
                high=np.inf,
                shape=(self.aec_env.num_threats * 2 + self.num_group,),
                dtype=np.float32,
            )

        if env.mask_obs:
            obs_dict = {"observations": obs_space}
            obs_dict["action_mask"] = gym.spaces.MultiBinary(env.action_space.n)
            self.observation_space = gym.spaces.Dict(obs_dict)
        else:
            self.observation_space = obs_space

        self.ga = EnhancedGeneticAlgorithm(self.ga_config)
        if self.ga_model is not None:
            self.ga.load_model(self.ga_model)

    def reset(self, *, seed=None, options=None):
        self.GA_result = np.full(self.num_group, self.aec_env.max_drones)
        obs, info = super().reset(seed=seed, options=options)

        self.group_size = self.aec_env.num_threats // self.num_group
        if self.high_level_window is not None:
            for x, y in self.high_level_window:
                self.aec_env.threat_levels[(x - 1) * self.group_size + y - 1] = (
                    self.high_level
                )

        if self.high_level_layer is not None and (
            0 <= self.high_level_layer <= self.num_group
        ):
            self.aec_env.threat_levels[
                self.high_level_layer
                * self.group_size : (self.high_level_layer + 1)
                * self.group_size
            ] = [self.high_level] * self.group_size

        self.sum_threats_list = [
            sum(
                self.aec_env.threat_levels[
                    i * self.group_size : (i + 1) * self.group_size
                ]
            )
            for i in range(self.num_group)
        ]

        self.GA_result = self.ga.predict(self.sum_threats_list)
        obs = self._update_obs(obs)
        return obs, info

    def step(self, action):
        obs, reward, te, tr, info = super().step(action)
        self.sum_actual_threat_list = [
            sum(
                self.aec_env.actual_threats[
                    i * self.group_size : (i + 1) * self.group_size
                ]
            )
            for i in range(self.num_group)
        ]
        if self.ga_update:
            self.ga.update(self.sum_threats_list, self.sum_actual_threat_list)

        obs = self._update_obs(obs)
        reward = self._update_reward(reward)
        return obs, reward, te, tr, info

    def _update_mask(self, agent=None):
        group_mask = [
            (
                0
                if sum(
                    self.aec_env.current_allocation[
                        i * self.group_size : (i + 1) * self.group_size
                    ]
                )
                >= self.GA_result[i]
                else 1
            )
            for i in range(self.num_group)
        ]
        pos_mask = [mask for mask in group_mask for _ in range(self.group_size)]
        return np.array(pos_mask).astype(np.int8)

    def _update_obs(self, obs):
        if self.env.mask_obs:
            if self.aec_env.dict_obs:
                obs["observations"]["pre_allocation"] = self.GA_result
            else:
                obs["observations"] = np.concatenate(
                    [
                        self.aec_env.threat_levels,
                        self.GA_result,
                        self.aec_env.current_allocation,
                    ],
                    dtype=np.float32,
                )
            obs["action_mask"] = self._update_mask()
        else:
            if self.aec_env.dict_obs:
                obs["pre_allocation"] = self.GA_result
            else:
                obs = np.concatenate(
                    [
                        self.aec_env.threat_levels,
                        self.GA_result,
                        self.aec_env.current_allocation,
                    ],
                    dtype=np.float32,
                )

        return obs

    def _update_reward(self, reward: float):
        ga_penalty = 0
        for i in range(self.num_group):
            group_sum = sum(
                self.aec_env.actual_allocation[
                    i * self.group_size : (i + 1) * self.group_size
                ]
            )
            if group_sum > self.GA_result[i]:
                ga_penalty -= abs(self.GA_result[i] - group_sum)

        return reward + ga_penalty


if __name__ == "__main__":
    env = TaskAllocationEnv(dict(mask_obs=True, min_drones=20))
    config = dict(
        ga_update=False,
        ga_model="0324_enhanced_ga_model.json",
        ga_config={
            "population_size": 100,
            "generations": 100,
            "elitism_count": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.8,
            "windows": 4,
            "food_supply": 20,
            "prob_options": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
        high_level_layer=7,
    )
    env = GeneticAlgoWrapper(env, config)
    obs, info = env.reset()
    print(obs, info)
    print("Done")
