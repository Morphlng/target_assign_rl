import gymnasium as gym

from target_assign_rl.target_assign_aec import TaskAllocationAEC


class TaskAllocationEnv(gym.Env):
    """Wraps the TaskAllocationAEC to be compatible with OpenAI Gym."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.aec_env = TaskAllocationAEC(config)

        self.action_space = self.aec_env.action_space(self.aec_env.possible_agents[0])

        self.mask_obs = config.get("mask_obs", False)
        obs_space = self.aec_env.observation_space(self.aec_env.possible_agents[0])
        if self.mask_obs:
            obs_dict = {"observations": obs_space}
            obs_dict["action_mask"] = gym.spaces.MultiBinary(self.action_space.n)
            self.observation_space = gym.spaces.Dict(obs_dict)
        else:
            self.observation_space = obs_space

    def reset(self, *, seed=None, options=None):
        self.aec_env.reset(seed, options)
        obs, _, _, _, info = self.aec_env.last()
        if self.mask_obs:
            obs = {"observations": obs}
            obs["action_mask"] = self.aec_env.action_mask(self.aec_env.agent_selection)

        return obs, info

    def step(self, action):
        self.aec_env.step(action)
        obs, reward, te, tr, info = self.aec_env.last()
        if self.mask_obs:
            obs = {"observations": obs}
            obs["action_mask"] = self.aec_env.action_mask(self.aec_env.agent_selection)

        # Looped back to the first agent, episode is done
        if self.aec_env._agent_selector.is_first():
            tr = True

        return obs, reward, te, tr, info

    def action_masks(self):
        """Stable-Baselines3 requires this method to be implemented."""
        return list(map(bool, self.aec_env.action_mask(self.aec_env.agent_selection)))

    def render(self):
        self.aec_env.render()

    def close(self):
        self.aec_env.close()


if __name__ == "__main__":
    env = TaskAllocationEnv(dict(render_config={"pause_at_end": True}))

    for _ in range(10):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, te, tr, info = env.step(action)
            done = te or tr
            env.render()

    env.close()
