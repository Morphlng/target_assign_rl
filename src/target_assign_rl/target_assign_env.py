import gymnasium as gym

from target_assign_rl.target_assign_aec import TaskAllocationAEC


class TaskAllocationEnv(gym.Env):
    """Wraps the TaskAllocationAEC to be compatible with OpenAI Gym."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.aec_env = TaskAllocationAEC(config)

        self.action_space = self.aec_env.action_space(self.aec_env.possible_agents[0])
        self.observation_space = gym.spaces.Dict(
            {
                "observations": self.aec_env.observation_space(
                    self.aec_env.possible_agents[0]
                ),
                "action_mask": gym.spaces.MultiBinary(self.action_space.n),
            }
        )

    def reset(self, *, seed=None, options=None):
        self.aec_env.reset(seed, options)
        obs, reward, te, tr, info = self.aec_env.last()
        action_mask = self.aec_env.action_mask(self.aec_env.agent_selection)
        return {"observations": obs, "action_mask": action_mask}, info

    def step(self, action):
        self.aec_env.step(action)
        obs, reward, te, tr, info = self.aec_env.last()
        action_mask = self.aec_env.action_mask(self.aec_env.agent_selection)
        return {"observations": obs, "action_mask": action_mask}, reward, te, tr, info

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
