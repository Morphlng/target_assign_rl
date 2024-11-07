import os

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv

from target_assign_rl import TaskAllocationEnv


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.chckpoint_path = os.path.join(log_dir, "checkpoint_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                _mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f}")
                    print(f"Last mean reward per episode: {_mean_reward:.2f}")

                # New best model, you could save the agent here
                if _mean_reward > self.best_mean_reward:
                    self.best_mean_reward = _mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model at {x[-1]} timesteps")
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)
                else:
                    if self.verbose > 0:
                        print(f"Saving checkpoint model at {x[-1]} timesteps")
                        print(f"Saving checkpoint model to {self.chckpoint_path}.zip")
                    self.model.save(self.chckpoint_path)

        return True


if __name__ == "__main__":
    log_dir = "./mask_ppo_task_allocation/"
    os.makedirs(log_dir, exist_ok=True)

    def env_creator(env_kwargs=None):
        return TaskAllocationEnv(
            dict(
                min_drones=20,
                possible_level=[0, 0.05, 0.1, 0.5, 0.8],
                threat_dist=[0.1, 0.3, 0.1, 0.35, 0.15],
                attack_prob=0.6,
            )
        )

    parallel_vec_env = make_vec_env(env_creator, n_envs=os.cpu_count())

    # Create evaluation environment
    eval_env = DummyVecEnv(
        [
            lambda: Monitor(
                env_creator(),
                os.path.join(log_dir, "eval"),
            )
        ]
    )

    # Create evaluation callback
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=f"{log_dir}/best_model",
        log_path=f"{log_dir}/eval_results",
        eval_freq=1000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create auto save callback
    auto_save_callback = SaveOnBestTrainingRewardCallback(
        check_freq=10000, log_dir=log_dir
    )

    model = MaskablePPO(
        "MlpPolicy",
        parallel_vec_env,
        batch_size=128,
        verbose=1,
        tensorboard_log=log_dir,
    )
    try:
        model.learn(
            total_timesteps=10_000_000, callback=[auto_save_callback, eval_callback]
        )
    finally:
        model.save("mask_ppo_task_allocation")
