from __future__ import annotations

from typing import Any, Dict, Tuple

import ray
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.tune.registry import get_trainable_cls

from target_assign_rl import TaskAllocationEnv


class CustomCallbacks(DefaultCallbacks):
    def on_train_result(self, *, algorithm: Algorithm, result: dict, **kwargs) -> None:
        if algorithm._storage:
            algorithm._storage.current_checkpoint_index += 1
            result["checkpoint_dir_name"] = algorithm._storage.checkpoint_dir_name
            algorithm._storage.current_checkpoint_index -= 1

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode | EpisodeV2 | Exception,
        env_index: int | None = None,
        **kwargs,
    ) -> None:
        if worker.config.batch_mode == "truncate_episodes":
            raise RuntimeError("You should not use truncate_episodes with this env")

        info = episode.last_info_for()
        episode.custom_metrics["coverage"] = info["coverage"]
        episode.custom_metrics["success_rate"] = info["success_rate"]
        episode.custom_metrics["kd_ratio"] = info["kd_ratio"]
        episode.custom_metrics["num_remaining_threat"] = info["num_remaining_threat"]

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: Any,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[Any, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        # Do a manual TD-decay
        rewards = postprocessed_batch[SampleBatch.REWARDS]
        last_r = rewards[-1]
        gamma = policies[policy_id].config["gamma"]

        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += gamma * last_r
            last_r = rewards[i]

        postprocessed_batch[SampleBatch.REWARDS] = rewards


if __name__ == "__main__":
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="DQN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--action_mask", action="store_true")
    parser.add_argument("--env_config", type=str, default="{}")
    parser.add_argument("--train_steps", type=int, default=int(1e6))
    args = parser.parse_args()

    algo: Algorithm = get_trainable_cls(args.algo)
    config: AlgorithmConfig = algo.get_default_config()
    if args.action_mask:
        custom_model = TorchActionMaskModel
        config.env_config.update({"mask_obs": True})
        if args.algo == "DQN":
            config.dueling = False
            config.hiddens = []
        elif args.algo == "PPO":
            config.model["vf_share_layers"] = True
    else:
        custom_model = None

    config = (
        config.framework("torch")
        .environment(TaskAllocationEnv, env_config=json.loads(args.env_config))
        .training(
            lr=1e-4,
            gamma=0.99,
            model={"custom_model": custom_model},
            _enable_learner_api=False,
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 1.0,
                "final_epsilon": 0.05,
                "epsilon_timesteps": 10000,
            }
        )
        .callbacks(CustomCallbacks)
        .rollouts(
            num_rollout_workers=0,
            num_envs_per_worker=os.cpu_count(),
            batch_mode="complete_episodes",
        )
        .resources(num_gpus=args.num_gpus, num_cpus_for_local_worker=4)
        .rl_module(_enable_rl_module_api=False)
    )

    if args.debug:
        ray.init(local_mode=True)
        config.num_envs_per_worker = 1
        config.num_gpus = 0

    tuner = tune.Tuner(
        algo,
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            "RLlib_TargetAssign" + ("_debug" if args.debug else ""),
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=100,
                checkpoint_score_attribute="episode_reward_mean",
                checkpoint_score_order="max",
            ),
            stop={"timesteps_total": args.train_steps, "episode_reward_mean": 6},
        ),
    )

    results = tuner.fit()
    print(results.get_best_result().get_best_checkpoint("episode_reward_mean", "max"))
