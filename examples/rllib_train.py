from typing import Any, Dict, Tuple

import ray
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
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
        **kwargs
    ) -> None:
        rewards = postprocessed_batch[SampleBatch.REWARDS]
        rewards.fill(rewards[-1])
        postprocessed_batch[SampleBatch.REWARDS] = rewards


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="DQN")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_gpus", type=int, default=0)
    parser.add_argument("--action_mask", action="store_true")
    args = parser.parse_args()

    algo: Algorithm = get_trainable_cls(args.algo)
    config: AlgorithmConfig = algo.get_default_config()
    if args.action_mask:
        custom_model = TorchActionMaskModel
        if args.algo == "DQN":
            config.dueling = False
            config.hiddens = []
        elif args.algo == "PPO":
            config.model["vf_share_layers"] = True
    else:
        custom_model = None

    config = (
        config.framework("torch")
        .environment(TaskAllocationEnv)
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
            stop={"timesteps_total": 1e6, "episode_reward_mean": 6},
        ),
    )

    results = tuner.fit()
    print(results.get_best_result().get_best_checkpoint("episode_reward_mean", "max"))
