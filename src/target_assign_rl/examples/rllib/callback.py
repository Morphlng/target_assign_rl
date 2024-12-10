from typing import Any, Dict, Tuple

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.torch_utils import convert_to_torch_tensor


class MetricCallback(DefaultCallbacks):
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


def LoadCheckpointCallback(
    checkpoint_path: str,
    strict=True,
    base_cls=MetricCallback,
):
    class _LoadCheckpointCallback(base_cls):
        def on_algorithm_init(self, *, algorithm: Algorithm, **kwargs) -> None:
            policies = Policy.from_checkpoint(checkpoint_path)
            weights = (
                {pid: policy.get_weights() for pid, policy in policies.items()}
                if isinstance(policies, dict)
                else {"default_policy": policies.get_weights()}
            )

            if strict:
                algorithm.set_weights(weights)
            else:
                worker = algorithm.workers.local_worker()
                for pid, weight in weights.items():
                    policy: TorchPolicyV2 = worker.policy_map[pid]
                    weight = convert_to_torch_tensor(weight, device=policy.device)
                    policy.model.load_state_dict(weight, strict=False)

    return _LoadCheckpointCallback
