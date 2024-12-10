import numpy as np
import torch.nn as nn
from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

from target_assign_rl import TaskAllocationEnv
from target_assign_rl.examples.rllib.callback import MetricCallback


class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Layer sizes
        self.input_size = int(np.product(obs_space.shape))
        self.hidden_layer1_size = 256
        self.hidden_layer2_size = 128

        # Shared layers (embeddings)
        self.shared_layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_layer1_size),
            nn.ReLU(),
            nn.Linear(self.hidden_layer1_size, self.hidden_layer2_size),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(self.hidden_layer2_size, num_outputs)

        # Value head
        self.value_head = nn.Linear(self.hidden_layer2_size, 1)

        # Variable to store the last embeddings and value
        self._embeddings = None
        self._value_out = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get observations and ensure correct shape
        obs = input_dict["obs_flat"].float()

        # Compute shared embeddings
        self._embeddings = self.shared_layers(obs)

        # Compute value function from embeddings
        self._value_out = self.value_head(self._embeddings).squeeze(1)

        # Compute policy output from embeddings
        policy_out = self.policy_head(self._embeddings)

        return policy_out, state

    @override(TorchModelV2)
    def value_function(self):
        return self._value_out


config = (
    BCConfig()
    .framework("torch")
    .environment(TaskAllocationEnv)
    .training(
        lr=1e-4,
        gamma=0.99,
        model={"custom_model": CustomModel},
        _enable_learner_api=False,
    )
    .evaluation(
        evaluation_interval=20,
        evaluation_duration=10,
        evaluation_config={"input": "sampler"},
    )
    .offline_data(input_="D:/NZL/git/github/target_assign_rl/rllib_data/*.json")
    .callbacks(MetricCallback)
    .rollouts(
        num_rollout_workers=0,
        num_envs_per_worker=1,
        batch_mode="complete_episodes",
    )
    .resources(num_gpus=1, num_cpus_for_local_worker=4)
    .rl_module(_enable_rl_module_api=False)
)

if __name__ == "__main__":
    from ray import train, tune

    tuner = tune.Tuner(
        "BC",
        param_space=config.to_dict(),
        run_config=train.RunConfig(
            "BC_OfflineTrain",
            checkpoint_config=train.CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=100,
            ),
            stop={
                "training_iteration": 3000,
            },
        ),
    )

    results = tuner.fit()
    print(results.get_best_result("episode_reward_mean", "max"))
