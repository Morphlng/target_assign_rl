import os
import numpy as np
from ray.rllib.evaluation.sample_batch_builder import SampleBatchBuilder
from ray.rllib.offline.json_writer import JsonWriter
from target_assign_rl import TaskAllocationEnv
from target_assign_rl.target_assign_agent import RuleAgent

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser("Collect offline data in RLlib format")
    parser.add_argument(
        "--save_dir", type=str, default=os.path.join(os.getcwd(), "rllib_data")
    )
    parser.add_argument("--num_episodes", type=int, default=100000)
    parser.add_argument("--max_file_size", type=int, default=64)
    args = parser.parse_args()
    env = TaskAllocationEnv()
    agent = RuleAgent(env.aec_env.num_threats)
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(
        args.save_dir,
        max_file_size=args.max_file_size * 1024 * 1024,
        compress_columns=[],
    )
    for eps_id in tqdm(range(args.num_episodes)):
        obs, prev_info = env.reset()
        prev_action = np.zeros_like(env.action_space.sample())
        prev_reward = 0
        done = False
        t = 0
        while not done:
            action = agent.predict(obs)
            new_obs, reward, terminated, truncated, new_info = env.step(action)
            batch_builder.add_values(
                t=t,
                eps_id=eps_id,
                agent_index=0,
                obs=obs,
                actions=action,
                action_prob=1.0,
                action_logp=0.0,
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=truncated,
                infos={},  # info is not useful for training
                new_obs=new_obs,
            )
            t += 1
            done = terminated or truncated
            obs = new_obs
            prev_action = action
            prev_reward = reward
            prev_info = new_info
        writer.write(batch_builder.build_and_reset())
