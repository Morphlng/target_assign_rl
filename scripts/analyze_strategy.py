import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from target_assign_rl import Agent, IQLAgent, RuleAgent, TaskAllocationAEC, raw_env


def inference_and_collect_data(
    env: TaskAllocationAEC, trained_agent: Agent, num_episodes: int = 100
):
    collected_data = []

    for _ in range(num_episodes):
        env.reset()
        episode_data = {
            "possible_level": env.possible_level.copy(),
            "threat_levels": env.threat_levels.copy(),
            "actual_threats": env.actual_threats.copy(),
        }

        for i, agent in enumerate(env.agents):
            state, _, te, tr, _ = env.last()
            action_mask = env.action_mask(agent)
            action = trained_agent.predict(state, action_mask)
            env.step(action)

        _, reward, _, _, info = env.last()
        episode_data["assignments"] = env.actual_allocation.copy()
        episode_data["eliminated_threats"] = env.eliminated_threats.copy()
        episode_data["final_reward"] = reward
        episode_data["coverage"] = info["coverage"]
        episode_data["threat_destroyed"] = info["threat_destroyed"]
        episode_data["drone_lost"] = info["drone_lost"]
        episode_data["num_remaining_threat"] = info["num_remaining_threat"]
        episode_data["num_actual_threat"] = env.num_actual_threat
        episode_data["drone_cost"] = env.drone_cost.copy()

        collected_data.append(episode_data)

    return collected_data


def generate_threat_heatmap_data(collected_data):
    num_threats = len(collected_data[0]["threat_levels"])
    num_episodes = len(collected_data)

    # Initialize arrays to store data
    threat_levels_sum = np.zeros(num_threats)
    assignments_sum = np.zeros(num_threats)
    eliminated_threats_cnt = np.zeros(num_threats)
    actual_threats_sum = np.zeros(num_threats)
    drone_cost_success_sum = np.zeros(num_threats)
    assignments_when_actual_cnt = np.zeros(num_threats)

    for episode_data in collected_data:
        threat_levels_sum += episode_data["threat_levels"]
        assignments_sum += episode_data["assignments"]
        eliminated_threats_cnt += episode_data["eliminated_threats"]
        actual_threats_sum += episode_data["actual_threats"]
        drone_cost_success_sum += (
            episode_data["drone_cost"] * episode_data["eliminated_threats"]
        )
        assignments_when_actual = (episode_data["assignments"] > 0) * episode_data[
            "actual_threats"
        ]
        assignments_when_actual_cnt += assignments_when_actual

    # Calculate averages and rates
    threat_levels_avg = threat_levels_sum / num_episodes
    assignments_avg = assignments_sum / num_episodes

    # Success rate: successful engagements / (actual threats with assignments)
    success_rate = np.divide(
        eliminated_threats_cnt,
        assignments_when_actual_cnt,
        out=np.zeros_like(eliminated_threats_cnt),
        where=assignments_when_actual_cnt != 0,
    )

    # Average drones used for successful engagements
    avg_drones_for_success = np.divide(
        drone_cost_success_sum,
        eliminated_threats_cnt,
        out=np.zeros_like(drone_cost_success_sum),
        where=eliminated_threats_cnt != 0,
    )

    # Failure rate: (actual threats with assignments - successful engagements) / (actual threats with assignments)
    failure_rate = np.divide(
        assignments_when_actual_cnt - eliminated_threats_cnt,
        assignments_when_actual_cnt,
        out=np.ones_like(assignments_when_actual_cnt),
        where=assignments_when_actual_cnt != 0,
    )

    # Coverage rate: (actual threats with assignments) / (actual threats)
    coverage_rate = np.divide(
        assignments_when_actual_cnt,
        actual_threats_sum,
        out=np.zeros_like(actual_threats_sum),
        where=actual_threats_sum != 0,
    )

    return np.array(
        [
            threat_levels_avg,
            assignments_avg,
            success_rate,
            avg_drones_for_success,
            failure_rate,
            coverage_rate,
        ]
    )


def plot_threat_heatmap(collected_data, show=False, path=None):
    # 生成热力图数据
    heatmap_data = generate_threat_heatmap_data(collected_data)

    # 设置图形大小
    plt.figure(figsize=(20, 12))

    # 创建热力图
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        xticklabels=range(1, 21),
        yticklabels=[
            "Avg Threat Level",
            "Avg Allocation",
            "Success Rate",
            "Avg Drones for Success",
            "Failure Rate",
            "Coverage Rate",
        ],
    )

    # 设置标题和标签
    plt.title("Threat Position Heatmap", fontsize=16)
    plt.xlabel("Threat Position", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)

    # 调整布局
    plt.tight_layout()
    plt.savefig(path or "threat_position_heatmap.png")

    # 显示图形
    if show:
        plt.show()


def analyze_multi_round_statistics(collected_data, show=False):
    # 创建一个DataFrame来存储每个回合的数据
    df = pd.DataFrame(
        [
            {
                "coverage": d["coverage"],
                "threats_destroyed": d["threat_destroyed"],
                "remaining_threats": d["num_remaining_threat"],
                "total_threats": d["num_actual_threat"],
                "threat_levels": d["threat_levels"],
                "actual_threats": d["actual_threats"],
                "assignments": d["assignments"],
                "eliminated_threats": d["eliminated_threats"],
            }
            for d in collected_data
        ]
    )

    # 计算总体统计数据
    overall_stats = pd.DataFrame(
        {
            "Coverage Rate": df["coverage"].mean(),
            "Avg Threats Destroyed": df["threats_destroyed"].mean(),
            "Avg Remaining Threats": df["remaining_threats"].mean(),
            "Avg Total Threats": df["total_threats"].mean(),
        },
        index=["Overall"],
    )

    # 按威胁度进行分组统计
    threat_level_stats = []
    possible_levels = collected_data[0]["possible_level"]

    for level in possible_levels:
        level_data = []
        for _, row in df.iterrows():
            level_threats = row["actual_threats"] & (row["threat_levels"] == level)
            level_data.append(
                {
                    "threats": level_threats.sum(),
                    "covered": (level_threats & (row["assignments"] > 0)).sum(),
                    "destroyed": (level_threats & (row["eliminated_threats"])).sum(),
                }
            )

        level_df = pd.DataFrame(level_data)
        if level_df["threats"].sum() > 0:
            threat_level_stats.append(
                {
                    "Threat Level": level,
                    "Coverage": level_df["covered"].sum() / level_df["threats"].sum(),
                    "Success": level_df["destroyed"].sum() / level_df["threats"].sum(),
                    "Avg Threats": level_df["threats"].mean(),
                    "Avg Destroyed": level_df["destroyed"].mean(),
                    "Avg Remaining": (
                        level_df["threats"] - level_df["destroyed"]
                    ).mean(),
                }
            )

    threat_level_df = pd.DataFrame(threat_level_stats).set_index("Threat Level")

    # 打印结果
    print("Overall Statistics:")
    print(overall_stats)
    print("\nStatistics by Threat Level:")
    print(threat_level_df)

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # 总体统计柱状图
    overall_stats.T.plot(kind="bar", ax=ax1, ylabel="Value")
    ax1.set_title("Overall Statistics")
    ax1.set_xlabel("")

    # 按威胁度统计的堆叠柱状图
    threat_level_df[["Avg Destroyed", "Avg Remaining"]].plot(
        kind="bar", stacked=True, ax=ax2
    )
    ax2.set_title("Average Threats by Threat Level")
    ax2.set_xlabel("Threat Level")
    ax2.set_ylabel("Number of Threats")

    plt.tight_layout()
    plt.savefig("multi_round_statistics.png")
    if show:
        plt.show()

    return overall_stats, threat_level_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--agent", type=str, default="rule")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    env = raw_env(
        dict(
            min_drones=20,
            possible_level=[0, 0.1, 0.4, 0.8],
            threat_dist=[0.15, 0.35, 0.35, 0.15],
        )
    )
    env.reset()
    if args.agent == "rule":
        agent = RuleAgent()
        agent_name = "Rule-based"
    else:
        agent = IQLAgent(env.state().shape[0], env.action_space(env.agents[0]).n)
        agent.load_checkpoint(args.agent)
        agent_name = f"IQL-Agent_{os.path.basename(args.agent).split('.')[0]}"

    collected_data = inference_and_collect_data(env, agent, args.num_episodes)
    plot_threat_heatmap(
        collected_data, show=args.show, path=f"{agent_name}_heatmap.png"
    )

    overall_stats, threat_level_stats = analyze_multi_round_statistics(collected_data)
    overall_stats.to_csv(f"{agent_name}_overall_stats.csv")
    threat_level_stats.to_csv(f"{agent_name}_threat_level_stats.csv")
