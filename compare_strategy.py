import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from target_assign_agent import IQLAgent, RuleAgent, RandomAgent
from target_assign_env import raw_env, TaskAllocationEnv


def simulate_drone_lost(
    trained_agent: IQLAgent,
    compare_agent: RuleAgent = None,
    num_episodes=100,
    max_drone_lost=4,
    env=None,
):
    if env is None:
        env = raw_env(dict(min_drones=20))

    compare_agent = trained_agent if compare_agent is None else compare_agent
    baselines = []
    comparison = []

    for episode in range(num_episodes):
        env.reset()
        drone_lost = np.random.randint(0, max_drone_lost + 1)

        for i, agent in enumerate(env.agents):
            state, _, te, tr, _ = env.last()
            action_mask = env.action_mask(agent)
            action = trained_agent.predict(state, action_mask)
            if drone_lost > 0 and i == len(env.agents) - 1:
                while True:
                    lost_drones = np.random.choice(
                        env.agents, drone_lost, replace=False
                    )
                    if agent not in lost_drones:
                        break
                for drone in lost_drones:
                    env.truncations[drone] = True
            env.step(action)

        _, original_reward, _, __, original_info = env.last()
        episode_data = {
            "possible_level": env.possible_level.copy(),
            "threat_levels": env.threat_levels.copy(),
            "actual_threats": env.actual_threats.copy(),
            "num_actual_threat": env.num_actual_threat,
        }
        episode_data["assignments"] = env.actual_allocation.copy()
        episode_data["drone_cost"] = env.drone_cost.copy()
        episode_data["successful_engagements"] = env.successful_engagements.copy()
        episode_data["final_reward"] = original_reward
        episode_data["coverage"] = original_info["coverage"]
        episode_data["threat_destroyed"] = original_info["threat_destroyed"]
        episode_data["drone_lost"] = original_info["drone_lost"]
        episode_data["num_remaining_threat"] = original_info["num_remaining_threat"]
        baselines.append(episode_data)

        # num_lost env
        num_drones = 20 - drone_lost
        env2 = TaskAllocationEnv(
            dict(
                min_drones=num_drones,
                max_drones=num_drones,
                attack_prob=env.attack_prob,
                possible_level=env.possible_level,
                threat_dist=env.threat_dist,
            )
        )
        env2.reset()
        env2.threat_levels = env.threat_levels.copy()
        env2.actual_threats = env.actual_threats.copy()
        env2.num_actual_threat = env.num_actual_threat
        env2.pre_allocation = env2.calculate_pre_allocation()

        for agent in env2.agents:
            state, _, te, tr, _ = env2.last()
            action_mask = env2.action_mask(agent)
            action = compare_agent.predict(state, action_mask)
            env2.step(action)

        _, new_reward, _, __, new_info = env2.last()
        new_data = {
            "possible_level": env2.possible_level.copy(),
            "threat_levels": env2.threat_levels.copy(),
            "actual_threats": env2.actual_threats.copy(),
            "num_actual_threat": env2.num_actual_threat,
        }
        new_data["assignments"] = env2.actual_allocation.copy()
        new_data["drone_cost"] = env2.drone_cost.copy()
        new_data["successful_engagements"] = env2.successful_engagements.copy()
        new_data["final_reward"] = new_reward
        new_data["coverage"] = new_info["coverage"]
        new_data["threat_destroyed"] = new_info["threat_destroyed"]
        new_data["drone_lost"] = new_info["drone_lost"]
        new_data["num_remaining_threat"] = new_info["num_remaining_threat"]
        comparison.append(new_data)

    return baselines, comparison


def generate_threat_heatmap_data(collected_data):
    num_threats = len(collected_data[0]["threat_levels"])
    num_episodes = len(collected_data)

    threat_levels_sum = np.zeros(num_threats)
    assignments_sum = np.zeros(num_threats)
    successful_engagements_cnt = np.zeros(num_threats)
    actual_threats_sum = np.zeros(num_threats)
    drone_cost_success_sum = np.zeros(num_threats)
    assignments_when_actual_cnt = np.zeros(num_threats)

    for episode_data in collected_data:
        threat_levels_sum += episode_data["threat_levels"]
        assignments_sum += episode_data["assignments"]
        successful_engagements_cnt += episode_data["successful_engagements"]
        actual_threats_sum += episode_data["actual_threats"]
        drone_cost_success_sum += (
            episode_data["drone_cost"] * episode_data["successful_engagements"]
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
        successful_engagements_cnt,
        assignments_when_actual_cnt,
        out=np.zeros_like(successful_engagements_cnt),
        where=assignments_when_actual_cnt != 0,
    )

    # Average drones used for successful engagements
    avg_drones_for_success = np.divide(
        drone_cost_success_sum,
        successful_engagements_cnt,
        out=np.zeros_like(drone_cost_success_sum),
        where=successful_engagements_cnt != 0,
    )

    # Failure rate: (actual threats with assignments - successful engagements) / (actual threats with assignments)
    failure_rate = np.divide(
        assignments_when_actual_cnt - successful_engagements_cnt,
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
    heatmap_data = generate_threat_heatmap_data(collected_data)

    plt.figure(figsize=(20, 12))
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

    plt.title("Threat Position Heatmap", fontsize=16)
    plt.xlabel("Threat Position", fontsize=12)
    plt.ylabel("Metrics", fontsize=12)
    plt.tight_layout()
    plt.savefig(path or "threat_position_heatmap.png")

    if show:
        plt.show()


def analyze_multi_round_statistics(collected_data, show=False):
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
                "successful_engagements": d["successful_engagements"],
            }
            for d in collected_data
        ]
    )

    overall_stats = pd.DataFrame(
        {
            "Coverage Rate": df["coverage"].mean(),
            "Avg Threats Destroyed": df["threats_destroyed"].mean(),
            "Avg Remaining Threats": df["remaining_threats"].mean(),
            "Avg Total Threats": df["total_threats"].mean(),
        },
        index=["Overall"],
    )

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
                    "destroyed": (
                        level_threats & (row["successful_engagements"])
                    ).sum(),
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
    print("Overall Statistics:")
    print(overall_stats)
    print("\nStatistics by Threat Level:")
    print(threat_level_df)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    overall_stats.T.plot(kind="bar", ax=ax1, ylabel="Value")
    ax1.set_title("Overall Statistics")
    ax1.set_xlabel("")

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


def create_agent(agent_info):
    env = raw_env(
        dict(
            min_drones=20,
            possible_level=[0, 0.1, 0.4, 0.8],
            threat_dist=[0.15, 0.35, 0.35, 0.15],
        )
    )
    env.reset()

    if agent_info == "rule":
        agent_name = "Rule-based"
    elif agent_info == "random":
        agent_name = "Random"
    else:
        agent_name = f"IQL_Agent_{os.path.basename(agent_info).split('.')[0]}"

    if agent_info == "rule":
        agent = RuleAgent()
    elif agent_info == "random":
        agent = RandomAgent()
    else:
        agent = IQLAgent(env.state().shape[0], env.action_space(env.agents[0]).n)
        agent.load_checkpoint(agent_info)

    return agent_name, agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--drone_lost", type=int, default=0)
    parser.add_argument("--base_agent", type=str, default="rule")
    parser.add_argument("--compare_agent", type=str, default="random")
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    base_agent_name, base_agent = create_agent(args.base_agent)
    compare_agent_name, compare_agent = create_agent(args.compare_agent)
    env = raw_env(
        dict(
            min_drones=20,
            # [0, 0.05, 0.5, 0.8]
            # [0.1, 0.4, 0.35, 0.15]
            possible_level=[0, 0.05, 0.1, 0.5, 0.8],
            threat_dist=[0.1, 0.3, 0.1, 0.35, 0.15],
            attack_prob=0.6,
        )
    )

    base_data, compare_data = simulate_drone_lost(
        base_agent, compare_agent, args.num_episodes, args.drone_lost, env
    )

    plot_threat_heatmap(
        base_data, show=args.show, path=f"{base_agent_name}_heatmap.png"
    )
    print(f"{base_agent_name}'s data:")
    base_overall_stats, base_threat_level_stats = analyze_multi_round_statistics(
        base_data
    )
    base_overall_stats.to_csv(f"{base_agent_name}_overall_stats.csv")
    base_threat_level_stats.to_csv(f"{base_agent_name}_threat_level_stats.csv")

    plot_threat_heatmap(
        compare_data, show=args.show, path=f"{compare_agent_name}_heatmap.png"
    )
    print(f"\n{compare_agent_name}'s data:")
    compare_overall_stats, compare_threat_level_stats = analyze_multi_round_statistics(
        compare_data
    )
    compare_overall_stats.to_csv(f"{compare_agent_name}_overall_stats.csv")
    compare_threat_level_stats.to_csv(f"{compare_agent_name}_threat_level_stats.csv")
