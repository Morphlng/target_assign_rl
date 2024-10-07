import os
import sys
from typing import Callable

import numpy as np
from gif_maker import PygameRecord

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from target_assign_agent import RandomAgent, RuleAgent
from target_assign_aec import TaskAllocationEnv, raw_env


def record_and_compare_scenario(
    baseline_agent: RuleAgent,
    compare_agent: RuleAgent,
    scenario_name: str,
    condition: Callable[[np.ndarray], bool],
    max_attempts: int = 1000,
    drone_lost: int = 0,
    env=None,
):
    baseline_name = type(baseline_agent).__name__
    compare_name = type(compare_agent).__name__

    if env is None:
        env = raw_env()

    for attempt in range(max_attempts):
        env.reset()
        if condition(env.threat_levels):
            with PygameRecord(
                f"{scenario_name}_{baseline_name}_{attempt}.gif", 30
            ) as recorder:
                for i, agent in enumerate(env.agents):
                    state = env.state()
                    action_mask = env.action_mask(agent)
                    action = baseline_agent.predict(state, action_mask)
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
                    env.render()
                    recorder.add_frame()

                # Make the last frame last longer
                for _ in range(30):
                    recorder.add_frame()
                recorder.save()

            # Create a new environment with the same configuration for the rule agent
            num_drones = env.num_drones - drone_lost
            env2 = TaskAllocationEnv(
                dict(
                    min_drones=num_drones,
                    max_drones=num_drones,
                    num_threats=env.num_threats,
                    attack_prob=env.attack_prob,
                    possible_level=env.possible_level,
                    threat_dist=env.threat_dist,
                    render_config=env.render_config,
                )
            )
            env2.reset()
            env2.threat_levels = env.threat_levels.copy()
            env2.actual_threats = env.actual_threats.copy()
            env2.num_actual_threat = env.num_actual_threat
            env2.pre_allocation = env2.calculate_pre_allocation()

            with PygameRecord(
                f"{scenario_name}_{compare_name}_{attempt}.gif", 30
            ) as recorder:
                for agent in env2.agents:
                    state = env2.state()
                    action_mask = env.action_mask(agent)
                    action = compare_agent.predict(state, action_mask)
                    env2.step(action)
                    env2.render()
                    recorder.add_frame()

                # Make the last frame last longer
                for _ in range(30):
                    recorder.add_frame()
                recorder.save()

            print(
                f"Successfully recorded {scenario_name} scenario on attempt {attempt + 1}"
            )
            return
    print(
        f"Failed to find a suitable {scenario_name} scenario after {max_attempts} attempts"
    )


def record_no_zero_threats(env, base_agent, compare_agent, drone_lost=0):
    def condition(threat_levels):
        return np.all(threat_levels > 0)

    record_and_compare_scenario(
        base_agent,
        compare_agent,
        "no_zero_threats",
        condition,
        drone_lost=drone_lost,
        env=env,
    )


def record_many_zero_threats(
    env, base_agent, compare_agent, drone_lost=0, threshold=0.3
):
    def condition(threat_levels):
        return np.mean(threat_levels == 0) >= threshold

    record_and_compare_scenario(
        base_agent,
        compare_agent,
        "many_zero_threats",
        condition,
        drone_lost=drone_lost,
        env=env,
    )


def record_many_medium_threats(
    env, base_agent, compare_agent, drone_lost=0, threshold=0.3
):
    def condition(threat_levels):
        return np.mean((threat_levels >= 0.2) & (threat_levels <= 0.7)) >= threshold

    record_and_compare_scenario(
        base_agent,
        compare_agent,
        "many_medium_threats",
        condition,
        drone_lost=drone_lost,
        env=env,
    )


def compare_and_record_scenarios(base_agent, compare_agent, env_config, drone_lost=0):
    env = raw_env(env_config)

    # Record and compare scenarios
    record_no_zero_threats(env, base_agent, compare_agent, drone_lost)
    record_many_zero_threats(env, base_agent, compare_agent, drone_lost)
    record_many_medium_threats(env, base_agent, compare_agent, drone_lost)


if __name__ == "__main__":
    env_config = dict(
        min_drones=20,
        possible_level=[0, 0.05, 0.1, 0.5, 0.8],
        threat_dist=[0.1, 0.3, 0.1, 0.35, 0.15],
        attack_prob=0.6,
    )
    rule_agent = RuleAgent()
    random_agent = RandomAgent()
    compare_and_record_scenarios(random_agent, rule_agent, env_config)
