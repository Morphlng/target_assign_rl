import os
import time
from copy import deepcopy

import numpy as np
import pygame
from scipy.optimize import linprog

from target_assign_rl import Agent, RuleAgent, raw_env
from target_assign_rl.examples.gif_maker import PygameRecord
from target_assign_rl.viz import get_threat_color, handle_events


class ReallocationDemo:
    def __init__(
        self,
        env_config: dict = None,
        width: int = 1200,
        height: int = 800,
        grid_size: int = 40,
        cols: int = 25,
        rows: int = 20,
        max_reallocation: int = 3,
        recording: bool = False,
        record_dir: str = "gifs",
        fps: int = 30,
    ):
        # Display settings
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.cols = cols
        self.rows = rows
        self.building_col = self.cols - 5

        # Recording settings
        self.recording = recording
        self.record_dir = record_dir
        self.fps = fps
        if self.recording:
            os.makedirs(self.record_dir, exist_ok=True)

        # Movement settings
        self.MOVE_SPEED = 2  # pixels per frame
        self.REALLOCATION_INTERVAL = 100  # frames
        self.ATTACK_RANGE = 1 * self.grid_size
        self.max_reallocation = max_reallocation

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("UAV Strike Process")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("SimHei", 24)

        self.env = raw_env(env_config)
        self.state_hist = []
        self.realloc_cnt = 0
        self.reset_simulation()

    @property
    def attack_x(self):
        return int(self.building_col * self.grid_size - self.ATTACK_RANGE)

    @property
    def building_start(self):
        return int(self.building_col * self.grid_size)

    def reset_simulation(self, state: dict = None):
        """Initialize a new strike mission"""
        state = state or {}
        if not state:
            self.env.reset()
        else:
            for key, value in state.items():
                setattr(self.env.unwrapped, key, value)

        self.frame_count = 0
        self.realloc_cnt = 0
        env_state = deepcopy(vars(self.env.unwrapped))

        # Initialize UAVs at left side
        self.uavs = []
        spacing = self.height / self.env.num_drones
        for i in range(self.env.num_drones):
            self.uavs.append(
                {
                    "id": f"drone_{i}",
                    "pos": np.array([0, i * spacing]),
                    "target_pos": np.array([0, i * spacing]),
                    "status": "active",
                    "current_row": None,
                    "target_row": None,
                    "in_transition": False,
                    "entered": False,
                }
            )

        # Initialize threat state
        window_map = state.get("window_mapping", None)
        if window_map is not None:
            self.window_mapping = window_map
        else:
            self.window_mapping = np.random.permutation(self.rows)
        self.window_threats = self.env.threat_levels[self.window_mapping]
        self.enemies = [
            {
                "row": row,
                "status": (
                    "active"
                    if self.env.actual_threats[self.window_mapping[row]]
                    else "none"
                ),
            }
            for row in range(self.rows)
        ]

        self.perform_allocation()

        # Make the initial position of UAVs the same as their target position
        for uav in self.uavs:
            row_assign = self.row_assignments[uav["target_row"]]
            row_num = len(row_assign)

            r_index = row_assign.index(uav)
            uav["pos"] = uav["target_pos"] + np.array(
                [(row_num - r_index - 1) * self.grid_size, 0]
            )
            uav["current_row"] = uav["target_row"]
            uav["in_transition"] = False

        env_state["window_mapping"] = self.window_mapping
        self.state_hist = [env_state]

    def perform_allocation(self, agent: Agent = None):
        """Allocate UAVs to targets based on current threat levels"""
        if agent is None:
            agent = RuleAgent(self.rows)

        # Step 1: Determine the number of UAVs required per target window
        target_assignments = [0] * self.rows
        for uav in self.uavs:
            if uav["status"] != "active":
                continue

            obs = self.env.observe(None)
            action_mask = self.env.action_mask(None)
            action = agent.predict(obs, action_mask)
            self.env.step(action)
            target_row = np.where(self.window_mapping == action)[0][0]
            target_assignments[target_row] += 1

        # Step 2: Define the linear programming problem
        num_uavs = len(self.uavs)
        num_rows = self.rows

        # Create the cost matrix: cost[i, j] is the distance of UAV `i` to target row `j`
        cost_matrix = np.zeros((num_uavs, num_rows))
        for i, uav in enumerate(self.uavs):
            if uav["status"] != "active":
                cost_matrix[i, :] = np.inf
                continue

            for j in range(num_rows):
                target_y = j * self.grid_size + self.grid_size / 2
                cost_matrix[i, j] = np.linalg.norm(
                    uav["pos"]
                    - np.array([self.building_col * self.grid_size, target_y]),
                )

        # Flatten the cost matrix to create the objective function for linprog
        c = cost_matrix.flatten()

        # Create the constraints
        # Constraint 1: Each UAV can only be assigned to one target row (row sum = 1)
        A_eq_uav = np.zeros((num_uavs, num_uavs * num_rows))
        for i in range(num_uavs):
            A_eq_uav[i, i * num_rows : (i + 1) * num_rows] = 1

        b_eq_uav = np.ones(num_uavs)  # Each UAV is assigned to exactly one target

        # Constraint 2: Each target row must receive the required number of UAVs
        A_eq_row = np.zeros((num_rows, num_uavs * num_rows))
        for j in range(num_rows):
            for i in range(num_uavs):
                A_eq_row[j, i * num_rows + j] = 1

        b_eq_row = target_assignments  # Each target row must receive the required number of UAVs

        # Combine the equality constraints
        A_eq = np.vstack([A_eq_uav, A_eq_row])
        b_eq = np.hstack([b_eq_uav, b_eq_row])

        # Bounds: Each variable (UAV-to-row assignment) must be binary (0 or 1)
        # Note: linprog only supports continuous variables, so we relax this to [0, 1]
        bounds = [(0, 1) for _ in range(num_uavs * num_rows)]

        # Step 3: Solve the linear programming problem
        result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        if not result.success:
            raise ValueError("Linear programming failed to find a solution.")

        # Step 4: Extract the assignment from the result
        assignment = result.x.reshape((num_uavs, num_rows))

        # Assign UAVs to their respective target rows
        for i, uav in enumerate(self.uavs):
            if uav["status"] != "active":
                continue

            target_row = np.argmax(assignment[i])
            if assignment[i, target_row] > 0:
                if uav["target_row"] != target_row:
                    uav["target_row"] = target_row
                    uav["in_transition"] = True
                    target_x = min(
                        uav["pos"][0]
                        + abs((uav["current_row"] or target_row) - target_row)
                        * self.grid_size,
                        self.attack_x,
                    )
                    target_y = target_row * self.grid_size + self.grid_size / 2
                    uav["target_pos"] = np.array(
                        [
                            target_x,  # Move forward
                            target_y,  # Center of target row
                        ]
                    )

        # Create a dictionary to track UAVs assigned to each row
        self.row_assignments: "dict[int, list]" = {}
        for uav in self.uavs:
            if uav["target_row"] is not None:
                if uav["target_row"] not in self.row_assignments:
                    self.row_assignments[uav["target_row"]] = []
                self.row_assignments[uav["target_row"]].append(uav)

        # Sort UAVs in each row by x-position (rightmost first)
        for row in self.row_assignments:
            self.row_assignments[row].sort(key=lambda u: (-u["pos"][0], u["id"]))

    def update_positions(self):
        """Update UAV positions"""

        # Update row assignments
        for row in self.row_assignments:
            self.row_assignments[row].sort(key=lambda u: (-u["pos"][0], u["id"]))

        for uav in self.uavs:
            if uav["status"] != "active":
                continue

            # Calculate movement
            direction = uav["target_pos"] - uav["pos"]
            distance = np.linalg.norm(direction)

            # Check if we need to queue this UAV
            row_uavs = self.row_assignments[uav["target_row"]]
            uav_index = row_uavs.index(uav)

            if distance >= self.MOVE_SPEED:
                # Normalize direction and apply speed
                movement = direction / distance * self.MOVE_SPEED
                uav["pos"] += movement

                # Update current row based on position
                uav["current_row"] = int(uav["pos"][1] / self.grid_size)
            else:
                uav["in_transition"] = False

            # Move forward if not in transition and not queued
            if not uav["in_transition"]:
                # If this UAV is not the first in queue, adjust target position
                if all(uav["status"] == "active" for uav in row_uavs[:uav_index]):
                    # Each subsequent UAV should stay one grid size behind the previous
                    desired_x = int(self.attack_x - (uav_index * self.grid_size))

                    # Update target position if needed
                    if uav["target_pos"][0] > desired_x:
                        uav["target_pos"][0] = desired_x

                # If all UAVs are in position, let them enter building one by one
                if all(self._is_approached(u) for u in row_uavs):
                    target_idx = self.window_mapping[uav["target_row"]]
                    drone_cost = self.env.drone_cost[target_idx]
                    if uav_index < drone_cost:
                        uav["target_pos"][0] = int(
                            self.building_start
                            + (drone_cost - uav_index) * self.grid_size / 2
                        )
                        uav["in_transition"] = True
                else:
                    # Normal forward movement until reaching queue position
                    uav["pos"][0] += self.MOVE_SPEED
                    uav["target_pos"][0] = uav["pos"][0]

            if uav["pos"][0] >= self.building_start:
                uav["entered"] = True

            # Check for engagement range
            if (
                uav["entered"]
                and self._is_reached(uav)
                and all(uav["status"] != "active" for uav in row_uavs[:uav_index])
            ):
                self.handle_engagement(uav)

    def handle_engagement(self, uav):
        """Handle UAV-threat engagement"""
        # Check for env's simulation result
        agent_id = uav["id"]
        row = uav["current_row"]
        index = self.window_mapping[row]
        r_index = self.row_assignments[row].index(uav)

        if r_index < self.env.drone_cost[index]:
            uav["status"] = "destroyed"
        elif self.env.truncations[agent_id]:
            uav["status"] = "damaged"

        # Whether successfully destroyed the threat
        destroyed = self.env.eliminated_threats[index]
        if destroyed:
            self.enemies[row]["status"] = "destroyed"

        uav["pos"][0] = uav["target_pos"][0]

    def update_threats(self, agent: Agent, state: dict = None):
        """Update threat distribution periodically

        Args:
            agent (Agent): The agent to perform reallocation
            state (dict, optional): The state to reset the environment. Defaults to None.

        Returns:
            bool: Whether reallocation is triggered
        """
        state = state or {}

        if (
            (self.frame_count % self.REALLOCATION_INTERVAL == 0)
            and (not self._is_approached())
            and (not any(uav["in_transition"] for uav in self.uavs))
            and (self.realloc_cnt < self.max_reallocation)
        ):
            if not state:
                self.env.reset()
            else:
                for key, value in state.items():
                    setattr(self.env.unwrapped, key, value)

            env_state = deepcopy(vars(self.env.unwrapped))
            window_map = state.get("window_mapping", None)
            if window_map is not None:
                self.window_mapping = window_map
            else:
                self.window_mapping = np.random.permutation(self.rows)

            self.window_threats = self.env.threat_levels[self.window_mapping]
            self.enemies = [
                {
                    "row": row,
                    "status": (
                        "active"
                        if self.env.actual_threats[self.window_mapping[row]]
                        else "none"
                    ),
                }
                for row in range(self.rows)
            ]
            # Trigger reallocation
            self.perform_allocation(agent)
            self.realloc_cnt += 1
            env_state["window_mapping"] = self.window_mapping
            self.state_hist.append(env_state)
            return True

        return False

    def draw(self):
        """Draw current game state"""
        self.screen.fill((200, 200, 200))

        # Draw grid and building
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(
                    col * self.grid_size,
                    row * self.grid_size,
                    self.grid_size,
                    self.grid_size,
                )
                if col >= self.building_col:
                    color = get_threat_color(self.window_threats[row])
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

        # Draw enemies when approached
        if self._is_approached():
            for enemy in self.enemies:
                if enemy["status"] != "none":
                    rect = pygame.Rect(
                        (self.building_col + 2) * self.grid_size,
                        enemy["row"] * self.grid_size,
                        self.grid_size,
                        self.grid_size,
                    )
                    color = (
                        (0, 0, 255) if enemy["status"] == "active" else (128, 128, 128)
                    )
                    pygame.draw.rect(
                        self.screen,
                        color,
                        pygame.Rect(
                            rect.centerx - self.grid_size // 3,
                            rect.centery - self.grid_size // 3,
                            self.grid_size * 2 // 3,
                            self.grid_size * 2 // 3,
                        ),
                    )

        # Draw UAVs
        for uav in self.uavs:
            radius = self.grid_size // 3
            pos = uav["pos"].astype(int)
            # black outline
            pygame.draw.circle(self.screen, (0, 0, 0), pos, radius + 2)
            # UAV circle
            color = (255, 0, 0) if uav["status"] == "active" else (128, 128, 128)
            pygame.draw.circle(self.screen, color, pos, radius)

        self._draw_statistics()
        self._draw_legend()
        pygame.display.flip()

    def _is_approached(self, uav=None):
        if uav is None:
            # Check if any UAV is within attack range
            return any(uav["pos"][0] >= self.attack_x for uav in self.uavs)
        else:
            r_index = self.row_assignments[uav["target_row"]].index(uav)
            return uav["pos"][0] >= self.attack_x - r_index * self.grid_size

    def _is_reached(self, uav):
        return np.linalg.norm(uav["pos"] - uav["target_pos"]) < self.MOVE_SPEED

    def _draw_statistics(self):
        """Draw episode statistics at the top-right corner."""
        stats = [
            f"无人机: {self.env.num_drones}",
            f"威胁阵位: {self.env.num_threats}",
            f"敌方单位: {self.env.num_actual_threat}",
        ]

        info = next(iter(self.env.infos.values()))
        if info:
            stats.extend(
                [
                    f"覆盖率: {info['coverage']:.2f}",
                    f"打击成功率: {info['success_rate']:.2f}",
                    f"击毁敌方数: {info['threat_destroyed']}",
                    f"我方损失数: {info['drone_lost']}",
                    f"战损比: {info['kd_ratio']:.2f}",
                    f"剩余敌方数: {info['num_remaining_threat']}",
                ]
            )

        # Create a semi-transparent background for statistics
        text_background = pygame.Surface((200, 400))
        text_background.set_alpha(200)
        text_background.fill((220, 220, 220))
        self.screen.blit(text_background, (self.width - 200, 50))

        # Draw each statistic entry
        for i, text in enumerate(stats):
            stat_text = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(stat_text, (self.width - 195, 60 + i * 40))

    def _draw_legend(self):
        """Draw legend in bottom right corner"""
        box_size = 20
        spacing = 30
        text_offset = 30
        legend_width = 190
        legend_height = 220

        # Calculate position from right side
        legend_x = self.screen.get_width() - legend_width
        legend_y = self.screen.get_height() - 250

        # Background for legend
        legend_background = pygame.Rect(
            legend_x - 5, legend_y - 5, legend_width, legend_height
        )
        pygame.draw.rect(self.screen, (220, 220, 220), legend_background)
        pygame.draw.rect(self.screen, (100, 100, 100), legend_background, 1)

        # UAV (Red circle)
        pygame.draw.circle(
            self.screen,
            (255, 0, 0),
            (legend_x + box_size // 2, legend_y + box_size // 2),
            box_size // 2,
        )
        text = self.font.render("无人机-存活", True, (0, 0, 0))
        self.screen.blit(text, (legend_x + text_offset, legend_y))

        # Enemy (Blue square)
        pygame.draw.rect(
            self.screen,
            (0, 0, 255),
            pygame.Rect(legend_x, legend_y + spacing, box_size, box_size),
        )
        text = self.font.render("敌人-存活", True, (0, 0, 0))
        self.screen.blit(text, (legend_x + text_offset, legend_y + spacing))

        # Destroyed units (Gray)
        pygame.draw.circle(
            self.screen,
            (128, 128, 128),
            (legend_x + box_size // 2, legend_y + spacing * 2 + box_size // 2),
            box_size // 2,
        )
        pygame.draw.rect(
            self.screen,
            (128, 128, 128),
            pygame.Rect(
                legend_x + box_size + 5, legend_y + spacing * 2, box_size, box_size
            ),
        )
        text = self.font.render("损毁单位", True, (0, 0, 0))
        self.screen.blit(
            text, (legend_x + text_offset + box_size + 5, legend_y + spacing * 2)
        )

        # Threat level colors
        threat_colors = [
            (get_threat_color(0.0), "无威胁阵位"),
            (get_threat_color(0.1), "低威胁阵位"),
            (get_threat_color(0.4), "中威胁阵位"),
            (get_threat_color(0.8), "高威胁阵位"),
        ]

        for i, (color, label) in enumerate(threat_colors):
            y_pos = legend_y + spacing * 3 + (i * spacing)
            pygame.draw.rect(
                self.screen, color, pygame.Rect(legend_x, y_pos, box_size, box_size)
            )
            text = self.font.render(label, True, (0, 0, 0))
            self.screen.blit(text, (legend_x + text_offset, y_pos))

    def run(self, agent: Agent):
        if self.recording:
            recorder = PygameRecord(
                os.path.join(
                    self.record_dir,
                    f"{type(agent).__name__}_{time.strftime('%Y%m%d-%H%M%S')}.gif",
                ),
                self.fps,
            )

        self.reset_simulation()
        running = True
        while running:
            self.frame_count += 1
            end = handle_events()

            self.update_threats(agent)
            self.update_positions()
            self.draw()
            if self.recording:
                recorder.add_frame()

            self.clock.tick(60)
            if end:
                running = False

        if self.recording:
            print("Saving recording...")
            recorder.save()

    def replay(self, agent: Agent, state_hist: list):
        if self.recording:
            recorder = PygameRecord(
                os.path.join(
                    self.record_dir,
                    f"{type(agent).__name__}_{time.strftime('%Y%m%d-%H%M%S')}.gif",
                ),
                self.fps,
            )

        idx = 0
        state = state_hist[idx]
        self.reset_simulation(state)
        idx = min(idx + 1, len(state_hist) - 1)

        running = True
        while running:
            self.frame_count += 1
            end = handle_events()

            state = state_hist[idx]
            if self.update_threats(agent, state):
                idx = min(idx + 1, len(state_hist) - 1)

            self.update_positions()
            self.draw()
            if self.recording:
                recorder.add_frame()

            self.clock.tick(60)
            if end:
                running = False

        if self.recording:
            print("Saving recording...")
            recorder.save()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    from target_assign_rl.target_assign_agent import RandomAgent

    demo = ReallocationDemo(recording=True, fps=60, max_reallocation=2)
    agent1 = RuleAgent(20)
    agent2 = RandomAgent(20)

    try:
        while True:
            demo.run(agent1)

            # Replay with random agent
            state_hist = demo.state_hist.copy()
            demo.replay(agent2, state_hist)
    finally:
        demo.close()
