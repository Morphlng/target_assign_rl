import numpy as np
import pygame

from target_assign_rl import Agent, RandomAgent, raw_env
from target_assign_rl.viz import get_threat_color, handle_events


class ReallocationDemo:
    def __init__(self, env_config: dict = None):
        env_config = env_config or {}

        # Display settings
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 800
        self.GRID_SIZE = 40
        self.COLS = 25
        self.ROWS = 20
        self.BUILDING_START = self.COLS - 5

        # Movement settings
        self.MOVE_SPEED = 2  # pixels per frame
        self.REALLOCATION_INTERVAL = 100  # frames
        self.ATTACK_RANGE = 1 * self.GRID_SIZE

        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("UAV Strike Process")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self.env = raw_env(env_config)
        self.reset_simulation()

    def reset_simulation(self):
        """Initialize a new strike mission"""
        self.env.reset()
        self.frame_count = 0

        # Initialize UAVs at left side
        self.uavs = []
        spacing = self.WINDOW_HEIGHT / self.env.num_drones
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
                }
            )

        # Initialize threat state
        self.window_mapping = np.random.permutation(self.ROWS)
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
            for row in range(self.ROWS)
        ]

        self.perform_allocation()

    def perform_allocation(self, agent: Agent = None):
        """Allocate UAVs to targets based on current threat levels"""
        if agent is None:
            agent = RandomAgent(self.ROWS)

        for uav in self.uavs:
            if uav["status"] != "active":
                continue

            obs = self.env.observe(None)
            action_mask = self.env.action_mask(None)
            action = agent.predict(obs, action_mask)
            self.env.step(action)
            target_row = np.where(self.window_mapping == action)[0][0]

            if target_row != uav["target_row"]:
                uav["target_row"] = target_row
                uav["in_transition"] = True
                uav["target_pos"] = np.array(
                    [
                        uav["pos"][0],  # Keep current x position
                        target_row * self.GRID_SIZE
                        + self.GRID_SIZE / 2,  # Center of target row
                    ]
                )

    def update_positions(self):
        """Update UAV positions"""
        all_reached = True

        # Create a dictionary to track UAVs assigned to each row
        row_assignments: "dict[int, list]" = {}
        for uav in self.uavs:
            if uav["target_row"] is not None:
                if uav["target_row"] not in row_assignments:
                    row_assignments[uav["target_row"]] = []
                row_assignments[uav["target_row"]].append(uav)

        # Sort UAVs in each row by x-position (rightmost first)
        for row in row_assignments:
            row_assignments[row].sort(key=lambda u: (-u["pos"][0], u["id"]))

        for uav in self.uavs:
            if uav["status"] != "active":
                continue

            # Calculate movement
            direction = uav["target_pos"] - uav["pos"]
            distance = np.linalg.norm(direction)

            # Check if we need to queue this UAV
            if uav["target_row"] is not None:
                row_uavs = row_assignments[uav["target_row"]]
                uav_index = row_uavs.index(uav)

                # Calculate desired x-position based on queue position
                attack_x = int(self.BUILDING_START * self.GRID_SIZE - self.ATTACK_RANGE)

                # If this UAV is not the first in queue, adjust target position
                if uav_index > 0:
                    # Each subsequent UAV should stay one grid size behind the previous
                    desired_x = int(attack_x - (uav_index * self.GRID_SIZE))

                    # Update target position if needed
                    if uav["target_pos"][0] > desired_x:
                        uav["target_pos"][0] = desired_x

                    # If leading UAV is in attack range, force queuing
                    leading_uav = row_uavs[0]
                    if leading_uav["pos"][0] >= attack_x:
                        uav["target_pos"][0] = desired_x

            if distance > self.MOVE_SPEED:  # If not at target
                all_reached = False
                # Normalize direction and apply speed
                direction = direction / distance * self.MOVE_SPEED * 2
                uav["pos"] += direction

                # Update current row based on position
                uav["current_row"] = int(uav["pos"][1] / self.GRID_SIZE)
            else:
                uav["in_transition"] = False

            # Move forward if not in transition and not queued
            if not uav["in_transition"]:
                # Check if we're in a queue and at our desired position
                should_move_forward = True
                if uav["target_row"] is not None:
                    row_uavs = row_assignments[uav["target_row"]]
                    uav_index = row_uavs.index(uav)
                    if uav_index > 0:  # If not the lead UAV
                        leading_uav = row_uavs[0]
                        if leading_uav["pos"][0] >= (
                            self.BUILDING_START * self.GRID_SIZE - self.ATTACK_RANGE
                        ):
                            should_move_forward = False
                    else:
                        # If we're the lead UAV, check if we're at the target position
                        if uav["pos"][0] >= attack_x:
                            should_move_forward = False

                if should_move_forward:
                    uav["pos"][0] += self.MOVE_SPEED
                    uav["target_pos"][0] = uav["pos"][0]

            # Check for engagement range
            if (
                self.BUILDING_START * self.GRID_SIZE - uav["pos"][0]
            ) <= self.ATTACK_RANGE:
                self.handle_engagement(uav)

        return all_reached

    def handle_engagement(self, uav):
        """Handle UAV-threat engagement"""
        # Check for env's simulation result
        agent_id = uav["id"]
        if self.env.terminations[agent_id]:
            uav["status"] = "destroyed"
        elif self.env.truncations[agent_id]:
            uav["status"] = "damaged"

        # Whether successfully destroyed the threat
        row = uav["current_row"]
        index = self.window_mapping[row]
        destroyed = self.env.eliminated_threats[index]
        if destroyed:
            self.enemies[row]["status"] = "destroyed"

    def update_threats(self, agent: Agent):
        """Update threat distribution periodically"""
        if (
            (self.frame_count % self.REALLOCATION_INTERVAL == 0)
            and (not self._is_approached())
            and (not any(uav["in_transition"] for uav in self.uavs))
        ):
            # Generate new threat distribution
            self.env.reset()
            self.window_mapping = np.random.permutation(self.ROWS)
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
                for row in range(self.ROWS)
            ]
            # Trigger reallocation
            self.perform_allocation(agent)

    def draw(self):
        """Draw current game state"""
        self.screen.fill((200, 200, 200))

        # Draw grid and building
        for row in range(self.ROWS):
            for col in range(self.COLS):
                rect = pygame.Rect(
                    col * self.GRID_SIZE,
                    row * self.GRID_SIZE,
                    self.GRID_SIZE,
                    self.GRID_SIZE,
                )
                if col >= self.BUILDING_START:
                    color = get_threat_color(self.window_threats[row])
                    pygame.draw.rect(self.screen, color, rect)
                else:
                    pygame.draw.rect(self.screen, (150, 150, 150), rect, 1)

        # Draw enemies when approached
        if self._is_approached():
            for enemy in self.enemies:
                if enemy["status"] != "none":
                    rect = pygame.Rect(
                        (self.BUILDING_START + 2) * self.GRID_SIZE,
                        enemy["row"] * self.GRID_SIZE,
                        self.GRID_SIZE,
                        self.GRID_SIZE,
                    )
                    color = (0, 0, 0) if enemy["status"] == "active" else (0, 255, 0)
                    pygame.draw.circle(
                        self.screen, color, rect.center, self.GRID_SIZE // 3
                    )

        # Draw UAVs
        for uav in self.uavs:
            color = (50, 50, 255) if uav["status"] == "active" else (255, 0, 0)
            pygame.draw.circle(
                self.screen, color, uav["pos"].astype(int), self.GRID_SIZE // 3
            )

        self._draw_statistics()
        pygame.display.flip()

    def _is_approached(self):
        return any(
            uav["pos"][0] >= self.BUILDING_START * self.GRID_SIZE - self.ATTACK_RANGE
            for uav in self.uavs
        )

    def _draw_statistics(self):
        """Draw episode statistics at the top-right corner."""
        stats = [
            f"Drones: {self.env.num_drones}",
            f"Threats: {self.env.num_threats}",
            f"Actual Threats: {self.env.num_actual_threat}",
        ]

        info = next(iter(self.env.infos.values()))
        if info:
            stats.extend(
                [
                    f"Coverage: {info['coverage']:.2f}",
                    f"Success Rate: {info['success_rate']:.2f}",
                    f"Threats Destroyed: {info['threat_destroyed']}",
                    f"Drones Lost: {info['drone_lost']}",
                    f"K/D Ratio: {info['kd_ratio']:.2f}",
                    f"Remaining Threats: {info['num_remaining_threat']}",
                ]
            )

        # Create a semi-transparent background for statistics
        text_background = pygame.Surface((200, 300))
        text_background.set_alpha(200)
        text_background.fill((220, 220, 220))
        self.screen.blit(text_background, (self.WINDOW_WIDTH - 200, 50))

        # Draw each statistic entry
        for i, text in enumerate(stats):
            stat_text = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(stat_text, (self.WINDOW_WIDTH - 190, 60 + i * 20))

    def run(self, agent: Agent):
        self.reset_simulation()
        running = True
        while running:
            self.frame_count += 1
            end = handle_events()

            self.update_threats(agent)
            self.update_positions()
            self.draw()

            self.clock.tick(60)
            if end:
                running = False

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    from target_assign_rl.target_assign_agent import RuleAgent

    demo = ReallocationDemo()
    agent = RuleAgent(20)

    try:
        while True:
            demo.run(agent)
    finally:
        demo.close()
