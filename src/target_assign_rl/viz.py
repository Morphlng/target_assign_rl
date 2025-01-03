import colorsys
import os
import sys
import pygame
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from target_assign_rl.target_assign_aec import TaskAllocationAEC


def get_threat_color(threat_level):
    if threat_level == 0:
        return (240, 240, 240)

    hue = 0.15 - (threat_level * 0.15)
    saturation = 0.5 + (threat_level * 0.5)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


def handle_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            return True
    return False


class BaseVisualizer:
    def __init__(self, config):
        self.config = config or {}
        self.screen_width = self.config.get("width", 1400)
        self.screen_height = self.config.get("height", 800)
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.font = pygame.font.Font(None, 24)

    def render(self, env_state: dict):
        raise NotImplementedError("Render method must be implemented by subclasses")

    def save_screenshot(self, timestamps, episode_count, agent_selection):
        folder = f"logs/{timestamps}_{episode_count}"
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        file_path = os.path.abspath(os.path.join(folder, f"{agent_selection}.png"))
        pygame.image.save(self.screen, file_path)

    def close(self):
        pygame.quit()


class BuildingVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        pygame.display.set_caption("Task Allocation - Drone Scheduling")

        self.x_padding, self.y_padding = 250, 80
        grid_width = self.screen_width - 2 * self.x_padding
        grid_height = self.screen_height - 2 * self.y_padding
        self.building_rect = pygame.Rect(
            self.x_padding, self.y_padding, grid_width, grid_height
        )

    def render(self, env_state):
        self.screen.fill((255, 255, 255))
        self._draw_building_and_windows(env_state)
        self._draw_flying_drones(env_state)
        self._draw_statistics(env_state)
        self._draw_legend(env_state)
        pygame.display.flip()

        if env_state["is_first_agent"] and self.config.get("pause_at_end", False):
            self.screen.blit(
                self.font.render("Press SPACE to continue", True, (0, 0, 0)),
                (self.screen_width // 2 - 100, 10),
            )
            pygame.display.flip()

            while not handle_events():
                pygame.time.wait(10)
        else:
            handle_events()

    def _draw_building_and_windows(self, env_state):
        pygame.draw.rect(self.screen, (200, 200, 200), self.building_rect)

        for idx, window in enumerate(env_state["windows"]):
            pygame.draw.rect(self.screen, (50, 50, 50), window, 2)
            if env_state["threat_levels"][idx] > 0:
                threat_prob = env_state["threat_levels"][idx]
                color = get_threat_color(threat_prob)
                prob_text = self.font.render(f"{threat_prob:.2f}", True, color)
                text_rect = prob_text.get_rect(center=(window.centerx, window.y + 20))
                self.screen.blit(prob_text, text_rect)

                if env_state["actual_threats"][idx]:
                    color = (
                        (255, 0, 0)
                        if not env_state["eliminated_threats"][idx]
                        else (0, 255, 0)
                    )
                    threat_icon = self.font.render("!", True, color)
                    threat_icon_rect = threat_icon.get_rect(center=window.center)
                    self.screen.blit(threat_icon, threat_icon_rect)

    def _draw_flying_drones(self, env_state):
        for drone in env_state["flying_drones"]:
            pos = drone["current_pos"]
            if drone.get("status") == "destroyed":
                color = (255, 0, 0)
            elif drone.get("status") == "damaged":
                color = (255, 165, 0)
            else:
                color = (0, 100, 255)
            pygame.draw.circle(self.screen, color, (int(pos[0]), int(pos[1])), 10)

    def _draw_statistics(self, env_state):
        """Draw episode statistics at the top-right corner."""
        stats = [
            f"Episode: {env_state['episode_count']}",
            f"Drones: {env_state['num_drones']}",
            f"Threats: {env_state['num_threats']}",
            f"Actual Threats: {env_state['num_actual_threat']}",
        ]

        info = next(iter(env_state["infos"].values()))
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
        text_background = pygame.Surface((220, 300))
        text_background.set_alpha(200)
        text_background.fill((220, 220, 220))
        self.screen.blit(text_background, (self.screen_width - 240, 50))

        # Draw each statistic entry
        for i, text in enumerate(stats):
            stat_text = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(stat_text, (self.screen_width - 230, 60 + i * 20))

    def _draw_legend(self, env_state):
        """Draw the legend in the bottom-right corner of the screen."""
        legend_items = [
            ("Active Drones", (0, 100, 255)),  # Blue
            ("Destroyed Drones", (255, 0, 0)),  # Red
            ("Damaged Drones", (255, 165, 0)),  # Orange
            ("Actual Threat (!)", (0, 0, 0)),
            ("Eliminate Threat (!)", (0, 0, 0)),
        ]

        # Calculate legend position (bottom-right corner)
        legend_x = self.screen_width - 200
        legend_y = self.screen_height - (len(legend_items) * 30) - 50

        # Draw each legend item
        for i, (text, color) in enumerate(legend_items):
            if text == "Actual Threat (!)":
                threat_icon = self.font.render("!", True, (255, 0, 0))
                self.screen.blit(threat_icon, (legend_x + 10, legend_y + i * 30))
            elif text == "Eliminate Threat (!)":
                threat_icon = self.font.render("!", True, (0, 255, 0))
                self.screen.blit(threat_icon, (legend_x + 10, legend_y + i * 30))
            else:
                pygame.draw.circle(
                    self.screen, color, (legend_x + 15, legend_y + i * 30 + 10), 10
                )
            legend_text = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(legend_text, (legend_x + 40, legend_y + i * 30))


class GridVisualizer(BaseVisualizer):
    def __init__(self, config):
        super().__init__(config)
        pygame.display.set_caption("Drone Task Allocation")

    def render(self, env_state):
        self.screen.fill((240, 240, 240))
        self._draw_allocation(env_state)
        self._draw_statistics(env_state)
        self._draw_legend(env_state)

        pygame.display.flip()
        if env_state["is_first_agent"] and self.config.get("pause_at_end", False):
            self.screen.blit(
                self.font.render("Press SPACE to continue", True, (0, 0, 0)),
                (self.screen_width // 2 - 100, 10),
            )
            pygame.display.flip()

            while not handle_events():
                pygame.time.wait(10)
        else:
            handle_events()

    def _draw_allocation(self, env_state):
        grid_width = self.screen_width - 300
        grid_height = self.screen_height - 100
        cell_width = grid_width // env_state["num_threats"]
        cell_height = grid_height // 3

        for i in range(env_state["num_threats"]):
            x = i * cell_width + 250

            # Threat level
            color = get_threat_color(env_state["threat_levels"][i])
            pygame.draw.rect(self.screen, color, (x, 50, cell_width, cell_height))

            # Write threat level in the cell
            if self.config.get("show_threat_level"):
                threat_text = self.font.render(
                    f"{env_state['threat_levels'][i]:.2f}", True, (0, 0, 0)
                )
                text_rect = threat_text.get_rect(
                    center=(x + cell_width // 2, 50 + cell_height // 2)
                )
                self.screen.blit(threat_text, text_rect)

            # Actual threat
            if env_state["actual_threats"][i]:
                threat_surface = pygame.Surface((cell_width, cell_height))
                threat_surface.set_alpha(128)
                threat_surface.fill((100, 200, 255))
                self.screen.blit(threat_surface, (x, 50 + cell_height))

            # Success
            if env_state["eliminated_threats"][i]:
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (x + 5, 50 + cell_height + 5),
                    (x + cell_width - 5, 50 + cell_height * 2 - 5),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    (255, 0, 0),
                    (x + cell_width - 5, 50 + cell_height + 5),
                    (x + 5, 50 + cell_height * 2 - 5),
                    3,
                )

        counter = np.zeros(env_state["num_threats"])
        for agent, pos in env_state["allocation_map"].items():
            counter[pos] = min(counter[pos] + 1, 10)
            x = pos * cell_width + 250
            color = (0, 100, 255)  # Default blue for active drones
            if env_state["terminations"].get(agent, False):
                color = (255, 0, 0)  # Red for destroyed drones
            elif env_state["truncations"].get(agent, False):
                color = (255, 165, 0)  # Orange for damaged drones
            pygame.draw.circle(
                self.screen,
                color,
                (x + cell_width // 2, 50 + cell_height * 2 + counter[pos] * 20 + 10),
                5,
            )

        # Draw grid lines
        for i in range(env_state["num_threats"] + 1):
            x = i * cell_width + 250
            pygame.draw.line(
                self.screen, (0, 0, 0), (x, 50), (x, 50 + cell_height * 2), 2
            )

        for i in range(3):
            y = i * cell_height + 50
            pygame.draw.line(
                self.screen, (0, 0, 0), (250, y), (self.screen_width - 50, y), 2
            )

    def _draw_statistics(self, env_state):
        texts = [
            f"Episode: {env_state['episode_count']}",
            f"Drones: {env_state['num_drones']}",
            f"Threats: {env_state['num_threats']}",
            f"Actual Threats: {env_state['num_actual_threat']}",
        ]

        info = next(iter(env_state["infos"].values()))
        if info:
            texts.extend(
                [
                    f"Coverage: {info['coverage']:.2f}",
                    f"Success Rate: {info['success_rate']:.2f}",
                    f"Threats Destroyed: {info['threat_destroyed']}",
                    f"Drones Lost: {info['drone_lost']}",
                    f"K/D Ratio: {info['kd_ratio']:.2f}",
                    f"Remaining Threats: {info['num_remaining_threat']}",
                ]
            )

        # Create a semi-transparent surface for text background
        text_surface = pygame.Surface((210, self.screen_height))
        text_surface.set_alpha(200)
        text_surface.fill((220, 220, 220))
        self.screen.blit(text_surface, (0, 0))

        for i, text in enumerate(texts):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))

    def _draw_legend(self, env_state):
        legend_items = [
            ("Threat Level", get_threat_color(0.5)),
            ("Actual Threat", (100, 200, 255)),
            ("Active Drone", (0, 100, 255)),
            ("Damaged Drone", (255, 165, 0)),
            ("Destroyed Drone", (255, 0, 0)),
            ("Eliminate", (255, 0, 0)),
        ]

        legend_x = 10
        legend_y = self.screen_height - 150
        for i, (text, color) in enumerate(legend_items):
            if text == "Threat Level":
                # Draw a gradient for threat level
                gradient_width = 40
                for j in range(gradient_width):
                    threat_level = j / (gradient_width - 1)
                    gradient_color = get_threat_color(threat_level)
                    pygame.draw.line(
                        self.screen,
                        gradient_color,
                        (legend_x + 10 + j, legend_y + 10 + i * 20),
                        (legend_x + 10 + j, legend_y + 25 + i * 20),
                    )
            elif text == "Eliminate":
                pygame.draw.line(
                    self.screen,
                    color,
                    (legend_x + 10, legend_y + 15 + i * 20),
                    (legend_x + 30, legend_y + 25 + i * 20),
                    3,
                )
                pygame.draw.line(
                    self.screen,
                    color,
                    (legend_x + 30, legend_y + 15 + i * 20),
                    (legend_x + 10, legend_y + 25 + i * 20),
                    3,
                )
            else:
                if text == "Actual Threat":
                    # Draw a semi-transparent rectangle for actual threat
                    threat_surface = pygame.Surface((15, 15))
                    threat_surface.set_alpha(128)
                    threat_surface.fill(color)
                    self.screen.blit(
                        threat_surface, (legend_x + 10, legend_y + 10 + i * 20)
                    )
                else:
                    # Draw a circle for drones
                    pygame.draw.circle(
                        self.screen,
                        color,
                        (legend_x + 17, legend_y + 17 + i * 20),
                        5,
                    )

            legend_text = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(legend_text, (legend_x + 55, legend_y + 10 + i * 20))


def create_visualizer(render_mode, config):
    if render_mode in ["human", "building"]:
        return BuildingVisualizer(config)
    elif render_mode == "grid":
        return GridVisualizer(config)
    else:
        return None
