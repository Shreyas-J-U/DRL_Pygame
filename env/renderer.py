# ============================================================================
# Pygame Renderer Module
# ============================================================================
# Handles visualization of environment state, entities, and predictions.
# ============================================================================

import pygame
import numpy as np
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, LANE_Y, ROAD_HEIGHT,
    CAR_COLOR, PED_COLOR, BACKGROUND_COLOR, ROAD_COLOR,
    PREDICTION_LINE_COLOR, PREDICTION_POINT_COLOR,
    FONT_SIZE, FPS, SHOW_PREDICTIONS, PREDICTION_STEPS
)
from utils.prediction import predict_trajectory


class Renderer:
    """Handles all rendering operations."""
    
    def __init__(self, title="Prediction-Aware Autonomous Driving"):
        """Initialize pygame renderer."""
        pygame.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption(title)
        
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, FONT_SIZE)
        self.font_small = pygame.font.Font(None, FONT_SIZE - 8)
        
        self.frame_count = 0
    
    def render(self, car, pedestrians, prediction_enabled=True, 
               episode_info=None, step_count=0, episode_reward=0):
        """
        Render complete scene.
        
        Args:
            car: Car object
            pedestrians: List of Pedestrian objects
            prediction_enabled: Whether to show predictions
            episode_info: Optional dict with episode information
            step_count: Current step in episode
            episode_reward: Cumulative reward so far
        """
        # Clear screen
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw road
        self._draw_road()
        
        # Draw pedestrians and their predictions
        self._draw_pedestrians(pedestrians)
        if prediction_enabled and SHOW_PREDICTIONS:
            self._draw_predictions(pedestrians)
        
        # Draw car
        self._draw_car(car)
        
        # Draw HUD (heads-up display)
        self._draw_hud(prediction_enabled, step_count, episode_reward, episode_info)
        
        pygame.display.flip()
        self.clock.tick(FPS)
        self.frame_count += 1
    
    def _draw_road(self):
        """Draw the road."""
        road_top = LANE_Y - ROAD_HEIGHT / 2
        pygame.draw.rect(self.screen, ROAD_COLOR, 
                        (0, road_top, SCREEN_WIDTH, ROAD_HEIGHT))
        
        # Draw center line
        pygame.draw.line(self.screen, (100, 100, 100),
                        (0, LANE_Y), (SCREEN_WIDTH, LANE_Y), 2)
    
    def _draw_car(self, car):
        """Draw the car."""
        pygame.draw.rect(self.screen, CAR_COLOR,
                        (car.x, car.y - car.height/2, car.width, car.height))
        
        # Draw direction indicator (small triangle)
        pygame.draw.polygon(self.screen, (0, 200, 0),
                           [(car.x + car.width, car.y - car.height/2),
                            (car.x + car.width, car.y + car.height/2),
                            (car.x + car.width + 5, car.y)])
    
    def _draw_pedestrians(self, pedestrians):
        """Draw all pedestrians as circles."""
        for ped in pedestrians:
            pygame.draw.circle(self.screen, PED_COLOR,
                             (int(ped.x), int(ped.y)), ped.radius)
            
            # Draw velocity indicator
            vel_scale = 2
            vx_scaled = int(ped.vx * vel_scale)
            vy_scaled = int(ped.vy * vel_scale)
            pygame.draw.line(self.screen, (200, 0, 0),
                           (int(ped.x), int(ped.y)),
                           (int(ped.x + vx_scaled), int(ped.y + vy_scaled)), 1)
    
    def _draw_predictions(self, pedestrians):
        """Draw predicted trajectories for pedestrians."""
        for ped in pedestrians:
            trajectory = predict_trajectory(ped, PREDICTION_STEPS)
            
            # Draw prediction line
            points = [(int(x), int(y)) for x, y in trajectory]
            
            if len(points) > 1:
                # Draw dotted line
                for i in range(len(points) - 1):
                    # Alternating dots and gaps
                    if i % 2 == 0:
                        pygame.draw.line(self.screen, PREDICTION_LINE_COLOR,
                                       points[i], points[i+1], 1)
            
            # Draw prediction points
            for i, (x, y) in enumerate(points):
                color_intensity = int(200 * (1 - i / len(points)))
                color = (color_intensity, color_intensity, 255)
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 3)
    
    def _draw_hud(self, prediction_enabled, step_count, episode_reward, episode_info):
        """Draw heads-up display with information."""
        # Prediction status
        pred_text = "Prediction: ON" if prediction_enabled else "Prediction: OFF"
        pred_color = (0, 255, 0) if prediction_enabled else (255, 100, 0)
        pred_surface = self.font_large.render(pred_text, True, pred_color)
        self.screen.blit(pred_surface, (10, 10))
        
        # Step counter
        step_text = f"Step: {step_count}"
        step_surface = self.font_small.render(step_text, True, (255, 255, 255))
        self.screen.blit(step_surface, (10, 40))
        
        # Reward
        reward_text = f"Reward: {episode_reward:.1f}"
        reward_color = (0, 255, 0) if episode_reward > 0 else (255, 0, 0)
        reward_surface = self.font_small.render(reward_text, True, reward_color)
        self.screen.blit(reward_surface, (10, 65))
        
        # FPS
        fps_text = f"FPS: {int(self.clock.get_fps())}"
        fps_surface = self.font_small.render(fps_text, True, (200, 200, 200))
        self.screen.blit(fps_surface, (SCREEN_WIDTH - 150, 10))
        
        # Additional info
        if episode_info:
            y_offset = SCREEN_HEIGHT - 100
            for key, value in episode_info.items():
                info_text = f"{key}: {value}"
                info_surface = self.font_small.render(info_text, True, (200, 200, 200))
                self.screen.blit(info_surface, (10, y_offset))
                y_offset += 25
    
    def close(self):
        """Close pygame window."""
        pygame.quit()
    
    def handle_events(self):
        """
        Handle pygame events.
        
        Returns:
            should_quit: Boolean indicating if user wants to close
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        
        return False
