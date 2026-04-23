# ============================================================================
# Gymnasium Environment: Prediction-Aware Autonomous Driving
# ============================================================================
# Custom Gymnasium environment for DRL-based autonomous driving with
# trajectory prediction integration.
# ============================================================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from config import (
    NUM_ACTIONS, OBSERVATION_DIM, DT, MAX_STEPS, SCREEN_WIDTH,
    PREDICTION_STEPS, NUM_PEDESTRIANS, PED_SPAWN_X_MIN, PED_SPAWN_X_MAX
)
from env.entities import Car, PedestrianManager
from utils.state import get_observation
from utils.reward import compute_reward


class PredictionAwareCarEnv(gym.Env):
    """
    Gymnasium environment for autonomous driving with trajectory prediction.
    
    The agent receives observations that include:
    - Current car state (position, velocity)
    - Current pedestrian positions
    - Predicted future pedestrian positions (if prediction_enabled=True)
    
    Actions:
    - 0: Accelerate
    - 1: Brake
    - 2: Move up (lane change)
    - 3: Move down (lane change)
    
    Rewards:
    - Forward progress
    - Safe distance maintenance
    - Collision penalty
    - Braking penalty
    - Prediction bonus
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, render_mode=None, prediction_enabled=True):
        """
        Initialize environment.
        
        Args:
            render_mode: Render mode ('human' or None)
            prediction_enabled: Whether to include predicted positions in obs
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.prediction_enabled = prediction_enabled
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )
        
        # Environment state
        self.car = None
        self.pedestrian_manager = None
        self.step_count = 0
        self.episode_reward = 0
        self.renderer = None
        
        # Initialize entities
        self._reset_entities()
    
    def _reset_entities(self):
        """Initialize/reset car and pedestrians."""
        self.car = Car()
        self.pedestrian_manager = PedestrianManager(NUM_PEDESTRIANS)
        self.step_count = 0
        self.episode_reward = 0
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed)
        
        self._reset_entities()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: Observation after action
            reward: Reward for the action
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional info
        """
        # Store previous state for reward calculation
        prev_x = self.car.x
        
        # Update entities
        self.car.update(action, DT)
        self.pedestrian_manager.update(DT)
        
        # Get pedestrians
        pedestrians = self.pedestrian_manager.get_pedestrians()
        
        # Compute reward
        reward, done = compute_reward(
            self.car, pedestrians, prev_x, action,
            prediction_enabled=self.prediction_enabled, dt=DT
        )
        
        self.episode_reward += reward
        self.step_count += 1
        
        # Check for episode termination
        truncated = self.step_count >= MAX_STEPS
        terminated = done or truncated
        
        # Get observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Render if enabled
        if self.render_mode == "human":
            self.render(info)
        
        return obs, reward, terminated, truncated, info
    
    def render(self, info=None):
        """
        Render the environment.
        
        Args:
            info: Optional info dict to display
        """
        if self.renderer is None:
            from env.renderer import Renderer
            self.renderer = Renderer()
        
        pedestrians = self.pedestrian_manager.get_pedestrians()
        
        self.renderer.render(
            self.car, pedestrians,
            prediction_enabled=self.prediction_enabled,
            episode_info=info,
            step_count=self.step_count,
            episode_reward=self.episode_reward
        )
        
        # Handle user events (window close)
        if self.renderer.handle_events():
            self.close()
    
    def close(self):
        """Close environment and renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
    
    def _get_observation(self):
        """Get observation vector."""
        pedestrians = self.pedestrian_manager.get_pedestrians()
        return get_observation(self.car, pedestrians, self.prediction_enabled)
    
    def _get_info(self):
        """Get info dictionary."""
        pedestrians = self.pedestrian_manager.get_pedestrians()
        
        min_distance = float('inf')
        for ped in pedestrians:
            dist = np.sqrt((self.car.x - ped.x)**2 + (self.car.y - ped.y)**2)
            min_distance = min(min_distance, dist)
        
        return {
            'car_x': self.car.x,
            'car_y': self.car.y,
            'car_velocity': self.car.velocity,
            'min_distance': min_distance,
            'num_pedestrians': len(pedestrians)
        }
    
    def seed(self, seed=None):
        """Set random seed."""
        super().seed(seed=seed)
        np.random.seed(seed)
        return [seed]
