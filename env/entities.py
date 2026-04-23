# ============================================================================
# Environment Entities: Car and Pedestrian
# ============================================================================

import numpy as np
from config import (
    CAR_INITIAL_X, CAR_INITIAL_Y,
    CAR_MAX_VELOCITY, CAR_ACCELERATION, CAR_BRAKE_DECELERATION,
    CAR_LANE_CHANGE_SPEED,
    LANE_Y, ROAD_HEIGHT, SCREEN_WIDTH, SCREEN_HEIGHT,
    PED_SPEED_MIN, PED_SPEED_MAX, PED_CROSS_SPEED_DEVIATION,
    NUM_PEDESTRIANS
)


class Car:
    """Represents the autonomous vehicle (ego agent)."""
    
    def __init__(self):
        self.x = CAR_INITIAL_X
        self.y = CAR_INITIAL_Y
        self.velocity = 0  # pixels per frame
        self.width = 40
        self.height = 20
        
    def update(self, action, dt):
        """
        Update car state based on action.
        
        Actions:
            0: accelerate
            1: brake
            2: move up (change lane)
            3: move down (change lane)
        """
        if action == 0:  # Accelerate
            self.velocity += CAR_ACCELERATION * dt
        elif action == 1:  # Brake
            self.velocity -= CAR_BRAKE_DECELERATION * dt
        elif action == 2:  # Move up
            self.y -= CAR_LANE_CHANGE_SPEED * dt
        elif action == 3:  # Move down
            self.y += CAR_LANE_CHANGE_SPEED * dt
        
        # Apply constraints
        self.velocity = np.clip(self.velocity, 0, CAR_MAX_VELOCITY)
        
        # Keep car on road (vertical bounds)
        self.y = np.clip(self.y, LANE_Y - ROAD_HEIGHT/2, LANE_Y + ROAD_HEIGHT/2)
        
        # Update position based on velocity
        self.x += self.velocity * dt
    
    def get_center(self):
        """Return center position of car."""
        return (self.x + self.width/2, self.y + self.height/2)
    
    def reset(self):
        """Reset car to initial state."""
        self.x = CAR_INITIAL_X
        self.y = CAR_INITIAL_Y
        self.velocity = 0


class Pedestrian:
    """Represents a pedestrian crossing the road."""
    
    def __init__(self, spawn_x):
        """
        Initialize pedestrian.
        
        Args:
            spawn_x: Horizontal spawn position
        """
        self.x = spawn_x
        self.y = -50  # Spawn above the road
        self.radius = 10
        
        # Velocity (pixels per frame)
        # vx: horizontal velocity (typically zero for crossing pedestrians)
        # vy: vertical velocity (crossing the road)
        self.vx = np.random.uniform(-10, 10)  # Slight horizontal drift
        self.vy = np.random.uniform(PED_SPEED_MIN, PED_SPEED_MAX)  # Crossing speed
        
        # Add some randomness for different behavior
        self._speed_variation = np.random.uniform(1 - PED_CROSS_SPEED_DEVIATION, 
                                                   1 + PED_CROSS_SPEED_DEVIATION)
        self.vy *= self._speed_variation
    
    def update(self, dt):
        """Update pedestrian position based on velocity."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Constrain horizontal movement to screen
        self.x = np.clip(self.x, 0, SCREEN_WIDTH)
    
    def is_off_screen(self):
        """Check if pedestrian has left the screen."""
        return self.y > SCREEN_HEIGHT + 100
    
    def reset(self, spawn_x):
        """Reset pedestrian to initial state."""
        self.x = spawn_x
        self.y = -50
        self.vx = np.random.uniform(-10, 10)
        self.vy = np.random.uniform(PED_SPEED_MIN, PED_SPEED_MAX)
        self._speed_variation = np.random.uniform(1 - PED_CROSS_SPEED_DEVIATION,
                                                   1 + PED_CROSS_SPEED_DEVIATION)
        self.vy *= self._speed_variation


class PedestrianManager:
    """Manages pool of pedestrians for the simulation."""
    
    def __init__(self, num_pedestrians=NUM_PEDESTRIANS):
        """Initialize pedestrian manager."""
        self.num_pedestrians = num_pedestrians
        self.pedestrians = []
        self._spawn_queue = []
        self._init_pedestrians()
    
    def _init_pedestrians(self):
        """Create initial set of pedestrians."""
        from config import PED_SPAWN_X_MIN, PED_SPAWN_X_MAX
        
        for _ in range(self.num_pedestrians):
            spawn_x = np.random.randint(PED_SPAWN_X_MIN, PED_SPAWN_X_MAX)
            ped = Pedestrian(spawn_x)
            self.pedestrians.append(ped)
    
    def update(self, dt):
        """Update all pedestrians and manage respawning."""
        from config import PED_SPAWN_X_MIN, PED_SPAWN_X_MAX
        
        for ped in self.pedestrians:
            ped.update(dt)
            
            # Respawn if off-screen
            if ped.is_off_screen():
                spawn_x = np.random.randint(PED_SPAWN_X_MIN, PED_SPAWN_X_MAX)
                ped.reset(spawn_x)
    
    def reset(self):
        """Reset all pedestrians."""
        from config import PED_SPAWN_X_MIN, PED_SPAWN_X_MAX
        
        for ped in self.pedestrians:
            spawn_x = np.random.randint(PED_SPAWN_X_MIN, PED_SPAWN_X_MAX)
            ped.reset(spawn_x)
    
    def get_pedestrians(self):
        """Return list of active pedestrians."""
        return self.pedestrians
