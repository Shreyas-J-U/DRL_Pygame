# ============================================================================
# Reward Function Module
# ============================================================================
# Defines reward structure for the autonomous driving task.
# ============================================================================

import numpy as np
from config import (
    COLLISION_PENALTY, PROGRESS_REWARD, SAFE_DISTANCE_REWARD,
    BRAKE_PENALTY, PREDICTION_BONUS,
    COLLISION_THRESHOLD, SAFE_DISTANCE
)


def compute_reward(car, pedestrians, prev_x, action, prediction_enabled=True, dt=1/30):
    """
    Compute reward based on environment state and action.
    
    Rewards:
        + Forward progress (PROGRESS_REWARD per pixel moved)
        + Safe distance from pedestrians (SAFE_DISTANCE_REWARD)
        - Collision penalty (COLLISION_PENALTY)
        - Braking penalty (BRAKE_PENALTY)
        + Prediction bonus (PREDICTION_BONUS if using predictions)
    
    Args:
        car: Car object
        pedestrians: List of Pedestrian objects
        prev_x: Previous x position of car (for progress calculation)
        action: Action taken (0-3)
        prediction_enabled: Whether prediction is enabled
        dt: Timestep
        
    Returns:
        reward: Scalar reward value
        done: Boolean indicating episode termination
    """
    reward = 0.0
    done = False
    
    # === COLLISION CHECK ===
    for ped in pedestrians:
        distance = np.sqrt((car.x - ped.x)**2 + (car.y - ped.y)**2)
        
        if distance < COLLISION_THRESHOLD:
            reward += COLLISION_PENALTY
            done = True
            return reward, done
    
    # === FORWARD PROGRESS ===
    # Reward for moving forward along the road
    progress = (car.x - prev_x) * PROGRESS_REWARD
    reward += progress
    
    # === SAFE DISTANCE REWARD ===
    # Reward for maintaining safe distance from pedestrians
    for ped in pedestrians:
        distance = np.sqrt((car.x - ped.x)**2 + (car.y - ped.y)**2)
        if distance > SAFE_DISTANCE:
            reward += SAFE_DISTANCE_REWARD
    
    # === BRAKING PENALTY ===
    # Small penalty to discourage unnecessary braking
    if action == 1:
        reward += BRAKE_PENALTY
    
    # === PREDICTION BONUS ===
    # Small bonus if prediction is enabled (encourages using available info)
    if prediction_enabled:
        reward += PREDICTION_BONUS
    
    return reward, done


def check_collision(car, pedestrians):
    """
    Check if car collides with any pedestrian.
    
    Args:
        car: Car object
        pedestrians: List of Pedestrian objects
        
    Returns:
        collision: Boolean indicating collision
        collided_pedestrian_idx: Index of collided pedestrian (or -1)
    """
    for idx, ped in enumerate(pedestrians):
        distance = np.sqrt((car.x - ped.x)**2 + (car.y - ped.y)**2)
        if distance < COLLISION_THRESHOLD:
            return True, idx
    
    return False, -1


def get_safety_metrics(car, pedestrians):
    """
    Compute safety metrics for logging/debugging.
    
    Args:
        car: Car object
        pedestrians: List of Pedestrian objects
        
    Returns:
        metrics: Dictionary with 'min_distance', 'safe_pedestrians', 'collision'
    """
    if not pedestrians:
        return {'min_distance': float('inf'), 'safe_pedestrians': 0, 'collision': False}
    
    distances = []
    for ped in pedestrians:
        distance = np.sqrt((car.x - ped.x)**2 + (car.y - ped.y)**2)
        distances.append(distance)
    
    min_dist = min(distances)
    safe_count = sum(1 for d in distances if d > SAFE_DISTANCE)
    collision, _ = check_collision(car, pedestrians)
    
    return {
        'min_distance': min_dist,
        'safe_pedestrians': safe_count,
        'collision': collision,
        'total_pedestrians': len(pedestrians)
    }
