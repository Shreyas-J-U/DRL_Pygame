# ============================================================================
# State Representation Module
# ============================================================================
# Converts environment state into observation suitable for neural network.
# Includes current positions and predicted future trajectories.
# ============================================================================

import numpy as np
from config import (
    OBSERVATION_DIM, CAR_STATE_DIM, PED_CURRENT_STATE_DIM,
    PREDICTION_STEPS, NUM_PEDESTRIANS, SCREEN_WIDTH, SCREEN_HEIGHT,
    CAR_MAX_VELOCITY
)
from utils.prediction import get_flattened_predictions


def normalize_value(value, min_val, max_val):
    """Normalize value to [-1, 1] range."""
    if max_val == min_val:
        return 0.0
    return 2 * (value - min_val) / (max_val - min_val) - 1


def get_observation(car, pedestrians, prediction_enabled=True):
    """
    Generate observation vector from environment state.
    
    Observation structure:
    [
        car_x_norm, car_y_norm, car_velocity_norm,  # 3 features
        ped0_x, ped0_y, ped1_x, ped1_y, ...,        # 2*N current positions
        pred_ped0_x_steps, ..., pred_pedN_y_steps    # 2*N*steps predicted positions
    ]
    
    All values are normalized to [-1, 1] range.
    
    Args:
        car: Car object
        pedestrians: List of Pedestrian objects
        prediction_enabled: Whether to include predicted positions
        
    Returns:
        obs: Normalized numpy array of shape (OBSERVATION_DIM,)
    """
    obs = []
    
    # === CAR STATE ===
    # Normalize car position and velocity
    car_x_norm = normalize_value(car.x, 0, SCREEN_WIDTH)
    car_y_norm = normalize_value(car.y, 0, SCREEN_HEIGHT)
    car_v_norm = normalize_value(car.velocity, 0, CAR_MAX_VELOCITY)
    
    obs.extend([car_x_norm, car_y_norm, car_v_norm])
    
    # === PEDESTRIAN CURRENT POSITIONS ===
    for ped in pedestrians:
        ped_x_norm = normalize_value(ped.x, 0, SCREEN_WIDTH)
        ped_y_norm = normalize_value(ped.y, 0, SCREEN_HEIGHT)
        obs.extend([ped_x_norm, ped_y_norm])
    
    # === PREDICTED PEDESTRIAN POSITIONS ===
    if prediction_enabled:
        predictions = get_flattened_predictions(pedestrians, PREDICTION_STEPS)
        # Normalize predictions
        for pred_val in predictions:
            # Predictions could be outside screen bounds, so use wider range
            pred_norm = normalize_value(pred_val, -SCREEN_WIDTH/2, SCREEN_WIDTH*1.5)
            obs.append(pred_norm)
    else:
        # If prediction disabled, fill with zeros
        obs.extend([0.0] * (OBSERVATION_DIM - CAR_STATE_DIM - PED_CURRENT_STATE_DIM))
    
    # Convert to numpy array and ensure correct shape
    obs = np.array(obs, dtype=np.float32)
    
    # Pad or trim to exact dimension
    if len(obs) < OBSERVATION_DIM:
        obs = np.pad(obs, (0, OBSERVATION_DIM - len(obs)), 'constant')
    elif len(obs) > OBSERVATION_DIM:
        obs = obs[:OBSERVATION_DIM]
    
    return obs


def get_state_dict(car, pedestrians, prediction_enabled=True):
    """
    Get structured state dictionary (for debugging/logging).
    
    Returns:
        state: Dictionary with keys 'car', 'pedestrians', 'predictions'
    """
    from utils.prediction import predict_all_trajectories
    
    state = {
        'car': {
            'x': car.x,
            'y': car.y,
            'velocity': car.velocity
        },
        'pedestrians': [
            {'x': ped.x, 'y': ped.y, 'vx': ped.vx, 'vy': ped.vy}
            for ped in pedestrians
        ]
    }
    
    if prediction_enabled:
        state['predictions'] = predict_all_trajectories(pedestrians, PREDICTION_STEPS)
    
    return state
