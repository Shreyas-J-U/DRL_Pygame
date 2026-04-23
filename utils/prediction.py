# ============================================================================
# Trajectory Prediction Module
# ============================================================================
# This module predicts future positions of pedestrians using linear extrapolation.
# In a real system, this would use Trajectron++ or similar trajectory models.
# ============================================================================

import numpy as np
from config import PREDICTION_STEPS, DT


def predict_trajectory(pedestrian, num_steps=PREDICTION_STEPS):
    """
    Predict future trajectory of a pedestrian using linear motion model.
    
    Args:
        pedestrian: Pedestrian object with x, y, vx, vy attributes
        num_steps: Number of future timesteps to predict
        
    Returns:
        trajectory: List of (x, y) tuples representing predicted positions
                   Length: num_steps
    """
    trajectory = []
    
    # Current state
    current_x = pedestrian.x
    current_y = pedestrian.y
    current_vx = pedestrian.vx
    current_vy = pedestrian.vy
    
    # Linear extrapolation: new_pos = current_pos + velocity * time
    for step in range(1, num_steps + 1):
        time_delta = step * DT
        
        pred_x = current_x + current_vx * time_delta
        pred_y = current_y + current_vy * time_delta
        
        trajectory.append((pred_x, pred_y))
    
    return trajectory


def predict_all_trajectories(pedestrians, num_steps=PREDICTION_STEPS):
    """
    Predict trajectories for all pedestrians.
    
    Args:
        pedestrians: List of Pedestrian objects
        num_steps: Number of future timesteps
        
    Returns:
        trajectories: Dict mapping pedestrian_index -> trajectory list
    """
    trajectories = {}
    for idx, ped in enumerate(pedestrians):
        trajectories[idx] = predict_trajectory(ped, num_steps)
    
    return trajectories


def get_flattened_predictions(pedestrians, num_steps=PREDICTION_STEPS):
    """
    Get all predictions flattened into a single array for neural network input.
    
    Args:
        pedestrians: List of Pedestrian objects
        num_steps: Number of prediction steps
        
    Returns:
        predictions: Flattened numpy array of shape (num_pedestrians * num_steps * 2,)
    """
    all_predictions = []
    
    for ped in pedestrians:
        traj = predict_trajectory(ped, num_steps)
        # Flatten trajectory: [x1, y1, x2, y2, ..., xN, yN]
        for pred_pos in traj:
            all_predictions.extend(pred_pos)
    
    return np.array(all_predictions, dtype=np.float32)
