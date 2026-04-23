# ============================================================================
# Training Script: PPO Agent with Trajectory Prediction
# ============================================================================
# Trains an agent using Stable-Baselines3 PPO algorithm on the autonomous
# driving environment with prediction awareness.
# ============================================================================

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from config import (
    LEARNING_RATE, TIMESTEPS_TRAINING, BATCH_SIZE, N_STEPS,
    GAMMA, GAE_LAMBDA
)
from env.car_env import PredictionAwareCarEnv


def setup_directories():
    """Create necessary directories for models and logs."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/training_log", exist_ok=True)


def train_model(prediction_enabled=True, timesteps=TIMESTEPS_TRAINING):
    """
    Train a PPO agent on the autonomous driving task.
    
    Args:
        prediction_enabled: Whether agent has access to predictions
        timesteps: Total timesteps to train
    """
    setup_directories()
    
    # Create environment
    print(f"Creating environment (prediction_enabled={prediction_enabled})...")
    env = PredictionAwareCarEnv(render_mode=None, prediction_enabled=prediction_enabled)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        env, best_model_save_path="logs/training_log",
        log_path="logs/training_log",
        eval_freq=5000, n_eval_episodes=5,
        deterministic=True, render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=5000, save_path="models",
        name_prefix=f"ppo_model_pred{prediction_enabled}"
    )
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        verbose=1,
        tensorboard_log="logs/training_log"
    )
    
    # Train model
    print(f"Training for {timesteps} timesteps...")
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    model_name = f"models/ppo_model_pred_{prediction_enabled}_final"
    model.save(model_name)
    print(f"Model saved to {model_name}")
    
    env.close()
    
    return model_name


def train_both_models():
    """Train two models: one with prediction, one without."""
    print("=" * 80)
    print("Training PPO Agent WITHOUT Trajectory Prediction")
    print("=" * 80)
    model_without = train_model(prediction_enabled=False, timesteps=TIMESTEPS_TRAINING)
    
    print("\n" + "=" * 80)
    print("Training PPO Agent WITH Trajectory Prediction")
    print("=" * 80)
    model_with = train_model(prediction_enabled=True, timesteps=TIMESTEPS_TRAINING)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Model without prediction: {model_without}")
    print(f"Model with prediction: {model_with}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO agent for autonomous driving")
    parser.add_argument("--prediction", type=bool, default=True,
                       help="Whether to include trajectory predictions (default: True)")
    parser.add_argument("--timesteps", type=int, default=TIMESTEPS_TRAINING,
                       help=f"Number of timesteps to train (default: {TIMESTEPS_TRAINING})")
    parser.add_argument("--train-both", action="store_true",
                       help="Train both models (with and without prediction)")
    
    args = parser.parse_args()
    
    if args.train_both:
        train_both_models()
    else:
        train_model(prediction_enabled=args.prediction, timesteps=args.timesteps)
