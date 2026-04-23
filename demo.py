# ============================================================================
# Demo Script: Run Trained PPO Agent
# ============================================================================
# Loads a trained PPO model and runs it in the autonomous driving environment
# with rendering enabled to visualize the agent's behavior.
# ============================================================================

import time
import argparse
from stable_baselines3 import PPO

from config import MAX_SIMULATION_TIME
from env.car_env import PredictionAwareCarEnv


def run_demo(model_path, episodes=3, max_time=MAX_SIMULATION_TIME, render=True):
    """
    Run the trained model in the environment with rendering.

    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to run
        max_time: Maximum time per episode in seconds
        render: Whether to render the environment
    """
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create environment with rendering
    render_mode = "human" if render else None
    env = PredictionAwareCarEnv(render_mode=render_mode, prediction_enabled=True)

    print(f"Running {episodes} demo episodes...")
    print("Controls: Close the window to stop the demo")

    for episode in range(episodes):
        print(f"\n--- Episode {episode + 1}/{episodes} ---")

        obs, info = env.reset()
        episode_reward = 0
        step = 0
        start_time = time.time()

        while True:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            # Check time limit
            if time.time() - start_time > max_time:
                print(f"Episode {episode + 1} timed out after {max_time} seconds")
                break

            if terminated or truncated:
                print(f"Episode {episode + 1} ended after {step} steps")
                print(".2f")
                break

            # Small delay for visualization
            if render:
                time.sleep(0.02)  # ~50 FPS

    env.close()
    print("\nDemo complete!")


def run_evaluation(model_path, num_episodes=10):
    """
    Evaluate the trained model without rendering.

    Args:
        model_path: Path to the saved model
        num_episodes: Number of evaluation episodes
    """
    print(f"Evaluating model from {model_path}...")

    # Load the trained model
    model = PPO.load(model_path)

    # Create environment without rendering
    env = PredictionAwareCarEnv(render_mode=None, prediction_enabled=True)

    rewards = []
    episode_lengths = []
    collisions = 0

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        step = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step += 1

            if terminated or truncated:
                rewards.append(episode_reward)
                episode_lengths.append(step)
                if info.get('collision', False):
                    collisions += 1
                break

    env.close()

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {sum(rewards)/len(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Episode Length: {sum(episode_lengths)/len(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Collisions: {collisions}/{num_episodes} ({collisions/num_episodes*100:.1f}%)")

    return rewards, episode_lengths, collisions


if __name__ == "__main__":
    import numpy as np

    parser = argparse.ArgumentParser(description="Run trained PPO agent demo")
    parser.add_argument("--model", type=str, default="models/ppo_model_pred_True_final.zip",
                       help="Path to trained model (default: final model)")
    parser.add_argument("--episodes", type=int, default=3,
                       help="Number of demo episodes to run")
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of evaluation episodes (if evaluating)")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation instead of demo")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for demo")

    args = parser.parse_args()

    if args.evaluate:
        run_evaluation(args.model, args.eval_episodes)
    else:
        run_demo(args.model, args.episodes, render=not args.no_render)