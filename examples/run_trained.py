"""Run and visualize a trained Swimmer-v5 policy.

Usage examples:
  python examples/run_trained.py --model models/test_run/ppo_swimmer --render human --steps 1000
  python examples/run_trained.py --model models/test_run/ppo_swimmer --render rgb_array --steps 500 --save-gif swimmer_run.gif

Ensure you have MuJoCo and required Python packages installed (see examples/requirements-train.txt).
"""
import argparse
import time
import os

import gymnasium as gym
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SB3 model (without .zip or with)")
    parser.add_argument("--render", choices=["human", "rgb_array"], default="human")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-gif", dest="save_gif", default=None, help="Path to save gif when using rgb_array")
    parser.add_argument("--curriculum-level", type=int, default=None, help="Optional curriculum level to set on env")
    parser.add_argument("--window-width", type=int, default=2500, help="Rendering window width in pixels")
    parser.add_argument("--window-height", type=int, default=1200, help="Rendering window height in pixels")
    args = parser.parse_args()

    model = PPO.load(args.model)

    # Configure larger default camera for better visibility
    default_camera_config = {
        "distance": 4.0,
        "azimuth": 90.0,
        "elevation": 0.0,
    }

    env = gym.make(
        "Swimmer-v5",
        render_mode=("rgb_array" if args.render == "rgb_array" else "human"),
        default_camera_config=default_camera_config,
    )
    # set curriculum level if requested
    if args.curriculum_level is not None:
        try:
            env.unwrapped.set_curriculum_level(int(args.curriculum_level))
            print(f"Set curriculum level to {args.curriculum_level}")
        except Exception:
            pass

    obs, info = env.reset()

    frames = []
    for i in range(int(args.steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        if args.render == "human":
            # env renders automatically in human mode; sleep a bit for visibility
            time.sleep(0.01)
        else:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset()

    if args.render == "human":
        print("\nRunning finished. Press Enter to close the window...")
        try:
            input()
        except Exception:
            pass

    env.close()


if __name__ == "__main__":
    main()
