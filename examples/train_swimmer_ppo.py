"""Train Swimmer-v5 with PPO (Stable-Baselines3 / PyTorch).

Usage:
    pip install -r examples/requirements-train.txt
    python examples/train_swimmer_ppo.py --timesteps 2000000

This script uses `SwimmerEnv.get_default_ppo_hyperparams()` for sensible defaults,
vectorized observation/reward normalization, and an optional curriculum wrapper.
"""
import argparse
import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class CurriculumWrapper(gym.Wrapper):
    """Simple curriculum wrapper that increases env curriculum level every N episodes."""

    def __init__(self, env, interval=50, max_level=5):
        super().__init__(env)
        self.episode_count = 0
        self.curriculum_interval = int(interval)
        self.max_level = int(max_level)
        self._first_reset = True

    def reset(self, **kwargs):
        if self.curriculum_interval <= 0:
            return super().reset(**kwargs)

        if self._first_reset:
            self._first_reset = False
        else:
            self.episode_count += 1
            if self.episode_count % self.curriculum_interval == 0:
                try:
                    current = int(self.unwrapped._curriculum_level)
                except Exception:
                    current = 0
                new_level = min(self.max_level, current + 1)
                try:
                    self.unwrapped.set_curriculum_level(new_level)
                    print(f"Curriculum advanced to {new_level} at episode {self.episode_count}")
                except Exception:
                    pass
        return super().reset(**kwargs)


def make_env(xml_file=None, curriculum_interval=50, max_level=5):
    kwargs = {}
    if xml_file is not None:
        kwargs["xml_file"] = xml_file
    env = gym.make("Swimmer-v5", **kwargs)
    env = CurriculumWrapper(env, interval=curriculum_interval, max_level=max_level)
    env = Monitor(env)
    return env


def main(args):
    vec_env = DummyVecEnv(
        [
            lambda: make_env(
                xml_file=args.xml_file,
                curriculum_interval=args.curriculum_interval,
                max_level=args.max_level,
            )
        ]
    )

    # Pull default hyperparams from the environment
    try:
        hp = vec_env.envs[0].unwrapped.get_default_ppo_hyperparams()
    except Exception:
        hp = {
            "learning_rate": 3e-4,
            "clip_range": 0.2,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
        }

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=hp.get("gamma", 0.99),
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=hp.get("learning_rate", 3e-4),
        clip_range=hp.get("clip_range", 0.2),
        n_steps=hp.get("n_steps", 2048),
        batch_size=hp.get("batch_size", 64),
        n_epochs=hp.get("n_epochs", 10),
        gamma=hp.get("gamma", 0.99),
        gae_lambda=hp.get("gae_lambda", 0.95),
        ent_coef=hp.get("ent_coef", 0.01),
        verbose=1,
    )

    model.learn(total_timesteps=int(args.timesteps))

    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, "ppo_swimmer")
    model.save(model_path)
    vec_env.save(f"{model_path}.vecnormalize.pkl")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=2000000, help="Total timesteps to train")
    parser.add_argument("--xml-file", dest="xml_file", default=None, help="Optional xml_file for Swimmer model")
    parser.add_argument("--curriculum-interval", type=int, default=0, help="Episodes between curriculum increases; 0 disables curriculum")
    parser.add_argument("--max-level", type=int, default=5, help="Maximum curriculum level")
    parser.add_argument("--save-dir", default="models", help="Directory to save trained model")
    args = parser.parse_args()
    main(args)
