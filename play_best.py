import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

from swimmer_ga import GenomeMLPController


def parse_hidden_sizes(raw: str):
    if not raw:
        return (16, 16)
    return tuple(int(p.strip()) for p in raw.split(",") if p.strip())


def load_genome(path: Path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "genome" in data:
            return np.asarray(data["genome"])
        # try first array
        for key in data.files:
            return np.asarray(data[key])
    return np.asarray(data)


def main():
    parser = argparse.ArgumentParser(description="Play back a saved Swimmer genome.")
    parser.add_argument("--genome", type=str, default="artifacts/best_genome.npy")
    parser.add_argument("--env", type=str, default="Swimmer-v5")
    parser.add_argument("--hidden-sizes", type=str, default="16,16")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--render", action="store_true", default=True)
    parser.add_argument("--no-render", action="store_false", dest="render")
    args = parser.parse_args()

    genome_path = Path(args.genome)
    if not genome_path.exists():
        raise SystemExit(f"Genome file not found: {genome_path}")

    genome = load_genome(genome_path)

    env = gym.make(args.env, render_mode="human" if args.render else None)
    try:
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))

        hidden = parse_hidden_sizes(args.hidden_sizes)
        controller = GenomeMLPController(obs_dim, action_dim, hidden)

        if genome.size != controller.genome_size:
            raise SystemExit(
                f"Genome size ({genome.size}) does not match controller ({controller.genome_size}). "
                "Pass matching --hidden-sizes or regenerate with the same architecture."
            )

        controller.load_genome(genome)

        obs, _ = env.reset()
        for step in range(args.steps):
            action = controller.act(genome, obs)
            obs, _, terminated, truncated, info = env.step(action)
            if args.render:
                env.render()
            if terminated or truncated:
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()
