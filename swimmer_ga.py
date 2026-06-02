from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass(slots=True)
class GAConfig:
    env_id: str = "Swimmer-v5"
    population_size: int = 32
    generations: int = 30
    elite_size: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    mutation_scale: float = 0.15
    hidden_sizes: tuple[int, int] = (16, 16)
    episodes_per_candidate: int = 1
    base_episode_length: int = 200
    curriculum_growth: int = 25
    render_best: bool = False
    seed: int | None = None
    output_dir: str = "artifacts"


class GenomeMLPController(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, int] = (16, 16)):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes

        layers: list[nn.Module] = []
        layer_sizes = [obs_dim, *hidden_sizes, action_dim]
        for index, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if index < len(layer_sizes) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

        self.genome_size = sum(parameter.numel() for parameter in self.parameters())

    def random_genome(self, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, 0.5, size=self.genome_size).astype(np.float32)

    def load_genome(self, genome: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters():
                size = parameter.numel()
                values = torch.as_tensor(genome[offset : offset + size], dtype=parameter.dtype).reshape(
                    parameter.shape
                )
                parameter.copy_(values)
                offset += size

    def act(self, genome: np.ndarray, observation: np.ndarray) -> np.ndarray:
        self.load_genome(genome)
        observation_tensor = torch.as_tensor(observation, dtype=torch.float32)
        with torch.no_grad():
            action_tensor = torch.tanh(self.network(observation_tensor))
        return action_tensor.cpu().numpy().astype(np.float32)


@dataclass(slots=True)
class EvaluationResult:
    fitness: float
    forward_reward: float
    energy_penalty: float
    instability_penalty: float
    oscillation_penalty: float
    episode_length: int


class SwimmerEvaluator:
    def __init__(self, env_id: str, controller: GenomeMLPController, seed: int | None = None):
        self.env_id = env_id
        self.controller = controller
        self.seed = seed

    def _make_env(self, render_mode: str | None = None):
        return gym.make(self.env_id, render_mode=render_mode)

    def evaluate(
        self,
        genome: np.ndarray,
        episode_length: int,
        episodes: int = 1,
        render: bool = False,
    ) -> EvaluationResult:
        total_fitness = 0.0
        total_forward = 0.0
        total_energy = 0.0
        total_instability = 0.0
        total_oscillation = 0.0
        total_steps = 0

        env = self._make_env(render_mode="human" if render else None)
        try:
            for episode_index in range(episodes):
                observation, _ = env.reset(seed=None if self.seed is None else self.seed + episode_index)
                previous_action = np.zeros(env.action_space.shape, dtype=np.float32)
                previous_velocity = np.zeros(2, dtype=np.float32)

                episode_fitness = 0.0
                episode_forward = 0.0
                episode_energy = 0.0
                episode_instability = 0.0
                episode_oscillation = 0.0

                for _ in range(episode_length):
                    action = self.controller.act(genome, observation)
                    action = np.clip(action, env.action_space.low, env.action_space.high)

                    next_observation, _, terminated, truncated, info = env.step(action)

                    forward_reward = float(info.get("reward_forward", 0.0))
                    control_penalty = float(-info.get("reward_ctrl", 0.0))
                    action_delta = float(np.mean(np.abs(action - previous_action)))
                    velocity_delta = float(abs(info.get("x_velocity", 0.0)) + abs(info.get("y_velocity", 0.0)) - np.sum(np.abs(previous_velocity)))
                    velocity_delta = abs(velocity_delta)

                    episode_forward += forward_reward
                    episode_energy += control_penalty
                    episode_instability += velocity_delta
                    episode_oscillation += action_delta

                    episode_fitness += forward_reward
                    episode_fitness -= 0.25 * control_penalty
                    episode_fitness -= 0.05 * velocity_delta
                    episode_fitness -= 0.02 * action_delta

                    previous_action = action
                    previous_velocity = np.array([info.get("x_velocity", 0.0), info.get("y_velocity", 0.0)], dtype=np.float32)
                    observation = next_observation
                    total_steps += 1

                    if render:
                        env.render()
                    if terminated or truncated:
                        break

                total_fitness += episode_fitness
                total_forward += episode_forward
                total_energy += episode_energy
                total_instability += episode_instability
                total_oscillation += episode_oscillation
        finally:
            env.close()

        scale = 1.0 / max(episodes, 1)
        return EvaluationResult(
            fitness=total_fitness * scale,
            forward_reward=total_forward * scale,
            energy_penalty=total_energy * scale,
            instability_penalty=total_instability * scale,
            oscillation_penalty=total_oscillation * scale,
            episode_length=total_steps // max(episodes, 1),
        )


class GeneticAlgorithmTrainer:
    def __init__(self, config: GAConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        probe_env = gym.make(config.env_id)
        try:
            obs_dim = int(np.prod(probe_env.observation_space.shape))
            action_dim = int(np.prod(probe_env.action_space.shape))
        finally:
            probe_env.close()
        self.controller = GenomeMLPController(obs_dim, action_dim, config.hidden_sizes)
        self.evaluator = SwimmerEvaluator(config.env_id, self.controller, seed=config.seed)
        self.population = [self.controller.random_genome(self.rng) for _ in range(config.population_size)]
        self.history = {
            "best_fitness": [],
            "average_fitness": [],
            "forward_reward": [],
            "energy_penalty": [],
            "instability_penalty": [],
            "oscillation_penalty": [],
        }

    def _episode_length_for_generation(self, generation: int) -> int:
        return self.config.base_episode_length + generation * self.config.curriculum_growth

    def _tournament_select(self, fitnesses: np.ndarray) -> int:
        contenders = self.rng.choice(len(fitnesses), size=self.config.tournament_size, replace=False)
        return int(contenders[np.argmax(fitnesses[contenders])])

    def _crossover(self, parent_a: np.ndarray, parent_b: np.ndarray) -> np.ndarray:
        if self.rng.random() > self.config.crossover_rate:
            return parent_a.copy()
        mask = self.rng.random(parent_a.shape) < 0.5
        child = np.where(mask, parent_a, parent_b)
        return child.astype(np.float32)

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        mutated = genome.copy()
        mutation_mask = self.rng.random(mutated.shape) < self.config.mutation_rate
        gaussian_noise = self.rng.normal(0.0, self.config.mutation_scale, size=mutated.shape)
        mutated[mutation_mask] += gaussian_noise[mutation_mask]
        return mutated.astype(np.float32)

    def evaluate_population(self, generation: int) -> tuple[np.ndarray, list[EvaluationResult]]:
        episode_length = self._episode_length_for_generation(generation)
        scores: list[float] = []
        results: list[EvaluationResult] = []

        for index, genome in enumerate(self.population):
            render = self.config.render_best and generation == self.config.generations - 1 and index == 0
            result = self.evaluator.evaluate(
                genome,
                episode_length=episode_length,
                episodes=self.config.episodes_per_candidate,
                render=render,
            )
            scores.append(result.fitness)
            results.append(result)

        return np.asarray(scores, dtype=np.float32), results

    def next_generation(self, fitnesses: np.ndarray) -> None:
        elite_count = min(self.config.elite_size, len(self.population))
        elite_indices = np.argsort(fitnesses)[-elite_count:][::-1]
        elites = [self.population[index].copy() for index in elite_indices]

        new_population = elites.copy()
        while len(new_population) < self.config.population_size:
            parent_a = self.population[self._tournament_select(fitnesses)]
            parent_b = self.population[self._tournament_select(fitnesses)]
            child = self._crossover(parent_a, parent_b)
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population[: self.config.population_size]

    def train(self) -> dict[str, list[float]]:
        best_genome = None
        best_fitness = -np.inf
        best_result: EvaluationResult | None = None

        for generation in range(self.config.generations):
            fitnesses, results = self.evaluate_population(generation)
            best_index = int(np.argmax(fitnesses))
            generation_best = float(fitnesses[best_index])
            generation_mean = float(np.mean(fitnesses))
            generation_result = results[best_index]

            self.history["best_fitness"].append(generation_best)
            self.history["average_fitness"].append(generation_mean)
            self.history["forward_reward"].append(generation_result.forward_reward)
            self.history["energy_penalty"].append(generation_result.energy_penalty)
            self.history["instability_penalty"].append(generation_result.instability_penalty)
            self.history["oscillation_penalty"].append(generation_result.oscillation_penalty)

            if generation_best > best_fitness:
                best_fitness = generation_best
                best_genome = self.population[best_index].copy()
                best_result = generation_result

            print(
                f"generation {generation:03d} | best={generation_best:.3f} | avg={generation_mean:.3f} | "
                f"forward={generation_result.forward_reward:.3f} | energy={generation_result.energy_penalty:.3f}"
            )

            if generation < self.config.generations - 1:
                self.next_generation(fitnesses)

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "best_genome.npy", best_genome)
        self._save_history_plot(output_dir / "fitness_history.png")

        if best_result is not None:
            print(
                "best genome summary: "
                f"fitness={best_fitness:.3f}, forward={best_result.forward_reward:.3f}, "
                f"energy={best_result.energy_penalty:.3f}, instability={best_result.instability_penalty:.3f}"
            )

        return self.history

    def _save_history_plot(self, plot_path: Path) -> None:
        generations = np.arange(1, len(self.history["best_fitness"]) + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(generations, self.history["best_fitness"], label="best fitness")
        plt.plot(generations, self.history["average_fitness"], label="average fitness")
        plt.plot(generations, self.history["forward_reward"], label="forward reward")
        plt.plot(generations, self.history["energy_penalty"], label="energy penalty")
        plt.xlabel("Generation")
        plt.ylabel("Score")
        plt.title("Swimmer-v5 genetic algorithm training")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

    def run_best_policy(self, genome_path: str | Path, steps: int = 1000) -> None:
        genome = np.load(genome_path)
        self.evaluator.evaluate(genome, episode_length=steps, episodes=1, render=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Swimmer-v5 controller with a genetic algorithm.")
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population-size", type=int, default=32)
    parser.add_argument("--elite-size", type=int, default=2)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-scale", type=float, default=0.15)
    parser.add_argument("--crossover-rate", type=float, default=0.9)
    parser.add_argument("--base-episode-length", type=int, default=200)
    parser.add_argument("--curriculum-growth", type=int, default=25)
    parser.add_argument("--episodes-per-candidate", type=int, default=1)
    parser.add_argument("--render-best", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--play-best", type=str, default="")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    config = GAConfig(
        population_size=args.population_size,
        generations=args.generations,
        elite_size=args.elite_size,
        tournament_size=args.tournament_size,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        mutation_scale=args.mutation_scale,
        episodes_per_candidate=args.episodes_per_candidate,
        base_episode_length=args.base_episode_length,
        curriculum_growth=args.curriculum_growth,
        render_best=args.render_best,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    trainer = GeneticAlgorithmTrainer(config)

    if args.play_best:
        trainer.run_best_policy(args.play_best)
        return

    trainer.train()


if __name__ == "__main__":
    main()
