# Gymnasium Swimmer Genetic Algorithm

This workspace now contains a first-pass genetic algorithm trainer for `Swimmer-v5`.

## Files

- `swimmer_ga.py`: PyTorch MLP controller encoded as a genome, fitness evaluation, selection, crossover, mutation, elitism, curriculum, and plotting.
- `swimmer_test.py`: simple environment smoke test.

## Install

```bash
pip install -r requirements.txt
```

## Run a small training test

```bash
python swimmer_ga.py --generations 10 --population-size 16
```

## View the best policy

```bash
python swimmer_ga.py --generations 10 --population-size 16 --render-best
```

The trainer writes artifacts to `artifacts/` by default, including the best genome and fitness plot.
