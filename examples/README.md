# Examples

This folder contains small example scripts for training and running the `Swimmer-v5` environment.

Files
- `train_swimmer_ppo.py` — training example using Stable-Baselines3 (PPO). Pulls default hyperparameters from the environment and includes a simple curriculum wrapper. Install dependencies with `pip install -r examples/requirements-train.txt`.
- `run_trained.py` — load a saved SB3 model and visualize it in `human` mode or render frames and save a GIF in `rgb_array` mode.
- `requirements-train.txt` — recommended packages for training with SB3 / PyTorch.

Quick run examples

- Dry-run training (1 step):
```
python examples/train_swimmer_ppo.py --timesteps 1 --save-dir models/test_run
```

- Visualize a saved model (human):
```
python examples/run_trained.py --model models/test_run/ppo_swimmer --render human --steps 1000
```

- Record a GIF (rgb frames):
```
python examples/run_trained.py --model models/test_run/ppo_swimmer --render rgb_array --steps 500 --save-gif swimmer_run.gif
```

Notes
- Ensure MuJoCo / OpenGL rendering backend is available if using `human` render mode. On headless servers prefer `rgb_array` + GIF/video recording.
- The training example uses SB3 which relies on PyTorch; please refer to `examples/requirements-train.txt` for versions.
- Swimmer usually needs on the order of millions of timesteps before the gait becomes obvious. The updated training script also saves VecNormalize statistics alongside the model so rollout uses the same observation scale.
