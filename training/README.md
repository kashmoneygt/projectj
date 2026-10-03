# Model Training

This folder contains model training scripts.

Currently, we have the following model training scripts onboarded:

**Flight Controller model** which is a [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (Apache 2.0) with a LoRA adapter. It is trained using imitation to get a basic understanding of flight and then PPO to be able to generalize flight well for multiple aircraft. Its state is text-based, not pixels-based.

```sh
uv sync

# imitation - behavior cloning on 128 autopilot flights
uv run train.py dagger

# RL (PPO) with 20 iterations of 2,048 sampled decisions. reward function is based on progress
# towards the objective and time passed (making it agnostic to model decision latency)
uv run train.py ppo

# 20 more iterations of PPO with random airfields, aircraft parameters, wind gusts, noisy readings, etc.
uv run train.py robust

# run model on both aircraft (cessna and drone) in held-out seeds (not seen in training)
uv run train.py evaluate --adapter output/robust/final --out output/robust/eval.json

# writes to models/flight-controller-Q8_0.gguf
uv run train.py export --adapter output/robust/final
```

To see/edit the reward function, take a look at [reward.py](./flight_controller/reward.py) and how it's used in [ppo.py](./flight_controller/ppo.py).

To fly the trained model in jym:

```sh
cd ../jym && uv run jym ui --players ../training/models/flight-controller-Q8_0.gguf,reference
```

For help at any step or to understand parameters, run `uv run train.py <stage> --help`.
