# jym

jym: A platform to test AI models in various environments.

Currently, we have the following environments onboarded:

**Flight Simulator environment** with physics implemented by [JSBSim](https://github.com/JSBSim-Team/jsbsim), flight dynamics software that has been [approved by NASA](https://nescacademy.nasa.gov/flightsim/2015)
    - Supports flying a Cessna 172 aircraft (its configuration like weight, wingspan, speed is defined in JSBSim repo)
    - Supports flying a drone quadcopter (config also defined in JSBSim repo)

Objectives in this environment include take-off and landing, flying a full circuit, and flying through rings (both easy and hard variants).

We have the following models onboarded:

- Jev, a closed source System 1 model from TypeSafe AI (need to set JEV_API_KEY in .env file)
- Laya, an open source System 1 model (need to run `uv sync --extra laya` first)
- Base Qwen3 0.6B 8bit model
- Our custom flight simulator model

## Run it

Needs [uv](https://docs.astral.sh/uv/), Node.js 22 and Docker (models run on llama.cpp in Docker, both CPU and GPU-passthrough supported).

```sh
uv run jym ui
```

Open http://127.0.0.1:8765.

There are two modes you can toggle at the top, "Watch" and "Play":
- "Watch" shows a model acting in the environment for that objective (configure via [models.toml](models.toml))
- "Play" lets you play in the environment yourself

## Plug-in support

All models and environments are defined as plug-ins.

So you can add an environment by declaring an `EnvironmentSpec` ([core.py](src/jym/core.py))
under the entry-point group `jym.environments`, like how the flight simulator is declared in [pyproject.toml](pyproject.toml).
