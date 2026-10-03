import os

import typer

from flight_controller import evaluation, export, imitation, ppo

# avoids GPU memory fragmentation from varying prompt lengths
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Train a Qwen3 0.6B flight controller for jym.")
app.command(help="Imitate the scripted pilot with behaviour cloning, then DAgger.")(imitation.dagger)
app.command(help="Fine-tune dagger's adapter with PPO, helped by the scripted pilot.")(ppo.ppo)
app.command(help="Run the last stage, PPO in harder conditions, from PPO's adapter.")(ppo.robust)
app.command(help="Fly the evaluation protocol on held-out seeds.")(evaluation.evaluate)
app.command(help="Merge an adapter into Qwen3 0.6B and write the Q8_0 GGUF.")(export.export)

if __name__ == "__main__":
    app()
