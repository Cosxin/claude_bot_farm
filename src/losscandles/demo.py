"""Generate a synthetic training run and write it as real TensorBoard event files.

Run with `python -m losscandles.demo`, then `tensorboard --logdir runs/` and open
the Candles tab. Torch-free: uses `losscandles.torch.LossCandlesWriter` when torch
is installed (to also demonstrate epoch-boundary enrichment), otherwise falls back
to `tensorboard.summary.Writer` directly.

N_STEPS is kept under TensorBoard's default reservoir-sampling cap (1000 points
per tag; see the `_DOWNSAMPLE_CAP` note in plugin.py) so the demo shows full,
undistorted candles with zero extra flags. Longer real runs will want
`tensorboard --logdir runs --samples_per_plugin scalars=100000` (see README).
"""

from __future__ import annotations

import argparse
import math
import os
import random

DEFAULT_LOGDIR = "runs/demo"
N_STEPS = 900
STEPS_PER_EPOCH = 90
SPIKE_LEN = 45  # a full default-window's worth, so the spike dominates a candle's close


def generate(seed: int = 0):
    """Yield (step, {tag: value}) for one synthetic training run."""
    rng = random.Random(seed)
    nan_step = int(N_STEPS * 0.55)
    spike_start = int(N_STEPS * 0.35)
    lr_drop_step = int(N_STEPS * 0.5)
    overfit_start = int(N_STEPS * 0.8)

    lr = 0.1
    for step in range(N_STEPS):
        if step == lr_drop_step:
            lr *= 0.1

        train_loss = 2.5 * math.exp(-step / 600) + 0.05 + rng.gauss(0, 0.008)
        train_loss = max(train_loss, 1e-4)
        if spike_start <= step < spike_start + SPIKE_LEN:
            train_loss *= rng.uniform(2.2, 3.0)

        val_loss = train_loss * 1.15 + rng.gauss(0, 0.004)
        if step > overfit_start:
            val_loss += (step - overfit_start) * 0.007

        grad_norm = abs(rng.gauss(1.0, 0.3)) + 0.5

        yield step, {
            "train/loss": float("nan") if step == nan_step else train_loss,
            "val/loss": max(val_loss, 1e-4),
            "lr": lr,
            "grad_norm": grad_norm,
            "epoch": step // STEPS_PER_EPOCH,
        }


def write_demo_run(logdir: str = DEFAULT_LOGDIR, seed: int = 0) -> str:
    os.makedirs(logdir, exist_ok=True)
    writer = _make_writer(logdir)
    try:
        current_epoch = None
        for step, values in generate(seed):
            epoch = values["epoch"]
            if epoch != current_epoch:
                current_epoch = epoch
                if hasattr(writer, "add_epoch_boundary"):
                    writer.add_epoch_boundary(step)
            for tag, value in values.items():
                writer.add_scalar(tag, value, step)
        if hasattr(writer, "set_volume_tag"):
            writer.set_volume_tag("grad_norm")
    finally:
        writer.close()
    return logdir


def _make_writer(logdir: str):
    try:
        from losscandles.torch import LossCandlesWriter

        return LossCandlesWriter(logdir)
    except ImportError:
        return _NativeWriterAdapter(logdir)


class _NativeWriterAdapter:
    """Adapts `tensorboard.summary.Writer` to the writer API `write_demo_run` needs,
    for environments without torch installed.
    """

    def __init__(self, logdir: str):
        from tensorboard.summary import Writer

        self._writer = Writer(logdir)

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self._writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic losscandles demo run.")
    parser.add_argument("--logdir", default=DEFAULT_LOGDIR, help=f"default: {DEFAULT_LOGDIR}")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    logdir = write_demo_run(args.logdir, args.seed)
    parent = os.path.dirname(os.path.normpath(logdir)) or "."
    print(f"Wrote demo run to {logdir}")
    print(f"Now run: tensorboard --logdir {parent}")


if __name__ == "__main__":
    main()
