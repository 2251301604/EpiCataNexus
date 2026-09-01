#!/usr/bin/env python3
"""Train EpiCataNexus from prepared tensor batches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from epicatanexus.config import load_config, resolve_path
from epicatanexus.io import MODEL_INPUT_KEYS, load_prepared_batches, move_batch
from epicatanexus.metrics import regression_metrics
from epicatanexus.models import EpiCataNexus


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_inputs(batch: dict) -> dict:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


@torch.no_grad()
def evaluate(model, batches, device) -> dict[str, float]:
    model.eval()
    targets, predictions = [], []
    for raw_batch in batches:
        batch = move_batch(raw_batch, device)
        prediction = model(**model_inputs(batch))
        predictions.extend(prediction.detach().cpu().view(-1).tolist())
        targets.extend(batch["target"].detach().cpu().view(-1).tolist())
    return regression_metrics(targets, predictions)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/kcat.yaml"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 3407))
    set_seed(seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    train_batches = load_prepared_batches(resolve_path(args.config, config["data"]["prepared_train_batches"]))
    valid_batches = load_prepared_batches(resolve_path(args.config, config["data"]["prepared_valid_batches"]))
    if any("target" not in batch for batch in train_batches + valid_batches):
        raise KeyError("Training and validation batches must contain 'target'.")

    model = EpiCataNexus.from_mapping(config["model"]).to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = max(1, int(training.get("accumulation_steps", 1)))
    patience = int(training.get("early_stopping_patience", 20))
    checkpoint = resolve_path(args.config, training["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_r2, stale_epochs = -float("inf"), 0

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, raw_batch in enumerate(train_batches, start=1):
            batch = move_batch(raw_batch, device)
            prediction = model(**model_inputs(batch))
            loss = torch.nn.functional.mse_loss(prediction.view(-1), batch["target"].view(-1))
            (loss / accumulation).backward()
            if step % accumulation == 0 or step == len(train_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(model, valid_batches, device)
        print(f"epoch={epoch} " + " ".join(f"{key}={value:.6f}" for key, value in metrics.items()))
        if metrics["r2"] > best_r2:
            best_r2, stale_epochs = metrics["r2"], 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": config["model"],
                    "task": config["task"],
                    "seed": seed,
                    "validation_metrics": metrics,
                },
                checkpoint,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    summary = checkpoint.with_suffix(".metrics.json")
    summary.write_text(json.dumps({"best_validation_r2": best_r2}, indent=2) + "\n", encoding="utf-8")
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

