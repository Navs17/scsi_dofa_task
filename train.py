"""Training loop: full fine-tuning baseline (or LoRA, once models/lora.py exists)."""
import argparse
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from dataset.dior import DIORDetectionDataset, collate_fn, NUM_CLASSES
from models.detector import build_dofa_faster_rcnn
from utils.seed import set_seed


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = DIORDetectionDataset(
        cfg["dataset"]["hf_cache_path"], split="train",
        max_samples=cfg["dataset"].get("max_samples"),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_fn,
    )

    model = build_dofa_faster_rcnn(
        checkpoint_path=cfg["model"]["dofa_checkpoint"],
        num_classes=NUM_CLASSES,
        img_size=cfg["dataset"]["img_size"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        use_lora=cfg["model"].get("use_lora", False),
        lora_r=cfg["model"].get("lora_r", 8),
        lora_alpha=cfg["model"].get("lora_alpha", 16),
    )
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    optimizer = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        epoch_start = time.time()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % cfg["train"]["log_every"] == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"loss={loss.item():.4f} "
                      f"({', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())})")

        avg_loss = running_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        print(f"== Epoch {epoch} done: avg_loss={avg_loss:.4f}, time={elapsed:.1f}s ==")

        ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)