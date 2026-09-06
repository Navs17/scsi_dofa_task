"""Evaluate a trained checkpoint's mAP on the DIOR test split."""
import argparse
import torch
import yaml

from dataset.dior import DIORDetectionDataset, NUM_CLASSES
from models.detector import build_dofa_faster_rcnn
from utils.metrics import evaluate_map


def main(config_path, checkpoint_path, max_images):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = DIORDetectionDataset(
        cfg["dataset"]["hf_cache_path"], split="test"
    )

    model = build_dofa_faster_rcnn(
        checkpoint_path=cfg["model"]["dofa_checkpoint"],
        num_classes=NUM_CLASSES,
        img_size=cfg["dataset"]["img_size"],
        use_lora=cfg["model"].get("use_lora", False),
        lora_r=cfg["model"].get("lora_r", 8),
        lora_alpha=cfg["model"].get("lora_alpha", 16),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_map(model, test_ds, device, max_images=max_images)
    print(f"\n=== Results for {checkpoint_path} ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_images", type=int, default=500,
                         help="Subset of test set for faster eval; use None for full set")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.max_images)