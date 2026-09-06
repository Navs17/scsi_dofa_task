"""Generate qualitative prediction images for a trained checkpoint."""
import argparse
import os

import torch
import yaml
from PIL import Image
import torchvision.transforms.functional as TF

from dataset.dior import DIORDetectionDataset, NUM_CLASSES
from models.detector import build_dofa_faster_rcnn
from utils.visualization import draw_boxes


def main(config_path, checkpoint_path, out_dir, num_images):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    test_ds = DIORDetectionDataset(cfg["dataset"]["hf_cache_path"], split="test")

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
    model.eval()

    for idx in range(num_images):
        image_tensor, target = test_ds[idx]
        pil_image = TF.to_pil_image(image_tensor)

        with torch.no_grad():
            pred = model([image_tensor.to(device)])[0]

        gt_img = draw_boxes(pil_image, target["boxes"], target["labels"])
        pred_img = draw_boxes(
            pil_image, pred["boxes"].cpu(), pred["labels"].cpu(),
            scores=pred["scores"].cpu(), score_thresh=0.3,
        )

        combined = Image.new("RGB", (gt_img.width * 2, gt_img.height))
        combined.paste(gt_img, (0, 0))
        combined.paste(pred_img, (gt_img.width, 0))
        out_path = os.path.join(out_dir, f"qual_{idx}.png")
        combined.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_images", type=int, default=6)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.out_dir, args.num_images)