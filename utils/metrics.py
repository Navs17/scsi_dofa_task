"""COCO-style mAP evaluation for the DIOR detection models."""
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def build_coco_gt(dataset):
    """Builds a COCO-format ground-truth object directly from our
    DIORDetectionDataset (test/val split), so we can reuse pycocotools'
    official mAP implementation without a separate annotation file.
    """
    images = []
    annotations = []
    categories = [{"id": i, "name": str(i)} for i in range(1, 21)]  # DIOR: 20 classes, 1-indexed
    ann_id = 1

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        image_id = int(target["image_id"].item())
        images.append({"id": image_id, "width": 0, "height": 0})  # size unused by COCOeval bbox metric

        boxes = target["boxes"]
        labels = target["labels"]
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box.tolist()
            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations, "categories": categories}
    coco_gt.createIndex()
    return coco_gt


@torch.no_grad()
def evaluate_map(model, dataset, device, max_images: int = None):
    """Runs the model over the dataset, collects predictions in COCO format,
    and computes mAP@[.5:.95] and mAP@.5 using pycocotools.
    """
    model.eval()
    coco_gt = build_coco_gt(dataset)

    results = []
    n = len(dataset) if max_images is None else min(max_images, len(dataset))

    for idx in range(n):
        image, target = dataset[idx]
        image_id = int(target["image_id"].item())

        pred = model([image.to(device)])[0]
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "image_id": image_id,
                "category_id": int(label.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(score.item()),
            })

    if len(results) == 0:
        print("WARNING: model produced zero predictions across the eval set.")
        return {}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "mAP@[.5:.95]": coco_eval.stats[0],
        "mAP@.5": coco_eval.stats[1],
        "mAP@.75": coco_eval.stats[2],
    }