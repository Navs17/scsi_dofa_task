"""PyTorch Dataset wrapper for DIOR (HichTala/dior, COCO-style bbox format),
converting to the {boxes, labels} format torchvision's detection models expect.
"""
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

DIOR_CLASSES = [
    'Airplane', 'Airport', 'Baseball field', 'Basketball court', 'Bridge',
    'Chimney', 'Dam', 'Expressway service area', 'Expressway toll station',
    'Golf course', 'Ground track field', 'Harbor', 'Overpass', 'Ship',
    'Stadium', 'Storage tank', 'Tennis court', 'Train station', 'Vehicle',
    'Wind mill',
]
# torchvision detection convention: label 0 is reserved for background,
# so we shift every DIOR category id up by 1.
NUM_CLASSES = len(DIOR_CLASSES) + 1  # +1 for background


class DIORDetectionDataset(Dataset):
    def __init__(self, hf_dataset_path: str, split: str, transforms=None):
        ds = load_from_disk(hf_dataset_path)
        self.data = ds[split]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        image = example["image"].convert("RGB")

        objects = example["objects"]
        boxes_xywh = objects["bbox"]     # list of [x, y, w, h]
        categories = objects["category"]  # 0-indexed DIOR class ids

        boxes = []
        labels = []
        for (x, y, w, h), cat in zip(boxes_xywh, categories):
            if w <= 0 or h <= 0:
                continue  # guard against degenerate boxes
            boxes.append([x, y, x + w, y + h])  # convert to x1,y1,x2,y2
            labels.append(cat + 1)  # shift for background class

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([example["image_id"]]),
            "area": torch.as_tensor(objects["area"], dtype=torch.float32),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))