"""Faster R-CNN detector using the DOFA feature pyramid backbone."""
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from models.dofa_backbone import DOFASimpleFPNBackbone


def build_dofa_faster_rcnn(checkpoint_path: str, num_classes: int, img_size: int = 800,
                            freeze_backbone: bool = False):
    backbone = DOFASimpleFPNBackbone(
        checkpoint_path, img_size=img_size, freeze_backbone=freeze_backbone
    )

    # One anchor size per FPN level (4 levels: strides 4, 8, 16, 32),
    # each with 3 aspect ratios -- standard FPN/Faster R-CNN convention.
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 4,
    )

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=img_size,
        max_size=img_size,
    )
    return model