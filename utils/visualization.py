"""Draws predicted (and optionally ground-truth) boxes on an image for
qualitative comparison figures in the presentation.
"""
from PIL import ImageDraw

from dataset.dior import DIOR_CLASSES


def draw_boxes(image, boxes, labels, scores=None, score_thresh=0.3, color="red", gt_color="lime"):
    """image: PIL.Image (RGB). boxes: list/tensor of [x1,y1,x2,y2].
    labels: list/tensor of int class ids (1-indexed, background=0).
    scores: optional list/tensor of confidence scores; if given, boxes
    below score_thresh are skipped (this is a prediction draw call).
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if scores is not None:
            score = scores[i]
            if score < score_thresh:
                continue
            box_color = color
        else:
            score = None
            box_color = gt_color

        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        class_name = DIOR_CLASSES[int(label) - 1] if 0 < int(label) <= len(DIOR_CLASSES) else str(label)
        text = f"{class_name}" + (f" {score:.2f}" if score is not None else "")
        draw.text((x1, max(0, y1 - 12)), text, fill=box_color)

    return img
