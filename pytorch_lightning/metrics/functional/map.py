#%%
import argparse
import json
from pytorch_lightning.metrics.utils import METRIC_EPS
from collections import OrderedDict
from typing import Tuple, List, Dict, Any, Union

import numpy as np
import torch

from collections import defaultdict


def group_by_key(list_dicts: List[dict], key: Any) -> defaultdict:
    """Groups list of dictionaries by key.
    >>> c = [{"a": 1, "b": "Wednesday"}, {"a": (1, 2, 3), "b": 16.5}]
    defaultdict(list,
            {1: [{'a': 1, 'b': 'Wednesday'}],
             (1, 2, 3): [{'a': (1, 2, 3), 'b': 16.5}]})
    Args:
        list_dicts:
        key:
    Returns:
    """
    groups: defaultdict = defaultdict(list)
    for detection in list_dicts:
        groups[detection[key]].append(detection)
    return groups


def get_envelope(precisions: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """Compute the envelope of the precision curve.

    Args:
      precisions:

    Returns: enveloped precision

    """
    for i in range(precisions.size(0) - 1, 0, -1):
        precisions[i - 1] = torch.max(precisions[i - 1], precisions[i])
    return precisions


def get_ap(recalls: Union[torch.Tensor, np.ndarray], precisions: Union[torch.Tensor, np.ndarray]) \
        -> Union[torch.Tensor, float]:
    """Calculate area under precision/recall curve.

    Args:
      recalls:
      precisions:

    Returns:

    """
    # correct AP calculation
    # first append sentinel values at the end
    # recalls = torch.tensor(recalls)
    # precisions = torch.tensor(precisions)

    recalls = torch.cat((torch.tensor([0.0]),
                         recalls,
                         torch.tensor([1.0])))

    precisions = torch.cat((torch.tensor([0.0]),
                            precisions,
                            torch.tensor([0.0])))

    precisions = get_envelope(precisions)

    # to calculate area under PR curve, look for points where X axis (recall) changes value
    i = torch.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = torch.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def get_overlaps(gt_boxes: Union[torch.Tensor, np.ndarray],
                 box: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    i_xmin = torch.max(gt_boxes[:, 0], box[0])
    i_ymin = torch.max(gt_boxes[:, 1], box[1])

    gt_xmax = gt_boxes[:, 0] + gt_boxes[:, 2]
    gt_ymax = gt_boxes[:, 1] + gt_boxes[:, 3]

    box_xmax = box[0] + box[2]
    box_ymax = box[1] + box[3]

    i_xmax = torch.min(gt_xmax, box_xmax)
    i_ymax = torch.min(gt_ymax, box_ymax)

    iw = torch.max(i_xmax - i_xmin, torch.tensor(0.0))
    ih = torch.max(i_ymax - i_ymin, torch.tensor(0.0))

    intersection = iw * ih

    union = box[2] * box[3] + gt_boxes[:, 2] * gt_boxes[:, 3] - intersection

    overlaps = intersection / (union + 1e-7)

    return overlaps


def recall_precision(
    gt: List[dict], predictions: List[dict], iou_threshold: float
) -> Tuple[np.array, np.array, np.array]:
    num_gts = len(gt)
    image_gts = group_by_key(gt, "image_id")

    image_gt_boxes = {
        img_id: torch.tensor([[float(z) for z in b["bbox"]] for b in boxes]) for img_id, boxes in image_gts.items()
    }
    image_gt_checked = {img_id: torch.zeros(len(boxes)) for img_id, boxes in image_gts.items()}

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tp = torch.zeros(num_predictions)
    fp = torch.zeros(num_predictions)

    for prediction_index, prediction in enumerate(predictions):
        box = torch.tensor(prediction["bbox"])

        max_overlap = -np.inf
        jmax = -1

        try:
            gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
            gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_overlaps(gt_boxes, box)

            max_overlap = torch.max(overlaps)
            jmax = torch.argmax(overlaps)

        if max_overlap >= iou_threshold:
            if gt_checked[jmax] == 0:
                tp[prediction_index] = 1.0
                gt_checked[jmax] = 1
            else:
                fp[prediction_index] = 1.0
        else:
            fp[prediction_index] = 1.0

    # compute precision recall
    fp = torch.cumsum(fp, axis=0)
    tp = torch.cumsum(tp, axis=0)

    recalls = tp / float(num_gts)

    # avoid divide by zero in case the first detection matches a difficult ground truth
    precisions = tp / (tp + fp + METRIC_EPS)

    ap = get_ap(recalls, precisions)

    return recalls, precisions, ap


def get_category2name(categories: List[Dict[str, Any]]) -> OrderedDict:
    """Creates mapping from category_id to category_name

    Args:
        categories: list of the type:
            "categories": [
                {"id": 1,"name": "person"},
                {"id": 2,"name": "bicycle"},
                {"id": 3,"name": "car"},
                {"id": 4,"name": "motorcycle"},
                {"id": 5,"name": "airplane"},
                ...
                {"id": 89,"name": "hair drier"},
                {"id": 90,"name": "toothbrush"}
            ]
    Returns: {<category_id>: <name>}

    """
    result = OrderedDict()
    for element in categories:
        result[element["id"]] = element["name"]
    return result
