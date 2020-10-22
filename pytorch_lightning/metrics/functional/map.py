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


def get_average_precision(precisions, recalls):
    r = torch.cat([torch.tensor([0]), recalls, torch.tensor([1])])
    p = torch.cat([torch.tensor([1]), precisions, torch.tensor([0])])#Changed 1
    return torch.trapz(p, r)


def get_iou(pred, target, box_format="top_corner"):
    if box_format=="top_corner":
        pred_x1, pred_y1, pred_width, pred_height = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_x1, target_y1, target_width, target_height = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        target_x2 = target_x1 + target_width
        target_y2 = target_y1 + target_height
        pred_x2 = pred_x1 + pred_width
        pred_y2 = pred_y1 + pred_height

    elif box_format=="corners":
        pred_x1, pred_y1, pred_x2, pred_y2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_x1, target_y1, target_x2, target_y2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    elif box_format=="mid_point":
        pred_mid_x, pred_mid_y, pred_width, pred_height = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        target_mid_x, target_mid_y, target_width, target_height = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

        target_x1 = target_mid_x-(target_width/2)
        target_y1 = target_mid_y-(target_height/2)
        target_x2 = target_mid_x + (target_width/2)
        target_y2 = target_mid_y + (target_height/2)

        pred_x1 = pred_mid_x-(pred_width/2)
        pred_y1 = pred_mid_y-(pred_height/2)
        pred_x2 = pred_mid_x + (pred_width/2)
        pred_y2 = pred_mid_y + (pred_height/2)

    i_xmin = np.maximum(target_x1, pred_x1)
    i_ymin = np.maximum(target_y1, pred_y1)
    i_xmax = -np.maximum(-target_x2, -pred_x2)
    i_ymax = -np.maximum(-target_y2, -pred_y2)
    iw = (i_xmax - i_xmin).clamp(0.0)
    ih = (i_ymax - i_ymin).clamp(0.0)
    intersection = iw * ih
    union = (pred_x2-pred_x1)*(pred_y2-pred_y1) + (target_x2-target_x1)*(target_y2-target_y1) - intersection

    return intersection/(union+METRIC_EPS)

@torch.jit.script
def groupby(x, keys):
    unique = torch.unique(keys)
    gp = []
    for i in unique:
        gp.append(x[keys == i])
    return gp
