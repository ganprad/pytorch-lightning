from abc import ABC
from typing import List, Optional, Any
import numpy as np
import torch
from pytorch_lightning.metrics.functional.map import group_by_key, get_iou, get_average_precision
from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.utils import METRIC_EPS


class MAP(Metric, ABC):
    def __init__(self,
                 iou_threshold: float = 0.5,
                 category2name: dict = {},
                 box_format="top_corner",
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 ):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         )
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.process_group = process_group

        self.iou_threshold = iou_threshold
        self.num_classes = len(category2name)
        self.box_format = box_format
        self.category2name = category2name
        self.average_precisions = {}


    def update(self, preds: Any, target: Any):

        assert len(target) == len(preds)

        for index, category_id in enumerate(target.keys()):
            if category_id in preds:  # if we predicted at leas one object for this class.

                ground_truth = target[category_id]
                predictions =  preds[category_id]
                num_gts = len(ground_truth)

                image_gts = group_by_key(ground_truth, "image_id")

                image_gt_boxes = {
                    img_id: np.array([[float(z) for z in b["bbox"]] for b in boxes]) for img_id, boxes in
                    image_gts.items()
                }
                image_gt_checked = {img_id: np.zeros(len(boxes)) for img_id, boxes in image_gts.items()}

                predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

                # go down dets and mark TPs and FPs
                num_predictions = len(predictions)
                tp = torch.zeros(num_predictions)
                fp = torch.zeros(num_predictions)

                for prediction_index, prediction in enumerate(predictions):
                    box = prediction["bbox"]

                    max_iou = torch.tensor(-np.inf)
                    jmax = torch.tensor([-1])

                    try:
                        gt_boxes = image_gt_boxes[prediction["image_id"]]  # gt_boxes per image
                        gt_checked = image_gt_checked[prediction["image_id"]]  # gt flags per image
                    except KeyError:
                        gt_boxes = []
                        gt_checked = None

                    if len(gt_boxes) > 0:
                        gt_boxes = torch.tensor(gt_boxes)
                        box = torch.tensor(box)
                        ious = get_iou(box.unsqueeze(0), gt_boxes, self.box_format)

                        max_iou = torch.max(ious)
                        jmax = torch.argmax(ious)

                    if max_iou >= self.iou_threshold:
                        if gt_checked[jmax] == 0:
                            tp[prediction_index] = 1.0
                            gt_checked[jmax] = 1
                        else:
                            fp[prediction_index] = 1.0
                    else:
                        fp[prediction_index] = 1.0

                fp = torch.cumsum(fp, axis=0)
                tp = torch.cumsum(tp, axis=0)

                recalls = tp / float(num_gts)
                precisions = tp / np.maximum(tp + fp, METRIC_EPS)
                ap = get_average_precision(precisions, recalls)

            self.average_precisions[self.category2name[category_id]] = ap

    def compute(self):
        return torch.mean(torch.tensor(list(self.average_precisions.values())))


class AveragePrecision(Metric, ABC):
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 ):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         )

        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.process_group = process_group
        self.average_precisions = None

    def update(self, precisions, recalls):
        self.average_precisions = get_average_precision(recalls, precisions)

    def compute(self):
        return self.average_precisions


class IOU(Metric, ABC):
    def __init__(self,
                 compute_on_step: bool = True,
                 dist_sync_on_step: bool = False,
                 process_group: Optional[Any] = None,
                 ):
        super().__init__(compute_on_step=compute_on_step,
                         dist_sync_on_step=dist_sync_on_step,
                         process_group=process_group,
                         )

        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.process_group = process_group
        self.average_precisions = None

    def update(self, pred, targets):
        self.ious = get_iou(pred, targets)

    def compute(self):
        return self.ious
