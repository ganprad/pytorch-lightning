import json
from typing import List, Optional, Any
import numpy as np
import torch
from pytorch_lightning.metrics.functional.map import get_category2name, group_by_key, recall_precision
from pytorch_lightning.metrics.metric import Metric


class MAP(Metric):
    def __init__(self,
                 num_classes: int = 80, #num_classes in COCO.
                 iou_threshold: float = 0.5,
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
        self.num_classes = num_classes

        # self.add_state("average_precision", default=torch.zeros(len(ground_truth)), dist_reduce_fx=None)
        # self.add_state("recalls", default=torch.zeros(len(ground_truth)), dist_reduce_fx=None)
        # self.add_state("precisions", default=torch.zeros(len(ground_truth)), dist_reduce_fx=None)
        self.add_state("map", default=torch.zeros([0]), dist_reduce_fx=None)
        self.add_state("average_precisions",
                       default=torch.zeros(self.num_classes),
                       dist_reduce_fx=None)

    def update(self, preds: Any, target: Any):

        assert len(target.keys()) == len(preds.keys())

        for i, category_id in enumerate(range(0, self.num_classes)):
            if category_id in target:
                _, _, average_precision = recall_precision(target[category_id],
                                                                          preds[category_id],
                                                                          self.iou_threshold)
                self.average_precisions[i] = torch.tensor(average_precision)

    def compute(self):
        map = torch.mean(self.average_precisions)
        return map


KEYS = ['image_id', 'bbox', 'score', 'category_id']

ground_truth = "ground_truth_annotations.json"
predictions = "predictions.json"
iou_threshold = 0.5

with open(ground_truth) as f:
    gt = json.load(f)

with open(predictions) as f:
    pred = json.load(f)

gt_annotations = gt["annotations"]
categories = gt["categories"]


gt_by_id = group_by_key(gt_annotations, "category_id")
pred_by_id = group_by_key(pred, "category_id")

# print(gt_by_id.keys())
# print(pred_by_id.keys())
assert len(gt_by_id.keys()) == len(pred_by_id.keys())

category2name = get_category2name(categories)

num_class_names = len(category2name.values())

print("Class_names in ground truth = ", sorted(category2name.values()))

aps = np.zeros(num_class_names)

# class_names: List[str] = []
#
# for i, category_id in enumerate(category2name.keys()):
#     if category_id in pred_by_id:  # if we predicted at least one object for this class.
#         recalls, precisions, average_precision = recall_precision(
#             gt_by_id[category_id], pred_by_id[category_id], iou_threshold
#         )
#         aps[i] = average_precision
#
#     class_names += [category2name[category_id]]
#
# mAP = np.mean(aps)
# print("Average per class mean average precision = ", mAP)
#
# for j in sorted(zip(class_names, aps.flatten().tolist())):
#     print(j)
#%%




            # TODO: pred_by_id and gt_by_id are dataset specific:
            # TODO: Do preds and target have to be a torch.tensor if so then how might structure the input?

map_test = MAP(iou_threshold=0.5)
map_out = map_test(pred_by_id, gt_by_id)
print(map_out)