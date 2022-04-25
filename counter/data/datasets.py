
r"""
Definition of datasets.

Author:
    Yiqun Chen
"""

from typing import Any, List, Tuple, Dict, Sequence, Union
import io
import contextlib
import logging
import numpy as np
import mmcv
from mmcv.utils import print_log
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.api_wrappers import COCO, COCOeval

from counter.utils.eval_MR_multisetup import COCOeval as CityPersonsCOCOeval


@DATASETS.register_module()
class DHDTrafficDataset(CocoDataset):
    CLASSES = ('Pedestrian', )
    def evaluate_det_segm(
        self, 
        results: List[Union[List, Tuple, Dict]], 
        result_files: Dict[str, str], 
        coco_gt: COCO, 
        metrics: Union[str, List[str]], 
        logger: logging.Logger=None, 
        classwise: bool=False, 
        proposal_nums: Sequence[int]=(1, 10, 100), 
        iou_thrs: Sequence[float]=None, 
        metric_items: Union[List[str], str]=None, 
    ) -> Dict[str, float]:
        return super().evaluate_det_segm(
            results, 
            result_files, 
            coco_gt, 
            metrics, 
            logger, 
            classwise, 
            proposal_nums, 
            iou_thrs, 
            metric_items, 
        )

    def _format_for_city_persons_eval(
        self, predictions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        def _format_height_width(p: Dict) -> Dict:
            p["height"] = p["bbox"][2]
            p["width"] = p["bbox"][3]
            return p

        return list(map(_format_height_width, predictions))

    @property
    def id2metric(self):
        return {
            0: "MRReasonable", 
            1: "MRReasonableSmall", 
            2: "MRReasonableHeavyOcclusion", 
            3: "MRAll", 
        }

    def _count_num_bboxes(img: str, predictions: List[Dict]) -> int:
            return sum([img == p["id"] for p in predictions])

    def count_num_bboxes(self, predictions: List[Dict]):
        num_preds_per_img = {}
        img_ids = set([p["id"] for p in predictions])
        for idx in img_ids:
            num_preds_per_img[idx] = self._count_num_bboxes(idx, predictions)
        return num_preds_per_img

    def calc_miss_rate(
        self, 
        results: List[Union[List, Tuple, Dict]], 
        result_files: Dict[str, str], 
        coco_gt: COCO, 
        metrics: str="miss_rate", 
        logger: logging.Logger=None, 
        classwise: bool=False, 
        proposal_nums: Sequence[int]=(1, 10, 100), 
        iou_thrs: Sequence[float]=None, 
        metric_items: Union[List[str], str]=None, 
    ) -> Dict[str, float]:
        r"""
        Calculate Miss Rate by using CityPersons Evaluation tools.

        Args:
            results: 
                Testing results of the dataset.
            result_files: 
                a dict contains json file path.
            coco_gt:
                COCO API object with ground truth annotation.
            metrics:
                This argument is placed to make consistent with 
                mmdet.datasets.coco.CocoDataset
            logger:
                used for printing related information during evaluation.
            classwise:
                This argument is placed to make consistent with 
                mmdet.datasets.coco.CocoDataset
            proposal_nums:
                Proposal number used for evaluating recalls.
            iou_thrs:
                IoU threshold used for evaluating recalls/mAPs. 
                [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] 
                will be used by default.
            metric_items:
                Metric items that will be returned. ``["AR@100", "AR@300",
                "AR@1000", "AR_s@1000", "AR_m@1000", "AR_l@1000" ]`` will be
                used when ``metric=="proposal"``, ``["mAP", "mAP_50", "mAP_75",
                "mAP_s", "mAP_m", "mAP_l"]`` will be returned by default.
        
        Returns:
            eval_results: COCO style evaluation metric.
        """
        metric = metrics
        iou_type = "bbox"
        eval_results = {}
        redirect_string = io.StringIO()

        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        msg = f'Evaluating {metric}...'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        predictions = mmcv.load(result_files["miss_rate"])
        predictions = self._format_for_city_persons_eval(predictions)
        with contextlib.redirect_stdout(redirect_string):
            for id_setup in range(0, 4):
                coco_det = coco_gt.loadRes(predictions)
                cocoEval = CityPersonsCOCOeval(coco_gt, coco_det, iou_type)
                # cocoEval.params.catIds = self.cat_ids
                cocoEval.params.imgIds = self.img_ids
                # cocoEval.params.maxDets = list(proposal_nums)
                # cocoEval.params.iouThrs = iou_thrs
                cocoEval.evaluate(id_setup)
                cocoEval.accumulate()
                eval_results[self.id2metric[id_setup]] = \
                    cocoEval.summarize(id_setup, None)
        print_log('\n' + redirect_string.getvalue(), logger=logger)
        return eval_results

    def evaluate(
        self, 
        results: List[Union[List, Tuple]], 
        metric: Union[str, List[str]]=["bbox", "miss_rate"], 
        logger: logging.Logger=None, 
        jsonfile_prefix: str=None, 
        classwise: bool=False, 
        proposal_nums: Sequence[int]=(1, 10, 100), 
        iou_thrs: Sequence[float]=None, 
        metric_items: Union[str, List[str]]=None, 
    ) -> Dict[str, float]:
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = [
            "bbox", "segm", "proposal", "proposal_fast", "miss_rate"
        ]
        det_metrics = metrics
        eval_results = {}
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        if "miss_rate" in metrics:
            result_files["miss_rate"] = result_files["bbox"]
            eval_results.update(
                self.calc_miss_rate(
                    results, 
                    result_files, 
                    coco_gt, 
                    "miss_rate", 
                    logger, 
                    classwise,
                    proposal_nums, 
                    iou_thrs,
                    metric_items
                )
            )
            det_metrics = list(metrics)
            det_metrics.remove("miss_rate")

        eval_results.update(
            self.evaluate_det_segm(
                results, 
                result_files, 
                coco_gt,
                det_metrics, 
                logger, 
                classwise,
                proposal_nums, 
                iou_thrs,
                metric_items
            )
        )

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results





