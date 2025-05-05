import os.path as osp

import numpy as np

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt2dSviewRgbImgTopDownDataset


@DATASETS.register_module()
class CephalometricDataset(Kpt2dSviewRgbImgTopDownDataset):
    """Cephalometric dataset for landmark detection.

    The dataset loads raw features and transform them into
    keypoints for orthodontic landmark detection.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        dataset_info (dict): A dict containing dataset information.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 dataset_info=None,
                 test_mode=False):
        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        
        self.ann_info['use_different_joint_weights'] = False
        self.ann_info['flip_pairs'] = []  # No flip pairs for now, could be added later
        
    def _get_db(self):
        """Load dataset."""
        gt_db = []
        bbox_id = 0
        
        for img_id in self.img_ids:
            img_ann_ids = self.coco.getAnnIds(imgIds=img_id)
            objs = self.coco.loadAnns(img_ann_ids)
            
            for obj in objs:
                if max(obj['keypoints']) == 0:
                    continue
                
                joints_3d = np.zeros((self.ann_info['num_joints'], 3),
                                     dtype=np.float32)
                joints_3d_visible = np.zeros(
                    (self.ann_info['num_joints'], 3), dtype=np.float32)
                
                keypoints = np.array(obj['keypoints']).reshape(-1, 3)
                
                joints_3d[:, :2] = keypoints[:, :2]
                joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])
                
                # Get center and scale
                image_info = self.coco.loadImgs(img_id)[0]
                center, scale = self._xywh2cs(*obj['bbox'][:4])
                
                gt_db.append({
                    'image_file': osp.join(self.img_prefix, image_info['file_name']),
                    'center': center,
                    'scale': scale,
                    'rotation': 0,
                    'joints_3d': joints_3d,
                    'joints_3d_visible': joints_3d_visible,
                    'dataset': self.dataset_name,
                    'bbox': obj['bbox'],
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                })
                
                bbox_id = bbox_id + 1
                
        return gt_db
        
    def evaluate(self, outputs, res_folder, metric='MRE', **kwargs):
        """Evaluate cephalometric keypoint detection.

        Args:
            outputs (list(dict)): Outputs of the top-down model.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Options: 'MRE' (mean radial error).

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['MRE']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'Metric {metric} is not supported')

        res_file = osp.join(res_folder, 'result_keypoints.json')
        
        kpts = []
        for output in outputs:
            preds = output['preds']
            boxes = output['boxes']
            image_paths = output['image_paths']
            bbox_ids = output['bbox_ids']

            batch_size = len(image_paths)
            for i in range(batch_size):
                image_id = self.name2id[image_paths[i][len(self.img_prefix):]]

                kpts.append({
                    'keypoints': preds[i].tolist(),
                    'center': boxes[i][0:2].tolist(),
                    'scale': boxes[i][2:4].tolist(),
                    'area': float(boxes[i][4]),
                    'score': float(boxes[i][5]),
                    'image_id': image_id,
                    'bbox_id': bbox_ids[i]
                })
        
        self._write_keypoint_results(kpts, res_file)
        
        info_str = self._report_metric(res_file, metrics)
        name_value = {}
        for key, value in info_str.items():
            name_value[key] = value
        
        return name_value
        
    def _report_metric(self, res_file, metrics):
        """Keypoint evaluation.

        Args:
            res_file (str): Json file stored prediction results.
            metrics (str | list[str]): Metric to be performed.
                Options: 'MRE'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        info_str = {}
        
        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)
        
        outputs = []
        gts = []
        
        for pred, item in zip(preds, self.db):
            outputs.append(np.array(pred['keypoints'])[:, :2])
            gts.append(np.array(item['joints_3d'])[:, :2])
            
        outputs = np.array(outputs)
        gts = np.array(gts)
        
        if 'MRE' in metrics:
            # Calculate Mean Radial Error (MRE)
            errors = np.sqrt(np.sum((outputs - gts) ** 2, axis=2))
            mre = np.mean(errors)
            
            info_str['MRE'] = mre
            
        return info_str 