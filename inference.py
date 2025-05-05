#!/usr/bin/env python
import os
import argparse
import cv2
import numpy as np
import torch
import mmcv
from mmcv.config import Config
from mmpose.apis import inference_top_down_pose_model, init_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo

def parse_args():
    parser = argparse.ArgumentParser(description='Inference on cephalometric images')
    parser.add_argument('--config', 
                      default='cephalometric_hrnetv2_w18_config.py',
                      help='Config file')
    parser.add_argument('--checkpoint', 
                      help='Checkpoint file')
    parser.add_argument('--img-path', 
                      help='Path to input image')
    parser.add_argument('--out-file', 
                      help='Path to output image')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # Build the model
    cfg = Config.fromfile(args.config)
    
    # Load dataset info
    dataset_info = DatasetInfo(cfg.data.test.dataset_info)
    
    # Load the model
    model = init_pose_model(
        cfg, args.checkpoint, device=args.device)
    
    # Load the image
    image = mmcv.imread(args.img_path)
    
    # Assuming we're working with full image bounding box
    h, w, _ = image.shape
    center = np.array([w // 2, h // 2])
    scale = np.array([w, h])
    
    # Create a dummy person info for top-down model input
    person_results = [{'bbox': np.array([0, 0, w, h, 1.0])}]
    
    # Do inference
    pose_results, returned_outputs = inference_top_down_pose_model(
        model,
        image,
        person_results,
        bbox_thr=0.0,
        format='xyxy',
        dataset_info=dataset_info)
    
    # Calculate mean radial error if ground truth is available
    # This would require ground truth keypoints
    
    # Visualize result
    vis_result = vis_pose_result(
        model,
        image,
        pose_results,
        dataset_info=dataset_info,
        kpt_score_thr=0.3,
        show=False)
    
    # Save result
    if args.out_file:
        mmcv.imwrite(vis_result, args.out_file)
        print(f"Visualization saved to {args.out_file}")
    else:
        mmcv.imshow(vis_result, 'result')
    
    # Print predicted keypoints
    print("Predicted keypoints:")
    keypoints = pose_results[0]['keypoints']
    for i, (x, y, score) in enumerate(keypoints):
        keypoint_name = dataset_info.keypoint_ids[i]['name']
        print(f"{keypoint_name}: ({x:.2f}, {y:.2f}), score: {score:.4f}")

if __name__ == '__main__':
    main() 