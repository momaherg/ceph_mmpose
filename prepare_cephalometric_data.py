import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
import ast

# Define landmark columns
landmark_cols = [
    'sella_x', 'sella_y', 'nasion_x', 'nasion_y', 'A point_x', 'A point_y',
    'B point_x', 'B point_y', 'upper 1 tip_x', 'upper 1 tip_y',
    'upper 1 apex_x', 'upper 1 apex_y', 'lower 1 tip_x', 'lower 1 tip_y',
    'lower 1 apex_x', 'lower 1 apex_y', 'ANS_x', 'ANS_y', 'PNS_x', 'PNS_y',
    'Gonion x', 'Gonion y', 'Menton_x', 'Menton_y', 'ST Nasion_x',
    'ST Nasion_y', 'Tip of the nose_x', 'Tip of the nose_y', 'Subnasal_x',
    'Subnasal_y', 'Upper lip_x', 'Upper lip_y', 'Lower lip_x',
    'Lower lip_y', 'ST Pogonion_x', 'ST Pogonion_y', 'gnathion_x',
    'gnathion_y'
]

# Create directories
os.makedirs('data/cephalometric/images/train', exist_ok=True)
os.makedirs('data/cephalometric/images/val', exist_ok=True)
os.makedirs('data/cephalometric/annotations', exist_ok=True)

# Load JSON data using pandas
print("Loading data from JSON...")
data = pd.read_json("/content/drive/MyDrive/Lala's Masters/train_data_pure_old_numpy.json")

# Helper function to convert image data to numpy array
def convert_to_numpy_array(img_data):
    # If the data is already a numpy array
    if isinstance(img_data, np.ndarray):
        img_array = img_data
    # If the data is a string representation of a list/array
    elif isinstance(img_data, str):
        try:
            # Try to convert string representation to actual array
            img_array = np.array(ast.literal_eval(img_data))
        except:
            print("Error converting string to array")
            return None
    # For other formats, try direct conversion
    else:
        try:
            img_array = np.array(img_data)
        except:
            print("Failed to convert to numpy array")
            return None
    
    # Check if the array is a flattened image (like 50176, 3)
    if len(img_array.shape) == 2 and img_array.shape[0] == 50176 and img_array.shape[1] == 3:
        # Reshape to 224x224x3 (assuming this is the original shape)
        img_array = img_array.reshape(224, 224, 3)
    
    # For grayscale depth images that might be flattened (50176,)
    if len(img_array.shape) == 1 and img_array.shape[0] == 50176:
        # Reshape to 224x224
        img_array = img_array.reshape(224, 224)
        
    return img_array

# Extract landmark names (without _x, _y)
landmark_names = []
for i in range(0, len(landmark_cols), 2):
    name = landmark_cols[i].replace('_x', '')
    if name.endswith('x'):  # For cases like 'Gonion x'
        name = name[:-2]
    landmark_names.append(name)

# Define COCO format
coco_train = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "cephalometric"}]
}
coco_val = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "cephalometric"}]
}

# Split the data
train_indices, val_indices = train_test_split(range(len(data)), test_size=0.2, random_state=42)

print("Processing train data...")
for idx, i in enumerate(tqdm(train_indices)):
    # Save the image
    img_id = int(data.iloc[i]['patient_id'])
    img_filename = f"{img_id:08d}.jpg"
    
    # Check if we have depth feature or image
    if 'depth_feature' in data.columns:
        # Process depth feature (convert to RGB for compatibility)
        depth_data = data.iloc[i]['depth_feature']
        depth_img = convert_to_numpy_array(depth_data)
        
        if depth_img is None:
            print(f"Skipping image {img_id} due to conversion error")
            continue
            
        depth_img = depth_img.astype(np.float32)
        # Normalize to 0-255
        depth_img = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-10) * 255).astype(np.uint8)
        
        # If the depth image is grayscale, convert to RGB
        if len(depth_img.shape) == 2:
            depth_img_rgb = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
            img = depth_img_rgb
        else:
            img = depth_img
    else:
        # Use regular image
        image_data = data.iloc[i]['Image']
        img = convert_to_numpy_array(image_data)
        
        if img is None:
            print(f"Skipping image {img_id} due to conversion error")
            continue
    
    # Check that image is a valid numpy array with correct dimensions
    if not isinstance(img, np.ndarray):
        print(f"Skipping image {img_id}: not a numpy array")
        continue
    
    # Ensure image has 3 channels for RGB
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Converting image {img_id} with shape {img.shape} to RGB")
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            print(f"Skipping image {img_id} with incompatible shape: {img.shape}")
            continue
    
    cv2.imwrite(f"data/cephalometric/images/train/{img_filename}", img)
    
    # Add image info
    img_height, img_width = img.shape[:2]
    coco_train["images"].append({
        "id": img_id,
        "file_name": img_filename,
        "height": img_height,
        "width": img_width
    })
    
    # Extract keypoints
    keypoints = []
    for j in range(0, len(landmark_cols), 2):
        x = float(data.iloc[i][landmark_cols[j]])
        y = float(data.iloc[i][landmark_cols[j+1]])
        keypoints.extend([x, y, 2])  # 2 means visible
    
    # Create annotation
    coco_train["annotations"].append({
        "id": idx + 1,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": keypoints,
        "bbox": [0, 0, img_width, img_height],  # Full image as bbox
        "area": img_width * img_height,
        "iscrowd": 0,
        "num_keypoints": len(landmark_names)
    })

print("Processing validation data...")
for idx, i in enumerate(tqdm(val_indices)):
    # Save the image
    img_id = int(data.iloc[i]['patient_id'])
    img_filename = f"{img_id:08d}.jpg"
    
    # Check if we have depth feature or image
    if 'depth_feature' in data.columns:
        # Process depth feature (convert to RGB for compatibility)
        depth_data = data.iloc[i]['depth_feature']
        depth_img = convert_to_numpy_array(depth_data)
        
        if depth_img is None:
            print(f"Skipping image {img_id} due to conversion error")
            continue
            
        depth_img = depth_img.astype(np.float32)
        # Normalize to 0-255
        depth_img = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-10) * 255).astype(np.uint8)
        
        # If the depth image is grayscale, convert to RGB
        if len(depth_img.shape) == 2:
            depth_img_rgb = cv2.cvtColor(depth_img, cv2.COLOR_GRAY2RGB)
            img = depth_img_rgb
        else:
            img = depth_img
    else:
        # Use regular image
        image_data = data.iloc[i]['Image']
        img = convert_to_numpy_array(image_data)
        
        if img is None:
            print(f"Skipping image {img_id} due to conversion error")
            continue
    
    # Check that image is a valid numpy array with correct dimensions
    if not isinstance(img, np.ndarray):
        print(f"Skipping image {img_id}: not a numpy array")
        continue
    
    # Ensure image has 3 channels for RGB
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Converting image {img_id} with shape {img.shape} to RGB")
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            print(f"Skipping image {img_id} with incompatible shape: {img.shape}")
            continue
    
    cv2.imwrite(f"data/cephalometric/images/val/{img_filename}", img)
    
    # Add image info
    img_height, img_width = img.shape[:2]
    coco_val["images"].append({
        "id": img_id,
        "file_name": img_filename,
        "height": img_height,
        "width": img_width
    })
    
    # Extract keypoints
    keypoints = []
    for j in range(0, len(landmark_cols), 2):
        x = float(data.iloc[i][landmark_cols[j]])
        y = float(data.iloc[i][landmark_cols[j+1]])
        keypoints.extend([x, y, 2])  # 2 means visible
    
    # Create annotation
    coco_val["annotations"].append({
        "id": idx + 1,
        "image_id": img_id,
        "category_id": 1,
        "keypoints": keypoints,
        "bbox": [0, 0, img_width, img_height],  # Full image as bbox
        "area": img_width * img_height,
        "iscrowd": 0,
        "num_keypoints": len(landmark_names)
    })

# Define flip pairs (anatomically corresponding pairs that flip during augmentation)
# For cephalometric landmarks, we might have some pairs like:
flip_pairs = []  # Fill with appropriate pairs if there are any

# Create dataset info
dataset_info = {
    "dataset_name": "cephalometric",
    "paper_info": {
        "author": "Custom",
        "title": "Cephalometric Dataset",
        "year": 2023,
    },
    "keypoint_info": {
        f"{i+1}": {"name": name, "id": i+1} for i, name in enumerate(landmark_names)
    },
    "skeleton_info": {},
    "joint_weights": [1.0] * len(landmark_names),
    "sigmas": [0.6] * len(landmark_names),
}

# Add flip_pairs to dataset_info
if flip_pairs:
    dataset_info["flip_pairs"] = flip_pairs

# Save COCO annotations
with open('data/cephalometric/annotations/train.json', 'w') as f:
    json.dump(coco_train, f)

with open('data/cephalometric/annotations/val.json', 'w') as f:
    json.dump(coco_val, f)

# Save dataset info
with open('data/cephalometric/annotations/dataset_info.json', 'w') as f:
    json.dump(dataset_info, f)

print("Data preparation complete!") 