# Cephalometric Landmark Detection

This project uses MMPose to train a model for automatically detecting cephalometric landmarks on lateral patient photographs, aiding in orthodontic diagnosis without x-ray radiation.

## Dataset

The dataset consists of 1501 patient records with lateral photographs and labeled cephalometric landmarks. The dataset provides 19 landmark points for each patient, including:
- Sella, Nasion
- A point and B point 
- Tips and apices of upper and lower incisors
- ANS (Anterior Nasal Spine), PNS (Posterior Nasal Spine)
- Gonion, Menton
- And other facial landmarks

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the data

The dataset is stored in `data/train_data_pure_depth.pkl`. Run the data preparation script to convert it to the MMPose format:

```bash
python prepare_cephalometric_data.py
```

This will:
- Extract images and landmarks from the pickle file
- Create a COCO-format dataset for MMPose
- Split the data into training and validation sets
- Save the dataset info and annotations

### 3. Training

Train the model using the HRNetV2 backbone:

```bash
python train_cephalometric.py
```

By default, this will:
- Use the HRNetV2-W18 model pretrained on AFLW dataset
- Fine-tune for 100 epochs
- Save checkpoints to `work_dirs/cephalometric/`
- Log training progress, including Mean Radial Error (MRE) for each epoch
- Use the best checkpoint (lowest MRE) for evaluation

### 4. Inference

To run inference on a new image:

```bash
python inference.py --checkpoint work_dirs/cephalometric/best_MRE.pth --img-path /path/to/image.jpg --out-file output.jpg
```

## Model Details

This project uses the HRNetV2-W18 model pretrained on facial landmark detection and fine-tunes it for cephalometric landmarks. The model is configured to detect 19 landmarks with the following features:

- High-resolution network architecture 
- Top-down keypoint detection approach
- Heatmap-based landmark localization
- Mean Radial Error (MRE) as the evaluation metric

## Configuration

The model configuration is specified in `cephalometric_hrnetv2_w18_config.py`. You can modify:
- Learning rate
- Batch size
- Data augmentation parameters
- Number of epochs
- Model architecture details

## Results

After training, the model's performance will be evaluated using Mean Radial Error (MRE), which measures the average Euclidean distance between predicted and ground truth landmarks. The training logs will display the MRE for each epoch, allowing you to track the model's progress. 