# Fruits Fruits Fruits

A complete fruit detection system using YOLOv8 for real-time object detection via webcam. This project takes fruit images with white backgrounds, augments them onto random indoor backgrounds, trains a YOLOv8 model, and enables real-time detection.

![](https://github.com/ParrIan/FruitVision/blob/main/gif.gif)

## Overview

This project implements an end-to-end pipeline for fruit detection:

1. **Data Augmentation**: Removes white backgrounds from fruit images and pastes them onto random indoor backgrounds with random transformations
2. **Dataset Preparation**: Organizes augmented data into YOLOv8-compatible format with YOLO annotations
3. **Model Training**: Trains YOLOv8 on Google Colab with GPU acceleration
4. **Real-Time Detection**: Runs live detection using webcam with temporal smoothing

## Installation

### Dependencies

The project requires:
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- PIL/Pillow
- NumPy

Install with:
```bash
pip install torch torchvision ultralytics opencv-python pillow numpy
```

### Dataset Setup

1. Download the [fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) and extract to `data/fruits-360/`
2. Extract background images: `unzip data/backgrounds.zip -d data/backgrounds/`

## Usage

### Step 1: Generate Augmented Dataset

#### I augmented all of my data (random placement of fruits over random backgrounds) for the sake of convenience that resulted in many domain gaps:
- The model works best when fruits are thrown as this most similarly resembles the training data (no occlusions from your hands).
- I also didn't attempt to add motion blur to the augmented data so if the fruit is thrown with a large velocity the mdoel will start to have detection issues.
- The Lighing in the data varies a little bit and is something you can play with in  **augment_data.py**, I didn't vary this much for my runs (0.9 - 1.1)

- Many more when it comes to variance of what you would see and expect in day-to-day fruits but my decisions sufficed  for getting this off the ground

```bash
python utils/augment_data.py
```

This creates `data/augmented_data/` with Training and Test splits, each containing:
- Fruit images on natural backgrounds
- Corresponding `.txt` annotation files

### Step 2: Prepare YOLO Dataset

```bash
python utils/prepare_yolo_dataset.py
```

This reorganizes into `data/yolo_dataset/` with proper YOLO structure.

### Step 3: Train Model on Colab

1. Zip the dataset: `zip -r yolo_dataset.zip data/yolo_dataset/`
2. Upload to Google Drive
3. Open `src/train_yolo.ipynb` in Google Colab
4. Set Runtime > Change runtime type > GPU
6. Run Cells and download trained model from `drive/MyDrive/yolo_runs/fruit_detector/weights/best.pt`

### Step 4: Run Real-Time Detection

```bash
cd src
python realtime_yolo.py
```

Customize detection parameters:
```python
detector = YOLORealtimeDetector(
    model_path='../models/weights/best.pt',
    camera_id=0,
    confidence_threshold=0.5,
    target_fruits=[],  # If you want to whitelist certain fruits. ex: ['Apple', 'Lemon']
    smoothing_frames=10,
    imgsz=640, # Model was trained on 640 but decreasing could help if you have performance issues. You will loose some accuracy, pretty noticable falloff at 320
    frame_skip=1  # Can help improve FPS performance by only processing every Nth Frame
)
detector.run()
```

## Model Architecture


### Legacy CNN (Reference)
The `detection_model.py` contains a custom CNN architecture:
- **Backbone**: 4 convolutional blocks (32→64→128→256 channels)
- **Heads**:
  - Classification: 512→256→num_classes
  - Bounding Box: 512→128→4 (x_center, y_center, width, height)
- **Loss**: Combined CrossEntropy + Smooth L1 Loss


## Training Metrics

Example training results on 131 fruit classes:
- **mAP50**: ~0.95 (Intersection over Union threshold = 0.5)
- **mAP50-95**: ~0.85 (averaged across IoU 0.5-0.95)
- **Precision**: ~0.93
- **Recall**: ~0.92

## License

This project uses the [fruits-360 dataset](https://www.kaggle.com/datasets/moltean/fruits) and YOLOv8 from Ultralytics (AGPL-3.0).
