"""
Object Detection Dataset Loader

Loads augmented fruit images with YOLO format bounding box annotations.
Each image has a corresponding .txt file with format:
class_id x_center y_center width height (all normalized 0-1)
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class FruitDetectionDataset(Dataset):
    """
    Dataset loader for fruit object detection with YOLO format annotations.

    Dataset structure:
    - data/augmented/Training/Apple 5/image1.jpg
    - data/augmented/Training/Apple 5/image1.txt (YOLO format label)
    """

    def __init__(self, root_dir, transform=None, target_size=(640, 480)):
        """
        Initialize detection dataset.

        Args:
            root_dir: Root directory containing class subdirectories
            transform: Optional transform for image augmentation
            target_size: Target image size (width, height)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.images = []
        self.annotations = []
        self.class_names = []
        self.class_to_idx = {}

        self._load_dataset()

    def _load_dataset(self):
        """Load all images and their corresponding annotations."""
        # Get class names from subdirectories
        self.class_names = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')
        ])

        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load images and annotations
        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)

            # Get all images in this class directory
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, filename)

                    # Check for corresponding label file
                    label_filename = os.path.splitext(filename)[0] + '.txt'
                    label_path = os.path.join(class_dir, label_filename)

                    if os.path.exists(label_path):
                        self.images.append(img_path)
                        self.annotations.append(label_path)

    def _load_annotation(self, annotation_path):
        """
        Load YOLO format annotation.

        Args:
            annotation_path: Path to .txt annotation file

        Returns:
            dict: {'class_id': int, 'bbox': [x_center, y_center, width, height]}
        """
        with open(annotation_path, 'r') as f:
            line = f.readline().strip()

        if not line:
            return None

        parts = line.split()
        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:5]]  # x_center, y_center, width, height (normalized)

        return {'class_id': class_id, 'bbox': bbox}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get image and annotation.

        Returns:
            tuple: (image_tensor, target_dict)
            - image_tensor: shape (3, H, W)
            - target_dict: {'class_id': int, 'bbox': [x_center, y_center, w, h]}
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        # Load annotation
        annotation_path = self.annotations[idx]
        target = self._load_annotation(annotation_path)

        if target is None:
            raise ValueError(f"Invalid annotation for {img_path}")

        # Resize image to target size
        image = image.resize(self.target_size, Image.BILINEAR)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor and normalize
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.tensor(image).permute(2, 0, 1)  # HWC -> CHW

        # Convert target to tensors
        target_tensor = {
            'class_id': torch.tensor(target['class_id'], dtype=torch.long),
            'bbox': torch.tensor(target['bbox'], dtype=torch.float32)
        }

        return image, target_tensor

    def get_class_names(self):
        """Return list of class names."""
        return self.class_names

    def get_num_classes(self):
        """Return number of classes."""
        return len(self.class_names)


def collate_fn(batch):
    """
    Custom collate function for detection dataloader.

    Args:
        batch: List of (image, target) tuples

    Returns:
        tuple: (images_tensor, targets_list)
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    # Stack images into batch tensor
    images = torch.stack(images, 0)

    return images, targets
