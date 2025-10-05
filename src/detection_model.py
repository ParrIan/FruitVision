"""
Fruit Object Detection Model

This module defines a CNN architecture for single-object fruit detection.
The model outputs both class predictions and bounding box coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FruitDetector(nn.Module):
    """
    CNN for single-object fruit detection.

    Architecture:
    - Shared feature extraction backbone (similar to FruitCNN)
    - Two separate heads:
      1. Classification head: predicts fruit class
      2. Bbox regression head: predicts bounding box (x_center, y_center, w, h)

    Input: RGB images of size (3, H, W) - default 640x480
    Output:
      - class_logits: (batch_size, num_classes)
      - bbox_pred: (batch_size, 4) - normalized [x_center, y_center, width, height]
    """

    def __init__(self, num_classes=219, input_size=(480, 640)):
        """
        Initialize the detection model.

        Args:
            num_classes: Number of fruit classes
            input_size: Input image size (height, width)
        """
        super(FruitDetector, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size

        # Shared feature extraction backbone
        # Conv block 1: Extract basic features
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2: Detect shape/texture
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Conv block 3: High-level patterns
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Conv block 4: Deep features for localization
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Use Global Average Pooling instead of flattening to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Shared fully connected layer (much smaller now)
        self.fc_shared = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)

        # Classification head
        self.fc_class = nn.Linear(512, 256)
        self.fc_class_out = nn.Linear(256, num_classes)

        # Bounding box regression head
        self.fc_bbox = nn.Linear(512, 128)
        self.fc_bbox_out = nn.Linear(128, 4)  # x_center, y_center, width, height

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input batch of shape (batch_size, 3, H, W)

        Returns:
            tuple: (class_logits, bbox_pred)
            - class_logits: (batch_size, num_classes)
            - bbox_pred: (batch_size, 4) [x_center, y_center, width, height] normalized
        """
        # Shared feature extraction
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # -> H/2 x W/2 x 32
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # -> H/4 x W/4 x 64
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))  # -> H/8 x W/8 x 128
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))  # -> H/16 x W/16 x 256

        # Global Average Pooling
        x = self.global_pool(x)  # -> 1 x 1 x 256
        x = x.view(x.size(0), -1)  # -> 256

        # Shared FC layer
        x = F.relu(self.fc_shared(x))
        x = self.dropout(x)

        # Classification head
        class_features = F.relu(self.fc_class(x))
        class_logits = self.fc_class_out(class_features)

        # Bounding box regression head
        bbox_features = F.relu(self.fc_bbox(x))
        bbox_pred = torch.sigmoid(self.fc_bbox_out(bbox_features))  # Sigmoid to keep in [0, 1]

        return class_logits, bbox_pred


class DetectionLoss(nn.Module):
    """
    Combined loss for object detection.

    Combines:
    - Classification loss (CrossEntropyLoss)
    - Bounding box regression loss (Smooth L1 Loss)
    """

    def __init__(self, bbox_weight=5.0):
        """
        Initialize detection loss.

        Args:
            bbox_weight: Weight for bbox loss relative to classification loss
        """
        super(DetectionLoss, self).__init__()
        self.bbox_weight = bbox_weight
        self.class_criterion = nn.CrossEntropyLoss()
        self.bbox_criterion = nn.SmoothL1Loss()

    def forward(self, class_logits, bbox_pred, targets):
        """
        Compute detection loss.

        Args:
            class_logits: (batch_size, num_classes) predicted class logits
            bbox_pred: (batch_size, 4) predicted bboxes [x_center, y_center, w, h]
            targets: List of dicts with 'class_id' and 'bbox'

        Returns:
            tuple: (total_loss, class_loss, bbox_loss)
        """
        batch_size = len(targets)

        # Extract ground truth class labels and bboxes
        class_labels = torch.stack([t['class_id'] for t in targets]).to(class_logits.device)
        bbox_targets = torch.stack([t['bbox'] for t in targets]).to(bbox_pred.device)

        # Classification loss
        class_loss = self.class_criterion(class_logits, class_labels)

        # Bounding box regression loss
        bbox_loss = self.bbox_criterion(bbox_pred, bbox_targets)

        # Combined loss
        total_loss = class_loss + self.bbox_weight * bbox_loss

        return total_loss, class_loss, bbox_loss


def calculate_iou(pred_boxes, target_boxes):
    """
    Calculate Intersection over Union (IoU) for bounding boxes.

    Args:
        pred_boxes: (N, 4) predicted boxes [x_center, y_center, w, h] normalized
        target_boxes: (N, 4) target boxes [x_center, y_center, w, h] normalized

    Returns:
        iou: (N,) IoU scores
    """
    # Convert center format to corner format
    def center_to_corners(boxes):
        x_center, y_center, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    pred_corners = center_to_corners(pred_boxes)
    target_corners = center_to_corners(target_boxes)

    # Calculate intersection
    x1_inter = torch.max(pred_corners[:, 0], target_corners[:, 0])
    y1_inter = torch.max(pred_corners[:, 1], target_corners[:, 1])
    x2_inter = torch.min(pred_corners[:, 2], target_corners[:, 2])
    y2_inter = torch.min(pred_corners[:, 3], target_corners[:, 3])

    inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

    # Calculate union
    pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
    target_area = target_boxes[:, 2] * target_boxes[:, 3]
    union_area = pred_area + target_area - inter_area

    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)

    return iou
