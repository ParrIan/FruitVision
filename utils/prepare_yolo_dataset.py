"""
Prepare YOLO dataset from augmented data.

YOLOv8 expects:
- dataset/
  - images/
    - train/
    - val/
  - labels/
    - train/
    - val/
"""

import os
import shutil
from pathlib import Path


def prepare_yolo_dataset(augmented_dir='../data/augmented',
                         output_dir='../data/yolo_dataset'):
    """
    Reorganize augmented data into YOLOv8 format.

    Args:
        augmented_dir: Directory with Training/Test subdirectories
        output_dir: Output directory for YOLO format
    """
    augmented_path = Path(augmented_dir)
    output_path = Path(output_dir)

    # Create YOLO directory structure
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

    print("Preparing YOLO dataset...")
    print(f"Source: {augmented_path}")
    print(f"Output: {output_path}\n")

    # Process training and validation splits
    splits = {
        'train': augmented_path / 'Training',
        'val': augmented_path / 'Test'
    }

    for split_name, split_dir in splits.items():
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found, skipping...")
            continue

        print(f"Processing {split_name} split...")

        images_out = output_path / 'images' / split_name
        labels_out = output_path / 'labels' / split_name

        total_images = 0
        total_labels = 0

        # Iterate through all class directories
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or class_dir.name.startswith('.'):
                continue

            print(f"  {class_dir.name}...", end=' ')

            class_images = 0
            class_labels = 0

            # Copy all images and labels
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Copy image
                    new_name = f"{class_dir.name}_{file_path.name}"
                    shutil.copy2(file_path, images_out / new_name)
                    class_images += 1

                    # Copy corresponding label
                    label_path = file_path.with_suffix('.txt')
                    if label_path.exists():
                        shutil.copy2(label_path, labels_out / new_name.replace(file_path.suffix, '.txt'))
                        class_labels += 1

            print(f"{class_images} images, {class_labels} labels")
            total_images += class_images
            total_labels += class_labels

        print(f"  Total: {total_images} images, {total_labels} labels\n")

    print("="*60)
    print("Dataset preparation complete!")
    print(f"Dataset location: {output_path}")
    print("="*60)


if __name__ == "__main__":
    prepare_yolo_dataset()
