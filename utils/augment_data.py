import os
import random
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
import argparse


def remove_white_background(image, threshold=240):
    """
    Remove white background from fruit images and make transparent.

    Args:
        image: PIL Image
        threshold: Pixel value threshold for white (0-255)

    Returns:
        PIL Image with transparent background
    """
    image = image.convert("RGBA")
    data = np.array(image)

    # Get RGB channels
    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    # Create mask for white pixels (all RGB channels above threshold)
    white_mask = (r > threshold) & (g > threshold) & (b > threshold)

    # Set alpha to 0 for white pixels (transparent)
    data[white_mask, 3] = 0

    return Image.fromarray(data)


def get_bounding_box(image):
    """
    Get bounding box of non-transparent pixels.

    Args:
        image: PIL Image with alpha channel

    Returns:
        Tuple of (x, y, width, height) or None if image is fully transparent
    """
    alpha = np.array(image.getchannel('A'))

    # Find non-transparent pixels
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)


def augment_fruit_image(fruit_img, background_img,
                        scale_range=(.3, 1.2),
                        rotation_range=(-15, 15),
                        brightness_range=(0.9, 1.1),
                        contrast_range=(0.9, 1.1)):
    """
    Paste fruit onto background with random augmentations.

    Args:
        fruit_img: PIL Image of fruit (with transparent background)
        background_img: PIL Image of background
        scale_range: Tuple of (min, max) scale factors
        rotation_range: Tuple of (min, max) rotation degrees
        brightness_range: Tuple of (min, max) brightness factors
        contrast_range: Tuple of (min, max) contrast factors

    Returns:
        Tuple of (augmented_image, yolo_bbox)
        yolo_bbox format: (x_center, y_center, width, height) normalized to 0-1
    """
    bg_width, bg_height = background_img.size

    # Random rotation
    rotation = random.uniform(*rotation_range)
    fruit_rotated = fruit_img.rotate(rotation, expand=True, resample=Image.BICUBIC)

    # Random scale
    scale = random.uniform(*scale_range)
    fruit_width, fruit_height = fruit_rotated.size
    new_width = int(fruit_width * scale)
    new_height = int(fruit_height * scale)
    fruit_scaled = fruit_rotated.resize((new_width, new_height), Image.LANCZOS)

    # Random brightness and contrast adjustments
    brightness_factor = random.uniform(*brightness_range)
    contrast_factor = random.uniform(*contrast_range)

    enhancer = ImageEnhance.Brightness(fruit_scaled)
    fruit_enhanced = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(fruit_enhanced)
    fruit_enhanced = enhancer.enhance(contrast_factor)

    # Random position (ensure fruit stays within bounds)
    max_x = bg_width - new_width
    max_y = bg_height - new_height

    if max_x <= 0 or max_y <= 0:
        # Fruit is too large, scale it down
        scale_down = min(bg_width / new_width, bg_height / new_height) * 0.8
        new_width = int(new_width * scale_down)
        new_height = int(new_height * scale_down)
        fruit_enhanced = fruit_enhanced.resize((new_width, new_height), Image.LANCZOS)
        max_x = bg_width - new_width
        max_y = bg_height - new_height

    pos_x = random.randint(0, max(0, max_x))
    pos_y = random.randint(0, max(0, max_y))

    # Paste fruit onto background
    result = background_img.convert("RGB").copy()
    result.paste(fruit_enhanced, (pos_x, pos_y), fruit_enhanced)

    # Calculate YOLO format bounding box (normalized)
    x_center = (pos_x + new_width / 2) / bg_width
    y_center = (pos_y + new_height / 2) / bg_height
    width = new_width / bg_width
    height = new_height / bg_height

    return result, (x_center, y_center, width, height)


def generate_augmented_dataset(fruit_dir, background_dir, output_dir,
                               class_names=None, max_backgrounds=None):
    """
    Generate augmented dataset by pasting fruits on backgrounds.

    Args:
        fruit_dir: Directory containing fruit class subdirectories
        background_dir: Directory containing background images
        output_dir: Output directory for augmented dataset
        class_names: List of class names (if None, use all subdirectories)
        max_backgrounds: Max number of backgrounds to use (None = use all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load background images
    background_files = [
        os.path.join(background_dir, f)
        for f in os.listdir(background_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    if max_backgrounds:
        background_files = background_files[:max_backgrounds]

    if not background_files:
        print(f"Error: No background images found in {background_dir}")
        return

    print(f"Loaded {len(background_files)} background images")

    # Get class names
    if class_names is None:
        class_names = sorted([
            d for d in os.listdir(fruit_dir)
            if os.path.isdir(os.path.join(fruit_dir, d)) and not d.startswith('.')
        ])

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"Found {len(class_names)} fruit classes")
    print(f"Generating augmented dataset in {output_dir}")
    print("=" * 60)

    total_generated = 0

    # Process each class
    for class_name in class_names:
        class_dir = os.path.join(fruit_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        class_idx = class_to_idx[class_name]

        # Create class subdirectories
        class_images_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_images_dir, exist_ok=True)

        # Get all fruit images for this class
        fruit_files = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        print(f"\n{class_name} (class {class_idx}): {len(fruit_files)} images")

        # Process each fruit image
        for idx, fruit_path in enumerate(fruit_files):
            try:
                # Load fruit image and remove white background
                fruit_img = Image.open(fruit_path).convert("RGB")
                fruit_transparent = remove_white_background(fruit_img)

                # Select random background
                bg_path = random.choice(background_files)
                bg_img = Image.open(bg_path).convert("RGB")

                # Generate augmented image
                aug_img, bbox = augment_fruit_image(fruit_transparent, bg_img)

                # Get original filename and save augmented image in class directory
                original_filename = os.path.basename(fruit_path)
                img_path = os.path.join(class_images_dir, original_filename)
                aug_img.save(img_path, quality=95)

                # Save YOLO format label with same name
                label_filename = os.path.splitext(original_filename)[0] + '.txt'
                label_path = os.path.join(class_images_dir, label_filename)

                with open(label_path, 'w') as f:
                    # YOLO format: class x_center y_center width height
                    f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

                total_generated += 1

                if (idx + 1) % 50 == 0:
                    print(f"  Processed {idx + 1}/{len(fruit_files)} images...")

            except Exception as e:
                print(f"  Error processing {fruit_path}: {e}")
                continue

        print(f"  Completed {class_name}")

    print("\n" + "=" * 60)
    print(f"Dataset generation complete!")
    print(f"Total images generated: {total_generated}")
    print(f"Dataset saved to: {output_dir}")
    print("=" * 60)

    # Save class names file
    classes_file = os.path.join(output_dir, 'classes.txt')
    with open(classes_file, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")

    print(f"Class names saved to: {classes_file}")


if __name__ == "__main__":
    # Hardcoded paths that mirror fruits-360 structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fruit_base_dir = os.path.join(base_dir, 'data', 'fruits-360')
    background_dir = os.path.join(base_dir, 'data', 'backgrounds')
    output_base_dir = os.path.join(base_dir, 'data', 'augmented_data')

    print("Fruit Object Detection Dataset Augmentation")
    print("=" * 60)
    print(f"Source: {fruit_base_dir}")
    print(f"Backgrounds: {background_dir}")
    print(f"Output: {output_base_dir}")
    print("=" * 60)
    print()

    # Generate training set
    print("Generating TRAINING dataset...")
    print("-" * 60)
    train_fruit_dir = os.path.join(fruit_base_dir, 'Training')
    train_output_dir = os.path.join(output_base_dir, 'Training')

    if os.path.exists(train_fruit_dir):
        generate_augmented_dataset(
            train_fruit_dir,
            background_dir,
            train_output_dir
        )
    else:
        print(f"Warning: Training directory not found: {train_fruit_dir}")

    print("\n" + "=" * 60 + "\n")

    # Generate test set
    print("Generating TEST dataset...")
    print("-" * 60)
    test_fruit_dir = os.path.join(fruit_base_dir, 'Test')
    test_output_dir = os.path.join(output_base_dir, 'Test')

    if os.path.exists(test_fruit_dir):
        generate_augmented_dataset(
            test_fruit_dir,
            background_dir,
            test_output_dir
        )
    else:
        print(f"Warning: Test directory not found: {test_fruit_dir}")

    print("\n" + "=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)
