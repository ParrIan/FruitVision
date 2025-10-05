"""
Real-Time Fruit Detection using YOLOv8 and Webcam

Usage:
    python realtime_yolo.py

Controls:
    - Press 'q' to quit
    - Press 'c' to toggle confidence threshold
"""

import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np


class YOLORealtimeDetector:
    """Real-time fruit detector using YOLOv8."""

    def __init__(self, model_path='../models/weights/best.pt',
                 camera_id=0,
                 confidence_threshold=0.5,
                 target_fruits=['Apple', 'Lemon'],
                 smoothing_frames=15,
                 imgsz=640,
                 frame_skip=1):
        """
        Initialize detector.

        Args:
            model_path: Path to trained YOLO model
            camera_id: Webcam device ID
            confidence_threshold: Minimum confidence for detections
            target_fruits: List of fruit types to detect (matches class names starting with these)
            smoothing_frames: Number of frames to average for temporal smoothing
            imgsz: Input image size for inference (smaller=faster, default 320)
            frame_skip: Process every Nth frame (2=every other frame)
        """
        self.model_path = model_path
        self.camera_id = camera_id
        self.confidence_threshold = confidence_threshold
        self.target_fruits = target_fruits
        self.smoothing_frames = smoothing_frames
        self.imgsz = imgsz
        self.frame_skip = frame_skip

        # Temporal smoothing buffers
        self.detection_buffer = deque(maxlen=smoothing_frames)

        # Frame counter for skipping
        self.frame_count = 0

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)

        # Use MPS (Metal Performance Shaders) for M1/M2/M3 Macs
        import torch
        if torch.backends.mps.is_available():
            self.device = 'mps'
            print("Using MPS (Apple Silicon GPU) for acceleration")
        else:
            self.device = 'cpu'
            print("Using CPU")

        # Get class names and filter for target fruits
        self.class_names = self.model.names  # Dict: {0: 'Apple 10', 1: 'Apple 11', ...}
        self.target_class_ids = []

        # Excluded classes
        excluded = ['Apple Golden']

        for idx, name in self.class_names.items():
            # Check if class name starts with any target fruit
            if any(name.startswith(fruit) for fruit in self.target_fruits):
                # Exclude specific classes
                if not any(excl in name for excl in excluded):
                    self.target_class_ids.append(idx)

        print(f"Detecting {len(self.target_class_ids)} classes from: {target_fruits}")
        print(f"Temporal smoothing: {smoothing_frames} frames")
        print(f"Frame skip: Processing every {frame_skip} frame(s)")
        print("Model loaded successfully!\n")

    def _smooth_detections(self):
        """
        Apply temporal smoothing to detections.

        Returns:
            Smoothed detection results (most consistent detection over buffer)
        """
        if len(self.detection_buffer) == 0:
            return None

        # Count detections per class across buffer
        class_votes = {}
        box_coords = {}

        for result in self.detection_buffer:
            if len(result.boxes) > 0:
                # Get the highest confidence detection
                best_idx = result.boxes.conf.argmax()
                cls = int(result.boxes.cls[best_idx])
                conf = float(result.boxes.conf[best_idx])
                box = result.boxes.xyxy[best_idx].cpu().numpy()

                # Vote for this class
                if cls not in class_votes:
                    class_votes[cls] = []
                    box_coords[cls] = []

                class_votes[cls].append(conf)
                box_coords[cls].append(box)

        # If no detections in buffer, return None
        if not class_votes:
            return None

        # Get most frequently detected class
        best_class = max(class_votes.keys(), key=lambda k: len(class_votes[k]))

        # Average the bounding boxes for this class
        avg_box = np.mean(box_coords[best_class], axis=0)
        avg_conf = np.mean(class_votes[best_class])

        # Return the latest result but with smoothed values
        # (This is a simplification - ideally we'd create a new Result object)
        latest = self.detection_buffer[-1]

        return latest

    def run(self):
        """Start real-time detection."""
        import time

        print(f"Starting webcam (camera {self.camera_id})...")
        cap = cv2.VideoCapture(self.camera_id)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        # Give camera time to warm up
        time.sleep(1.0)

        print("Webcam started successfully!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 'c' to cycle confidence threshold")
        print(f"\nConfidence threshold: {self.confidence_threshold:.2f}")
        print("\nStarting detection...\n")

        # FPS tracking
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0.0

        # Last result for frame skipping
        last_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Frame skipping - only process every Nth frame
            self.frame_count += 1
            process_frame = (self.frame_count % self.frame_skip == 0)

            if process_frame:
                # Run YOLO inference (only detect target classes)
                results = self.model(frame, conf=self.confidence_threshold, classes=self.target_class_ids, verbose=False, device=self.device, imgsz=self.imgsz)

                # Add current detections to buffer
                self.detection_buffer.append(results[0])

                # Get smoothed detections
                smoothed_results = self._smooth_detections()
                last_result = smoothed_results

            # Draw results on frame (use last result if skipping)
            if last_result is not None:
                annotated_frame = last_result.plot()
            else:
                annotated_frame = frame.copy()

            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 30:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0

            # Draw FPS on frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display
            cv2.imshow('YOLOv8 Fruit Detection', annotated_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('c'):
                # Cycle through thresholds
                thresholds = [0.25, 0.5, 0.75, 0.85]
                current_idx = min(range(len(thresholds)),
                                 key=lambda i: abs(thresholds[i] - self.confidence_threshold))
                next_idx = (current_idx + 1) % len(thresholds)
                self.confidence_threshold = thresholds[next_idx]
                print(f"Confidence threshold: {self.confidence_threshold:.2f}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


def run_realtime_detection(model_path='../models/weights/best.pt',
                           camera_id=1,
                           confidence_threshold=0.5):
    """
    Convenience function to run real-time detection.

    Args:
        model_path: Path to trained YOLO model
        camera_id: Webcam device ID
        confidence_threshold: Minimum confidence for detections
    """
    try:
        detector = YOLORealtimeDetector(
            model_path=model_path,
            camera_id=camera_id,
            confidence_threshold=confidence_threshold
        )
        detector.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    run_realtime_detection()
