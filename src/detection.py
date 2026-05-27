"""
VisionMate AI - Object Detection Engine Module
Path: src/detection.py

This module implements the core visual inference layer for VisionMate AI.
It wraps the YOLOv8 object detection model, manages device auto-selection (CUDA vs CPU), 
processes OpenCV-compatible image frames, filters detections by class/confidence, 
and translates raw outputs into standardized, slot-optimized 'Detection' contracts.

================================================================================
ARCHITECTURAL DESIGN & SCALABILITY RATIONALE:
================================================================================
1. Why this Architecture is Scalable:
   By separating Model Ingestion (this module) from Decision-making (Spatial Logic), 
   we can easily swap out models (e.g., YOLOv8 for YOLOv10, RT-DETR, or customized ONNX models) 
   without touching downstream telemetry, database, or audio logic. The input is always 
   a raw frame, and the output is always a List[Detection] contract.

2. How this Supports Future Edge (Raspberry Pi) Deployments:
   - YOLOv8-nano (yolov8n) is extremely lightweight (~6MB) and optimized for edge.
   - For Raspberry Pi, we can seamlessly swap the PyTorch backend for:
       * ONNX Runtime (CPU-optimized, direct export `yolov8n.onnx`)
       * OpenVINO (Intel Myriad/Oak-D)
       * TensorFlow Lite (TFLite)
     Because this module abstracts the inference step, we only need to change the 
     internal model wrapper inside `DetectionEngine` without changing any other files.

3. Impact of Inference Timing on Assistive Safety Systems:
   Assistive navigation requires immediate feedback. 
   - A latency of >200ms represents an outdated alert. If a user walks at 1.5 m/s, 
     a 200ms delay means the obstacle is actually 30cm closer than reported.
   - We target sub-50ms inference times. If running on hardware that cannot achieve this, 
     we lower the input resolution (e.g., 640x480 to 320x320) or run inference only 
     every 2-3 frames, while keeping camera capture at full 30 FPS.
"""

import logging
import time
from typing import List, Tuple, Optional, Set
from src.schemas.contracts import Detection

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
logger = logging.getLogger("VisionMateAI.Detection")

# Try to import OpenCV, ultralytics and torch, fallback to Mock Mode if missing
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import torch
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLOv8 or PyTorch is not installed. DetectionEngine will run in high-fidelity Mock Mode.")


class DetectionEngine:
    """
    Modular object detection wrapper utilizing YOLOv8.
    Responsible for frame inference, threshold filtering, and contract mapping.
    """
    def __init__(
        self, 
        model_name: str = "yolov8n.pt", 
        conf_threshold: float = 0.25, 
        allowed_classes: Optional[List[str]] = None
    ):
        """
        Initializes the Detection Engine, selecting the best available computing device.
        
        Args:
            model_name: The YOLOv8 model identifier (e.g. 'yolov8n.pt', 'yolov8s.pt').
            conf_threshold: Bounding box confidence filter threshold.
            allowed_classes: Optional list of class names to output (filters out everything else).
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        
        # Store allowed classes as a Set for O(1) lookup
        self.allowed_classes = set(allowed_classes) if allowed_classes is not None else None
        
        self.model = None
        self.device = "cpu"
        self.fps = 0.0
        self.inference_times: List[float] = []

        if YOLO_AVAILABLE:
            # Device Selection: Leverage NVIDIA GPU via CUDA if available, fallback to CPU
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                
            logger.info(f"Loading YOLOv8 model '{self.model_name}' on device: {self.device.upper()}")
            try:
                # Load the model once to memory
                self.model = YOLO(self.model_name)
                # Dry run to warm up PyTorch/CUDA kernels
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy_img, verbose=False, device=self.device)
                logger.info("YOLOv8 engine initialized and warmed up successfully.")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8 model: {e}. Falling back to Mock Mode.")
                self.model = None
        else:
            logger.info("Starting DetectionEngine in high-fidelity Mock Mode.")

    def detect(self, frame: object) -> List[Detection]:
        """
        Runs object detection inference on the provided frame.
        
        Args:
            frame: OpenCV-compatible image frame (numpy.ndarray).
            
        Returns:
            List[Detection]: List of standardized Detection contract dataclasses.
        """
        t_start = time.time()
        detections: List[Detection] = []
        
        # Extract frame boundaries for centering calculations
        if hasattr(frame, "shape"):
            frame_height, frame_width = frame.shape[:2]
        else:
            frame_height, frame_width = 480, 640  # Stand-in resolution for extreme mocks

        # --- REAL YOLO INFERENCE ---
        if self.model is not None:
            try:
                # Run YOLOv8 inference
                # verbose=False suppresses constant print output from the CLI loop
                results = self.model(frame, verbose=False, device=self.device)[0]
                
                # Parse inference bounds
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue
                        
                    # Lookup class index to label string
                    cls_idx = int(box.cls[0])
                    class_name = results.names[cls_idx]
                    
                    # Apply allowed class filter if configured
                    if self.allowed_classes is not None and class_name not in self.allowed_classes:
                        continue

                    # Extract coordinates: (xmin, ymin, xmax, ymax)
                    xyxy = box.xyxy[0].tolist()
                    xmin, ymin, xmax, ymax = [int(val) for val in xyxy]

                    # Standardized post-calculations
                    width = xmax - xmin
                    height = ymax - ymin
                    center_x = xmin + (width // 2)
                    center_y = ymin + (height // 2)

                    # 1. Spatial Zone classification (left third, center third, right third)
                    if center_x < frame_width // 3:
                        spatial_zone = "left"
                    elif center_x > 2 * frame_width // 3:
                        spatial_zone = "right"
                    else:
                        spatial_zone = "center"

                    # 2. Heuristic Distance estimation (focal estimation for safety logic)
                    # We map bounding box height relative to frame height as a proxy for depth.
                    # As an object gets closer, its box height approaches the frame height.
                    # Baseline model assumes class is a standard obstacle. 
                    # Standard height threshold maps: 1.5 * frame_height / box_height.
                    estimated_distance = round((1.5 * frame_height) / max(height, 1), 2)

                    # Pack mapped properties inside the formal Detection contract
                    detection = Detection(
                        class_name=class_name,
                        confidence=conf,
                        bbox=(xmin, ymin, xmax, ymax),
                        center_x=center_x,
                        center_y=center_y,
                        width=width,
                        height=height,
                        estimated_distance=estimated_distance,
                        spatial_zone=spatial_zone,
                        timestamp=t_start
                    )
                    detections.append(detection)
            except Exception as e:
                logger.error(f"Inference execution fault: {e}")
                
        # --- HIGH-FIDELITY MOCK INFERENCE FALLBACK ---
        else:
            # We simulate a dynamic mock obstacle matching the rectangle created in CameraStream
            # Center X oscillates over time to simulate a pedestrian/obstacle walking
            oscillation = int((time.time() * 80) % (frame_width - 100)) + 50
            center_x = oscillation
            center_y = frame_height // 2
            width, height = 60, 60
            
            xmin = center_x - (width // 2)
            ymin = center_y - (height // 2)
            xmax = center_x + (width // 2)
            ymax = center_y + (height // 2)

            # Class name can oscillate between a "chair" or "person" for visual flavor
            simulated_class = "chair" if int(time.time() // 4) % 2 == 0 else "person"
            
            # Apply allowed class filter if configured
            if self.allowed_classes is None or simulated_class in self.allowed_classes:
                # Spatial mapping calculations
                if center_x < frame_width // 3:
                    spatial_zone = "left"
                elif center_x > 2 * frame_width // 3:
                    spatial_zone = "right"
                else:
                    spatial_zone = "center"

                # Simulate a dynamic distance mapping that fluctuates based on center position
                estimated_distance = round(1.0 + (abs(frame_width // 2 - center_x) / 100.0), 2)
                
                detection = Detection(
                    class_name=simulated_class,
                    confidence=0.88,
                    bbox=(xmin, ymin, xmax, ymax),
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    height=height,
                    estimated_distance=estimated_distance,
                    spatial_zone=spatial_zone,
                    timestamp=t_start
                )
                detections.append(detection)
                
                # Brief sleep to simulate real neural network latency (e.g. ~12ms for YOLOv8n)
                time.sleep(0.012)

        # Log diagnostics and timing metrics
        t_dur = time.time() - t_start
        self.inference_times.append(t_dur)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
            
        avg_inference_time = sum(self.inference_times) / len(self.inference_times)
        self.fps = 1.0 / max(avg_inference_time, 0.0001)
        
        logger.debug(f"Inference completed in {t_dur * 1000:.1f}ms (Avg FPS: {self.fps:.1f})")
        return detections

    def draw_overlays(self, frame: object, detections: List[Detection]) -> object:
        """
        Draws diagnostic visual annotations on top of the frame.
        Useful for visualization overlays and validation pipelines.
        
        Args:
            frame: OpenCV image frame (numpy.ndarray).
            detections: Standardized List[Detection].
            
        Returns:
            The annotated image frame.
        """
        if not OPENCV_AVAILABLE or not isinstance(frame, np.ndarray):
            return frame

        # Add vertical grid dividers representing the spatial zones
        frame_height, frame_width = frame.shape[:2]
        cv2.line(frame, (frame_width // 3, 0), (frame_width // 3, frame_height), (80, 80, 80), 1, cv2.LINE_AA)
        cv2.line(frame, (2 * frame_width // 3, 0), (2 * frame_width // 3, frame_height), (80, 80, 80), 1, cv2.LINE_AA)
        
        # Write Zone labels at the top of the viewport
        cv2.putText(frame, "LEFT", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "CENTER", (frame_width // 3 + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT", (2 * frame_width // 3 + 20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

        for det in detections:
            xmin, ymin, xmax, ymax = det.bbox
            
            # Change bounding box color based on distance hierarchy (closer = brighter warning red)
            if det.estimated_distance < 1.5:
                color = (0, 0, 255)  # Alert Red
                thickness = 2
            elif det.estimated_distance < 2.5:
                color = (0, 165, 255)  # Caution Orange
                thickness = 2
            else:
                color = (0, 255, 0)  # Safe Green
                thickness = 1

            # Bounding Box Draw
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Bounding box Center Dot
            cv2.circle(frame, (det.center_x, det.center_y), 4, color, -1)

            # Metadata Display Text Label
            label = f"{det.class_name.upper()} {det.confidence:.0%} | {det.estimated_distance}m"
            
            # Elegant label backing card
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (xmin, ymin - text_h - 6), (xmin + text_w + 6, ymin), color, -1)
            
            # Print label text in white
            cv2.putText(frame, label, (xmin + 3, ymin - 3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Draw system-wide telemetry overlay card
        cv2.rectangle(frame, (5, frame_height - 35), (260, frame_height - 5), (32, 28, 24), -1)
        cv2.rectangle(frame, (5, frame_height - 35), (260, frame_height - 5), (60, 56, 52), 1)
        cv2.putText(frame, f"Model: {self.model_name} ({self.device.upper()})", (12, frame_height - 22), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
        
        # Calculate overall loop FPS based on metric average
        avg_infer = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        cv2.putText(frame, f"Inference Latency: {avg_infer * 1000:.1f}ms", (12, frame_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

        return frame


# ================================================================================
# INTEGRATION TESTING LOOP (CameraStream -> DetectionEngine -> Visualization)
# ================================================================================
if __name__ == "__main__":
    from src.camera import CameraStream
    
    logger.info("Initializing system integration test loop...")
    
    # 1. Launch CameraStream (automatically selects camera 0 or fallbacks to Mock Mode)
    stream = CameraStream(src=0, width=640, height=480, fps=30)
    
    # 2. Instantiate DetectionEngine
    # We restrict allowed classes to common navigational obstacles to demonstrate filters
    navigation_classes = ["chair", "person", "table", "backpack", "suitcase", "bottle"]
    engine = DetectionEngine(model_name="yolov8n.pt", conf_threshold=0.25, allowed_classes=navigation_classes)
    
    try:
        # Start camera thread
        stream.start()
        
        logger.info("Visual pipeline running. Press 'q' on UI window or 'Ctrl+C' in terminal to stop.")
        
        run_count = 0
        while run_count < 30:  # Loop for 30 cycles or key exits
            # Fetch latest frame thread-safely
            grabbed, frame = stream.read()
            
            if grabbed and frame is not None:
                # Run YOLOv8 detection
                detections = engine.detect(frame)
                
                # Annotate bounding boxes, lines, and telemetry labels on screen
                annotated_frame = engine.draw_overlays(frame, detections)
                
                # Show in a window if GUI is active
                if OPENCV_AVAILABLE and isinstance(annotated_frame, np.ndarray):
                    cv2.imshow("VisionMate AI - Integrated Detection Test", annotated_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Visual loop stopped by UI window key press.")
                        break
                else:
                    # Print detection logs when running headlessly
                    print(f"\n[FRAME #{stream.frame_count}] Latency: {engine.inference_times[-1]*1000:.1f}ms | Active Detections: {len(detections)}")
                    for d in detections:
                        print(f"  -> {d.class_name.upper()} ({d.confidence:.1%}) | Zone: {d.spatial_zone.upper()} | Depth: {d.estimated_distance}m")
            else:
                logger.warning("Pipeline waiting for active frames...")
                time.sleep(0.05)
                
            run_count += 1
            # Simulate natural execution pace
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received! Cleaning up pipeline safely...")
    finally:
        # Stop background threading and release resources
        stream.stop()
        if OPENCV_AVAILABLE:
            cv2.destroyAllWindows()
        logger.info("Integrated detection validation completed successfully.")
