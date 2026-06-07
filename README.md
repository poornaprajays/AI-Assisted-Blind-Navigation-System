# AI-Assisted Blind Navigation System

VisionMate AI is a high-performance, real-time assistive navigation system designed to empower visually impaired individuals. It translates raw spatial computer vision detections into structured, filtered, and non-intrusive safety alerts.

## Project Architecture

The pipeline consists of the following core modules:

1. **Camera Stream (`src/camera.py`):** Dedicated, thread-safe camera capture utilizing a lag-free, double-buffered background thread to prevent queue latency.
2. **Detection Engine (`src/detection.py`):** Real-time YOLOv8 object detector optimized for fast inference (supporting physical cameras and synthetic mock frames).
3. **Spatial Context Engine (`src/context_engine.py`):** Spatial intelligence filter that prioritizes imminent obstacles (e.g., stairs, vehicles) and employs cooldowns to prevent cognitive overload.
4. **Data Contracts (`src/schemas/contracts.py`):** Optimized dataclasses using Python `__slots__` to minimize garbage collection churn and latency.

---

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```

### Running the Tests
To verify the engine rules, metrics, and cooldown logic:
```bash
python -m pytest tests/test_context_engine.py -v
```

### Running the Simulation
To run the local end-to-end simulation pipeline:
```bash
python -m src.context_engine
```
