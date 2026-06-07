"""
VisionMate AI - Spatial Context & Assistive Reasoning Engine
Path: src/context_engine.py

This module implements the core "intelligence" layer of the assistive navigation system.
A raw list of object detections is insufficient for a visually impaired user. 
Simply announcing every object in the field of view (e.g., "chair, table, cup, person, bottle") 
leads to extreme cognitive overload, making navigation confusing and highly dangerous.

This engine functions as a context-aware reasoning filter. It evaluates bounding boxes, 
depth distances, and horizontal spatial zones, prioritizing obstacles that pose an 
immediate hazard while dynamically throttling repetitive alerts.

================================================================================
CRITICAL NAVIGATIONAL DESIGN DECISIONS:
================================================================================
1. Why Assistive AI Requires Intelligent Filtering:
   Visual environments are dense. A blind user walking through a room only cares about 
   active barriers in their immediate walking vector. Announcing a clock on the wall 
   or a chair 4 meters away distracts from a wet floor sign directly ahead. The engine 
   prunes ~90% of raw neural network outputs to focus entirely on immediate safety.

2. Why Constant Announcements are Dangerous (Cognitive Overload):
   Human hearing is highly spatial and transient. Constant text-to-speech feedback blocks 
   environmental audio cues (like ambient echoes, oncoming traffic noise, or walking canes). 
   By implementing adaptive alert throttling (cooldowns), the system remains quiet 
   unless a novel safety threat is detected.

3. Future-Ready Architecture (Temporal and Spatial Tracking):
   The processing interface takes `List[Detection]` and returns `List[Alert]`. This 
   decoupling allows future tracking algorithms (like Kalman Filters or ByteTrack) to be 
   inserted directly before or inside the context engine to track moving hazards (like 
   oncoming pedestrians or vehicles) over multiple frames.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import time
import uuid
from typing import List, Dict, Tuple, Optional

from src.schemas.contracts import Detection

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
logger = logging.getLogger("VisionMateAI.ContextEngine")


class AlertPriority(str, Enum):
    """
    Emergency priority hierarchy determining visual overlay warnings and speech queues.
    """
    LOW = "low"            # General background objects (informative only)
    MEDIUM = "medium"      # Nearby objects outside direct path
    HIGH = "high"          # Dynamic hazards or close objects in walking vector
    CRITICAL = "critical"  # Immediate safety stop required (e.g. obstacle center close)


class AlertCategory(str, Enum):
    """
    Standardized categorization schema for navigational alerts.
    """
    OBSTACLE = "obstacle"     # Static structures (chairs, tables, walls)
    VEHICLE = "vehicle"       # Fast dynamic threats (cars, bicycles, buses)
    STAIRCASE = "staircase"   # High falling hazard terrain changes
    PERSON = "person"         # Dynamic moving pedestrians
    NAVIGATION = "navigation" # System state prompts or destination cues
    EMERGENCY = "emergency"   # Critical crash warning or system failure


@dataclass(slots=True)
class Alert:
    """
    Standardized safety alert contract dispatched to the TTS voice engine or mobile UI.
    """
    alert_id: str
    category: AlertCategory
    priority: AlertPriority
    message: str
    source_detection: Optional[Detection]
    timestamp: float

    def to_dict(self) -> dict:
        """
        Converts the Alert to a standard dictionary.
        """
        return asdict(self)


class ContextEngine:
    """
    Spatial reasoning system evaluating environmental threats, classifying zones,
    and suppressing repetitive notifications using dynamic class/zone throttling.
    """
    def __init__(
        self,
        danger_distance_m: float = 1.5,
        caution_distance_m: float = 2.5,
        default_cooldown_s: float = 3.5
    ):
        """
        Initializes the Context Engine.
        
        Args:
            danger_distance_m: Depth threshold in meters representing an immediate hazard.
            caution_distance_m: Depth threshold representing items to watch out for.
            default_cooldown_s: Time in seconds before repeating an alert of the same class/zone.
        """
        self.danger_distance_m = danger_distance_m
        self.caution_distance_m = caution_distance_m
        self.default_cooldown_s = default_cooldown_s

        # Cooldown database mapping key (class_name, spatial_zone) -> expiration epoch timestamp
        self.cooldown_tracker: Dict[Tuple[str, str], float] = {}

        # Class maps to categorize typical object detections
        self.vehicle_classes = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
        self.staircase_classes = {"stairs", "escalator", "step", "stairway"}
        self.person_classes = {"person", "pedestrian"}

        # Metrics trackers
        self.alerts_generated = 0
        self.detections_processed = 0
        self.suppressed_alerts_count = 0

    def process(self, detections: List[Detection]) -> List[Alert]:
        """
        Analyzes a frame's detections, applies spatial filtering, priority logic,
        and cooldown suppression, returning high-priority navigation alerts.
        
        Args:
            detections: List of Detection dataclasses output by the inference engine.
            
        Returns:
            List[Alert]: Filtered and prioritized safety Alerts.
        """
        t_now = time.time()
        active_alerts: List[Alert] = []
        self.detections_processed += len(detections)

        # Sort detections by distance (nearest first) to guarantee nearest threats are evaluated first
        sorted_detections = sorted(detections, key=lambda d: d.estimated_distance)

        for det in sorted_detections:
            # 1. Distant Object Pruning (Ignore distant objects to save cognitive load)
            if det.estimated_distance > 3.5:
                # Distant background details are entirely ignored in real-time navigation
                continue

            # Generate a unique key mapping to suppress duplicate alerts in the same horizontal sector
            cooldown_key = (det.class_name, det.spatial_zone)
            
            # Check if this warning is currently throttled in the cooldown tracker
            if cooldown_key in self.cooldown_tracker:
                if t_now < self.cooldown_tracker[cooldown_key]:
                    self.suppressed_alerts_count += 1
                    continue  # Skip generating duplicate alert (Suppression active!)

            # 2. Determine Alert Category
            if det.class_name in self.vehicle_classes:
                category = AlertCategory.VEHICLE
            elif det.class_name in self.staircase_classes:
                category = AlertCategory.STAIRCASE
            elif det.class_name in self.person_classes:
                category = AlertCategory.PERSON
            else:
                category = AlertCategory.OBSTACLE

            # 3. CORE REASONING ENGINE (Priority Assignment & Filter Rules)
            priority: Optional[AlertPriority] = None
            message = ""

            # Rule A: Emergency Vehicles / Moving Dynamic Threats Close
            if category == AlertCategory.VEHICLE and det.estimated_distance <= self.caution_distance_m:
                if det.estimated_distance <= self.danger_distance_m:
                    priority = AlertPriority.CRITICAL
                    message = f"Emergency: Oncoming vehicle {det.spatial_zone} at {det.estimated_distance} meters."
                else:
                    priority = AlertPriority.HIGH
                    message = f"Caution: Vehicle detected {det.spatial_zone} at {det.estimated_distance} meters."

            # Rule B: Staircase Danger (Extremely high priority due to fall hazard)
            elif category == AlertCategory.STAIRCASE:
                if det.estimated_distance <= self.danger_distance_m:
                    priority = AlertPriority.CRITICAL
                    message = f"Danger: Stairs directly ahead! Stop."
                else:
                    priority = AlertPriority.HIGH
                    message = f"Stairs ahead, {det.estimated_distance} meters."

            # Rule C: Center Walking Path Obstacles
            elif det.spatial_zone == "center":
                if det.estimated_distance <= self.danger_distance_m:
                    priority = AlertPriority.CRITICAL
                    message = f"Warning: {det.class_name} directly ahead, {det.estimated_distance} meters."
                elif det.estimated_distance <= self.caution_distance_m:
                    priority = AlertPriority.HIGH
                    message = f"{det.class_name} ahead in center, {det.estimated_distance} meters."
                else:
                    # Center but distant
                    priority = AlertPriority.MEDIUM
                    message = f"Upcoming {det.class_name} in path."

            # Rule D: Side Obstacles (Left or Right)
            elif det.spatial_zone in ["left", "right"]:
                if det.estimated_distance <= self.danger_distance_m:
                    # Close enough to clip shoulders / cane bounds
                    priority = AlertPriority.HIGH
                    message = f"Avoid: {det.class_name} close on your {det.spatial_zone}."
                elif det.estimated_distance <= self.caution_distance_m:
                    # Caution range on sides; low warning priority
                    priority = AlertPriority.LOW
                    message = f"{det.class_name} to your {det.spatial_zone}."
                else:
                    # Distant objects on side - IGNORED completely to protect against cognitive clutter
                    continue

            # 4. Dispatch Alert and Set Cooldown
            if priority is not None:
                alert_uuid = str(uuid.uuid4())
                alert = Alert(
                    alert_id=alert_uuid,
                    category=category,
                    priority=priority,
                    message=message,
                    source_detection=det,
                    timestamp=t_now
                )
                active_alerts.append(alert)
                self.alerts_generated += 1

                # Establish alert cooldown to suppress identical announcements
                # Critical emergency categories have a shorter cooldown so user receives warnings faster
                cooldown_dur = self.default_cooldown_s
                if priority == AlertPriority.CRITICAL:
                    cooldown_dur = 1.5  # Critical dangers must be re-evaluated and voiced frequently
                elif priority == AlertPriority.LOW:
                    cooldown_dur = 6.0  # Background items are heavily throttled

                self.cooldown_tracker[cooldown_key] = t_now + cooldown_dur
                logger.debug(f"Dispatched alert: [{priority.upper()}] {message} (Cooldown active for {cooldown_dur}s)")

        # Periodic cleanup of expired cooldown logs to preserve system memory slots
        self._cleanup_expired_cooldowns(t_now)

        return active_alerts

    def _cleanup_expired_cooldowns(self, current_time: float) -> None:
        """
        Internal garbage collection method removing expired keys from the cooldown tracker.
        """
        expired_keys = [k for k, expire_time in self.cooldown_tracker.items() if current_time >= expire_time]
        for k in expired_keys:
            del self.cooldown_tracker[k]


# ================================================================================
# ENTIRE PIPELINE SIMULATION & LOCAL TESTING HOOK
# ================================================================================
if __name__ == "__main__":
    from src.camera import CameraStream
    from src.detection import DetectionEngine

    logger.info("Initializing complete pipeline simulation: Camera -> Detection -> Context Reasoning...")

    # 1. Spawn Camera
    stream = CameraStream(src="mock", width=640, height=480, fps=30)
    
    # 2. Spawn YOLO Detector (restricting classes for direct visual navigation examples)
    engine = DetectionEngine(model_name="yolov8n.pt", conf_threshold=0.25, allowed_classes=["chair", "person", "car"])
    
    # 3. Spawn Spatial Context Reasoning Engine
    reasoner = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5, default_cooldown_s=4.0)

    try:
        # Start camera thread
        stream.start()
        
        logger.info("Pipeline operating. Press 'Ctrl+C' in terminal to stop.")
        
        run_count = 0
        while run_count < 30:  # Loop for 30 execution ticks
            # Fetch latest frame thread-safely
            grabbed, frame = stream.read()
            
            if grabbed and frame is not None:
                # Step 1: Run Detection
                detections = engine.detect(frame)
                
                # Step 2: Run Reasoning Filters
                alerts = reasoner.process(detections)
                
                # Step 3: Print Output Pipeline results
                print(f"\n[FRAME #{stream.frame_count}] Processed {len(detections)} Raw Detections.")
                print(f"  Alerts Suppressed in cooldown: {reasoner.suppressed_alerts_count}")
                print(f"  Active Alerts Dispatched: {len(alerts)}")
                for alert in alerts:
                    print(f"    * [{alert.priority.upper()}] ({alert.category.name}) -> {alert.message}")
            else:
                logger.warning("Pipeline waiting for frame ingestion...")
                time.sleep(0.05)
                
            run_count += 1
            time.sleep(0.08)  # Processing interval
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received! Cleaning up pipeline safely...")
    finally:
        # Stop background threading and release hardware
        stream.stop()
        logger.info(f"Integrated simulation complete. Total detections: {reasoner.detections_processed} | Alerts voiced: {reasoner.alerts_generated}")
