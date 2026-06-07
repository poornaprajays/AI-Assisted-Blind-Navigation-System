import pytest
import time
from src.schemas.contracts import Detection
from src.context_engine import ContextEngine, Alert, AlertPriority, AlertCategory


def create_detection(
    class_name: str,
    estimated_distance: float,
    spatial_zone: str = "center",
    confidence: float = 0.90,
    bbox=(100, 150, 300, 450),
    timestamp=None
) -> Detection:
    """Helper to construct a standard Detection object."""
    if timestamp is None:
        timestamp = time.time()
    
    # Pre-calculate centering metrics
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + (width // 2)
    center_y = ymin + (height // 2)
    
    return Detection(
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        center_x=center_x,
        center_y=center_y,
        width=width,
        height=height,
        estimated_distance=estimated_distance,
        spatial_zone=spatial_zone,
        timestamp=timestamp
    )


def test_distant_object_pruning():
    """Verify objects further than 3.5m are ignored entirely."""
    engine = ContextEngine()
    
    # 1. Distant center obstacle (distance = 4.0m)
    det_far = create_detection(class_name="chair", estimated_distance=4.0, spatial_zone="center")
    
    # 2. Near center obstacle (distance = 1.2m)
    det_near = create_detection(class_name="chair", estimated_distance=1.2, spatial_zone="center")
    
    alerts = engine.process([det_far, det_near])
    
    # Only the near detection should generate an alert
    assert len(alerts) == 1
    assert alerts[0].source_detection == det_near
    assert engine.detections_processed == 2
    assert engine.alerts_generated == 1


def test_priority_rule_vehicles():
    """Verify vehicle category priority mapping rules."""
    engine = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    
    # Critical vehicle (distance = 1.0m)
    det_critical = create_detection(class_name="car", estimated_distance=1.0, spatial_zone="center")
    alerts_crit = engine.process([det_critical])
    assert len(alerts_crit) == 1
    assert alerts_crit[0].priority == AlertPriority.CRITICAL
    assert alerts_crit[0].category == AlertCategory.VEHICLE
    assert "Emergency:" in alerts_crit[0].message

    # High vehicle (distance = 2.0m)
    engine_high = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    det_high = create_detection(class_name="car", estimated_distance=2.0, spatial_zone="center")
    alerts_high = engine_high.process([det_high])
    assert len(alerts_high) == 1
    assert alerts_high[0].priority == AlertPriority.HIGH
    assert alerts_high[0].category == AlertCategory.VEHICLE
    assert "Caution:" in alerts_high[0].message


def test_priority_rule_staircases():
    """Verify staircase category priority mapping rules."""
    engine = ContextEngine(danger_distance_m=1.5)
    
    # Critical stairs (distance = 1.2m)
    det_crit = create_detection(class_name="stairs", estimated_distance=1.2, spatial_zone="center")
    alerts_crit = engine.process([det_crit])
    assert len(alerts_crit) == 1
    assert alerts_crit[0].priority == AlertPriority.CRITICAL
    assert alerts_crit[0].category == AlertCategory.STAIRCASE
    assert "Stop" in alerts_crit[0].message

    # High stairs (distance = 2.8m)
    engine_high = ContextEngine(danger_distance_m=1.5)
    det_high = create_detection(class_name="stairs", estimated_distance=2.8, spatial_zone="center")
    alerts_high = engine_high.process([det_high])
    assert len(alerts_high) == 1
    assert alerts_high[0].priority == AlertPriority.HIGH
    assert "Stairs ahead" in alerts_high[0].message


def test_priority_rule_center_obstacles():
    """Verify center walking path obstacle priority mapping rules."""
    engine = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    
    # 1. Critical center obstacle
    det_crit = create_detection(class_name="chair", estimated_distance=1.0, spatial_zone="center")
    alerts_crit = engine.process([det_crit])
    assert len(alerts_crit) == 1
    assert alerts_crit[0].priority == AlertPriority.CRITICAL
    assert "directly ahead" in alerts_crit[0].message

    # 2. High center obstacle
    engine_high = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    det_high = create_detection(class_name="chair", estimated_distance=2.0, spatial_zone="center")
    alerts_high = engine_high.process([det_high])
    assert len(alerts_high) == 1
    assert alerts_high[0].priority == AlertPriority.HIGH
    assert "ahead in center" in alerts_high[0].message

    # 3. Medium center obstacle
    engine_med = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    det_med = create_detection(class_name="chair", estimated_distance=3.0, spatial_zone="center")
    alerts_med = engine_med.process([det_med])
    assert len(alerts_med) == 1
    assert alerts_med[0].priority == AlertPriority.MEDIUM
    assert "Upcoming" in alerts_med[0].message


def test_priority_rule_side_obstacles():
    """Verify side obstacles (outside walking path) mapping rules."""
    engine = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    
    # 1. High side obstacle (danger range)
    det_high = create_detection(class_name="table", estimated_distance=1.2, spatial_zone="left")
    alerts_high = engine.process([det_high])
    assert len(alerts_high) == 1
    assert alerts_high[0].priority == AlertPriority.HIGH
    assert "Avoid:" in alerts_high[0].message

    # 2. Low side obstacle (caution range, outside walking path)
    engine_low = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    det_low = create_detection(class_name="table", estimated_distance=2.0, spatial_zone="left")
    alerts_low = engine_low.process([det_low])
    assert len(alerts_low) == 1
    assert alerts_low[0].priority == AlertPriority.LOW
    assert "to your left" in alerts_low[0].message

    # 3. Ignored distant side obstacle (> caution range)
    engine_ignore = ContextEngine(danger_distance_m=1.5, caution_distance_m=2.5)
    det_ignore = create_detection(class_name="table", estimated_distance=3.0, spatial_zone="left")
    alerts_ignore = engine_ignore.process([det_ignore])
    assert len(alerts_ignore) == 0


def test_alert_cooldown_system():
    """Verify cooldown suppression logic and alert throttling."""
    engine = ContextEngine(default_cooldown_s=4.0)
    
    # First detection creates an alert
    det1 = create_detection(class_name="chair", estimated_distance=1.2, spatial_zone="center")
    alerts1 = engine.process([det1])
    assert len(alerts1) == 1
    assert engine.suppressed_alerts_count == 0
    assert engine.alerts_generated == 1
    
    # Second detection of same class and zone within cooldown period gets suppressed
    det2 = create_detection(class_name="chair", estimated_distance=1.2, spatial_zone="center")
    alerts2 = engine.process([det2])
    assert len(alerts2) == 0
    assert engine.suppressed_alerts_count == 1
    assert engine.alerts_generated == 1

    # Different class in same zone does NOT get suppressed
    det_diff_class = create_detection(class_name="person", estimated_distance=1.2, spatial_zone="center")
    alerts3 = engine.process([det_diff_class])
    assert len(alerts3) == 1
    assert engine.alerts_generated == 2

    # Same class in different zone does NOT get suppressed
    det_diff_zone = create_detection(class_name="chair", estimated_distance=1.2, spatial_zone="left")
    alerts4 = engine.process([det_diff_zone])
    assert len(alerts4) == 1
    assert engine.alerts_generated == 3


def test_cooldown_cleanup_after_expiration():
    """Verify cooldown maps expire and allow alerting again after timeout."""
    # We use a short default cooldown or test by artificially modifying the tracker
    engine = ContextEngine(default_cooldown_s=0.1)
    
    det1 = create_detection(class_name="chair", estimated_distance=2.0, spatial_zone="center")
    alerts1 = engine.process([det1])
    assert len(alerts1) == 1
    
    # Sleep slightly longer than default cooldown
    time.sleep(0.15)
    
    # Now it should generate alert again and clear the expired cooldown
    alerts2 = engine.process([det1])
    assert len(alerts2) == 1
    assert engine.suppressed_alerts_count == 0
    
    # Let's verify internal _cleanup_expired_cooldowns method directly
    engine.cooldown_tracker[("person", "center")] = time.time() - 10.0 # Expired
    engine.cooldown_tracker[("vehicle", "center")] = time.time() + 10.0 # Active
    
    engine._cleanup_expired_cooldowns(time.time())
    
    assert ("person", "center") not in engine.cooldown_tracker
    assert ("vehicle", "center") in engine.cooldown_tracker


def test_priority_based_cooldown_durations():
    """Verify that CRITICAL alerts have a shorter cooldown (1.5s) than default."""
    engine = ContextEngine(default_cooldown_s=5.0)
    
    t_now = time.time()
    # 1. Critical alert
    det_crit = create_detection(class_name="stairs", estimated_distance=1.0, spatial_zone="center")
    engine.process([det_crit])
    
    # Cooldown should be t_now + 1.5s
    crit_key = ("stairs", "center")
    assert crit_key in engine.cooldown_tracker
    assert abs(engine.cooldown_tracker[crit_key] - (t_now + 1.5)) < 0.2
    
    # 2. Low alert
    det_low = create_detection(class_name="table", estimated_distance=2.0, spatial_zone="left")
    engine.process([det_low])
    
    # Cooldown should be t_now + 6.0s
    low_key = ("table", "left")
    assert low_key in engine.cooldown_tracker
    assert abs(engine.cooldown_tracker[low_key] - (t_now + 6.0)) < 0.2
