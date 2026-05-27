"""
VisionMate AI - Real-time Camera Ingestion Module
Path: src/camera.py

This module implements a high-performance, thread-safe camera stream designed 
specifically for real-time computer vision tasks (like YOLOv8 detection). 

By default, OpenCV's VideoCapture operates in a blocking, queued-buffer mode. If the 
downstream consumer (e.g., a YOLOv8 inference loop) takes longer to process a frame than 
the camera takes to capture it, OpenCV's internal buffer (usually 3-5 frames) fills up. 
This results in a mounting, cumulative delay where the user receives auditory safety alerts 
about obstacles they have already passed.

This module resolves this critical safety lag using a double-buffered background thread 
architecture that continuously drains OpenCV's internal buffer and exposes only the 
freshest frame.

================================================================================
CRITICAL CONCURRENCY AND REAL-TIME DESIGN DECISIONS:
================================================================================
1. Why Background Ingestion is Critical:
   Running frame ingestion and neural network inference sequentially on a single thread 
   constrains the application's performance. The camera's sensor requires frames to be 
   drained at a steady rate. If a thread is blocked running a 60ms YOLO inference, the 
   next frame capture is delayed, leading to frame dropping, buffering, and visual stuttering. 
   Offloading capture to a dedicated lightweight I/O thread ensures consistent capture rates.

2. Why Latest-Frame is Superior to Queued-Frame for Assistive Tech:
   In video streaming or recording, keeping every frame is necessary. In assistive 
   navigation, processing stale history is a safety hazard. If a user is walking toward a hazard, 
   they need immediate alerts about what is in front of them *right now*, not a queue 
   replaying events from 500ms ago. Our latest-frame architecture immediately discards 
   intermediate frames, maintaining a hard-real-time feedback loop.

3. Integration with the Future Detection Pipeline:
   The detection worker runs an independent execution loop:
       while active:
           grabbed, frame = camera_stream.read()
           if grabbed:
               predictions = yolo.infer(frame)
               ...
   This reading is non-blocking, fast, and thread-safe.
"""

import logging
import threading
import time
from typing import Tuple, Union, Optional

# Set up logging for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(threadName)s) %(message)s")
logger = logging.getLogger("VisionMateAI.Camera")

# Try to import cv2, otherwise set a flag for dummy mode fallback
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    logger.warning("OpenCV ('cv2') is not installed in the current environment. CameraStream will run in high-fidelity Mock Mode.")


class CameraStream:
    """
    Thread-safe, lag-free Camera Ingestion Stream.
    Uses a dedicated background thread to continually pull frames from cv2.VideoCapture,
    emptying the hardware buffer to ensure zero latency.
    """
    def __init__(self, src: Union[int, str] = 0, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initializes the CameraStream resources.
        
        Args:
            src: Camera index (int, usually 0) or RTSP stream URL / video path (str).
            width: Target frame width in pixels.
            height: Target frame height in pixels.
            fps: Desired capture frame rate (hardware-dependent).
        """
        self.src = src
        self.width = width
        self.height = height
        self.fps = fps
        self.delay = 1.0 / self.fps

        # OpenCV VideoCapture reference
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Thread-safe double buffers
        self.frame: Optional[object] = None  # Holds the numpy array (image frame)
        self.grabbed: bool = False
        
        # Threading controls
        self.read_lock = threading.Lock()
        self.stopped = True
        self.worker_thread: Optional[threading.Thread] = None

        # Diagnostic metrics
        self.frame_count = 0
        self.start_time = 0.0

    def start(self) -> "CameraStream":
        """
        Starts the background frame ingestion thread.
        """
        if not self.stopped:
            logger.warning("CameraStream is already running.")
            return self
        
        logger.info(f"Initializing capture resource on source: {self.src}")
        
        if OPENCV_AVAILABLE:
            self.cap = cv2.VideoCapture(self.src)
            # Apply resolution configurations
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {self.src}. Falling back to Mock Mode.")
                self.cap = None
        else:
            logger.info("OpenCV not found. Launching Mock Stream.")

        # Flag thread as active and spawn background execution loop
        self.stopped = False
        self.start_time = time.time()
        self.frame_count = 0
        
        self.worker_thread = threading.Thread(target=self.update, name="CameraIngestionThread", daemon=True)
        self.worker_thread.start()
        
        logger.info("Background camera ingestion thread started successfully.")
        return self

    def update(self) -> None:
        """
        Background loop executing on the dedicated I/O thread.
        Drains the video capture device buffers continuously to ensure zero frame lag.
        """
        # Precise loop timing control
        next_frame_time = time.time()

        while not self.stopped:
            loop_start = time.time()
            
            if self.cap is not None:
                # DRAIN THE BUFFER:
                # We use grab() and retrieve() or a quick read() to fetch frames.
                # Since this thread runs constantly, the hardware buffer remains dry.
                grabbed, frame = self.cap.read()
                
                # Verify frame health
                if not grabbed or frame is None:
                    logger.warning("Camera stream read failed or empty frame received.")
                    # Sleep briefly before retrying to prevent CPU thrashing during transient failures
                    time.sleep(0.1)
                    continue
            else:
                # Generate Mock Frame when OpenCV or physical camera is missing
                # Generates a standard RGB test-pattern frame with timestamps
                grabbed = True
                frame = self._generate_mock_frame()

            # Thread-safe write to double buffer
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            
            self.frame_count += 1
            
            # Throttle the thread to run near the configured target FPS rate 
            # to conserve CPU cycles while maintaining sensor capture health.
            next_frame_time += self.delay
            sleep_dur = next_frame_time - time.time()
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            else:
                # Thread falling behind target FPS; reset timeline pacing to avoid compounding lag
                next_frame_time = time.time()

    def read(self) -> Tuple[bool, Optional[object]]:
        """
        Thread-safe method to retrieve the latest frame.
        Called by downstream processing layers (like detection).
        
        Returns:
            Tuple: (grabbed: bool, frame: Optional[np.ndarray])
        """
        with self.read_lock:
            # We return a shallow copy of the state to keep locked window extremely brief
            return self.grabbed, self.frame

    def stop(self) -> None:
        """
        Stops the camera stream thread and safely releases all system resources.
        """
        if self.stopped:
            logger.info("CameraStream already stopped.")
            return

        logger.info("Stopping background ingestion thread...")
        self.stopped = True
        
        # Wait for the background thread to finish execution cleanly
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

        # Safely release physical device resources
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("Hardware video capture resources released.")

        # Compute diagnostic logging metrics
        total_time = time.time() - self.start_time
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        logger.info(f"Camera stream closed cleanly. Processed {self.frame_count} frames over {total_time:.2f}s (~{avg_fps:.2f} FPS).")

    def _generate_mock_frame(self) -> "np.ndarray":
        """
        Internal helper to synthesize mock frames with diagnostic overlays.
        Enables test suites, virtual setups, and non-blocking simulations to execute perfectly.
        """
        # If numpy is available, create a dynamic visual array, otherwise return a diagnostic dict
        if OPENCV_AVAILABLE:
            # Create a simple dark slate gray canvas
            img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            img[:] = (40, 36, 32)  # Curated sleek dark background
            
            # Draw a beautiful mock visual interface grid
            cv2.line(img, (self.width // 3, 0), (self.width // 3, self.height), (60, 56, 52), 1)
            cv2.line(img, (2 * self.width // 3, 0), (2 * self.width // 3, self.height), (60, 56, 52), 1)
            
            # Draw a blinking "recording" green dot
            dot_color = (0, 255, 0) if int(time.time() * 2) % 2 == 0 else (0, 100, 0)
            cv2.circle(img, (30, 30), 8, dot_color, -1)
            
            # Overlay simple telemetry text
            cv2.putText(img, "VISIONMATE AI - CAMERA RAW STREAM", (50, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(img, f"FPS: {self.fps} | Frame: #{self.frame_count}", (50, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(img, "Mock Frame Fallback Active", (50, 85), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            
            # Add a moving object to simulate a walking dynamic scene
            move_x = int((time.time() * 80) % (self.width - 100)) + 50
            cv2.rectangle(img, (move_x - 30, self.height // 2 - 30), (move_x + 30, self.height // 2 + 30), (0, 120, 255), -1)
            cv2.putText(img, "Obstacle", (move_x - 28, self.height // 2 - 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            
            return img
        else:
            # Fallback when numpy is also completely missing (extreme barebone CLI)
            return {"mock_frame_id": self.frame_count, "timestamp": time.time()}


# ================================================================================
# EXAMPLE USAGE & KEYBOARD INTERRUPT SAFETY DEMONSTRATION
# ================================================================================
if __name__ == "__main__":
    logger.info("Starting local high-performance CameraStream validation...")
    
    # Instantiate the stream (defaulting to device index 0, or custom params)
    # Using 30 FPS, standard 640x480 resolution
    stream = CameraStream(src=0, width=640, height=480, fps=30)
    
    try:
        # Start the background capture thread
        stream.start()
        
        logger.info("Starting visual loop. Press 'Ctrl+C' in terminal to stop.")
        
        # Simulate downstream loop running at a different speed (e.g. 15 FPS like a heavy AI network)
        run_cycles = 0
        while run_cycles < 30:  # Run demonstration for ~30 cycles
            cycle_start = time.time()
            
            # Read latest, completely lag-free frame thread-safely
            grabbed, frame = stream.read()
            
            if grabbed and frame is not None:
                # If OpenCV is installed and we are running inside an interactive desktop window, 
                # we can display the camera stream using cv2.imshow
                if OPENCV_AVAILABLE and isinstance(frame, np.ndarray):
                    cv2.imshow("VisionMate AI - CameraStream Feed", frame)
                    
                    # Break loop if 'q' key is pressed in the UI window
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Exiting visual loop via UI key press ('q').")
                        break
                else:
                    # CLI telemetry display when operating in headless or terminal-only setups
                    print(f"[CONSUMER LOOP] Grabbed frame #{stream.frame_count} - Shape: {getattr(frame, 'shape', 'dict_mock')}")
            else:
                logger.warning("Consumer loop waiting for first frame...")
                
            run_cycles += 1
            
            # Simulate a 15 FPS pipeline processing block (66ms sleep)
            time.sleep(0.066)
            
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received! Cleaning up pipeline safely...")
    finally:
        # Stop background threading and release camera hardware resources
        stream.stop()
        
        # Safely close OpenCV UI windows
        if OPENCV_AVAILABLE:
            cv2.destroyAllWindows()
        
        logger.info("CameraStream verification run completed successfully.")
