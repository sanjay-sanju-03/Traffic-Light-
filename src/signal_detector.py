# Shared Traffic Signal Detection Module
# Centralizes HSV color ranges and detection logic

import cv2
import numpy as np

class TrafficSignalDetector:
    """Detects traffic signal colors using HSV color space."""
    
    # HSV Color Ranges (Strict for traffic signals - high saturation)
    RED_LOWER1 = np.array([0, 100, 80])
    RED_UPPER1 = np.array([12, 255, 255])
    RED_LOWER2 = np.array([168, 100, 80])
    RED_UPPER2 = np.array([180, 255, 255])
    
    YELLOW_LOWER = np.array([20, 100, 80])
    YELLOW_UPPER = np.array([35, 255, 255])
    
    GREEN_LOWER = np.array([45, 100, 80])
    GREEN_UPPER = np.array([90, 255, 255])
    
    # Morphological kernel for noise reduction
    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Minimum pixel threshold to avoid false positives
    MIN_PIXELS = 50
    
    def __init__(self):
        self.signal_names = {
            'red': 'RED SIGNAL',
            'yellow': 'YELLOW SIGNAL',
            'green': 'GREEN SIGNAL',
            'none': 'NO SIGNAL'
        }
        self.signal_colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'none': (255, 255, 255)
        }
    
    def detect(self, frame):
        """
        Detect traffic signal in frame.
        
        Args:
            frame: OpenCV image (BGR format)
            
        Returns:
            tuple: (signal_name, display_text, color_bgr)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create color masks
        red_mask = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1) + \
                   cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        
        # Apply morphological operations to reduce noise
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.KERNEL)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.KERNEL)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.KERNEL)
        
        # Count pixels
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Decision logic with minimum threshold
        if red_pixels > self.MIN_PIXELS and red_pixels > yellow_pixels and red_pixels > green_pixels:
            signal_key = 'red'
        elif yellow_pixels > self.MIN_PIXELS and yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            signal_key = 'yellow'
        elif green_pixels > self.MIN_PIXELS and green_pixels > red_pixels and green_pixels > yellow_pixels:
            signal_key = 'green'
        else:
            signal_key = 'none'
        
        return signal_key, self.signal_names[signal_key], self.signal_colors[signal_key]
    
    def get_debug_masks(self, frame):
        """Get individual color masks for debugging."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1) + \
                   cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        
        return {'red': red_mask, 'yellow': yellow_mask, 'green': green_mask}
