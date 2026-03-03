# Traffic Signal and Sign Detection Engine
# Detects traffic lights (Red/Yellow/Green) and traffic signs (Stop/Yield/Speed Limit)

import cv2
import numpy as np
from skimage import measure

class TrafficDetector:
    """Detects traffic signals and signs using HSV and shape analysis."""
    
    # ========== TRAFFIC LIGHTS (HSV Ranges) ==========
    RED_LOWER1 = np.array([0, 100, 80])
    RED_UPPER1 = np.array([12, 255, 255])
    RED_LOWER2 = np.array([168, 100, 80])
    RED_UPPER2 = np.array([180, 255, 255])
    
    YELLOW_LOWER = np.array([20, 100, 80])
    YELLOW_UPPER = np.array([35, 255, 255])
    
    GREEN_LOWER = np.array([45, 100, 80])
    GREEN_UPPER = np.array([90, 255, 255])
    
    # ========== TRAFFIC SIGNS (HSV Ranges) ==========
    # Red signs (Stop, Yield)
    SIGN_RED_LOWER1 = np.array([0, 80, 100])
    SIGN_RED_UPPER1 = np.array([15, 255, 255])
    SIGN_RED_LOWER2 = np.array([165, 80, 100])
    SIGN_RED_UPPER2 = np.array([180, 255, 255])
    
    # White signs (Speed Limit)
    SIGN_WHITE_LOWER = np.array([0, 0, 200])
    SIGN_WHITE_UPPER = np.array([180, 30, 255])
    
    # Morphological kernel
    KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    KERNEL_SIGN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    def __init__(self):
        self.signal_names = {
            'red': 'RED LIGHT',
            'yellow': 'YELLOW LIGHT',
            'green': 'GREEN LIGHT',
            'none': 'NO SIGNAL'
        }
        
        self.sign_names = {
            'stop': 'STOP SIGN',
            'yield': 'YIELD SIGN',
            'speed_limit': 'SPEED LIMIT',
            'none': 'NO SIGN'
        }
        
        self.signal_colors = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'none': (255, 255, 255)
        }
        
        self.sign_colors = {
            'stop': (0, 0, 255),
            'yield': (0, 165, 255),
            'speed_limit': (0, 0, 0),
            'none': (255, 255, 255)
        }
    
    def detect_light(self, frame):
        """Detect traffic light color."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        red_mask = cv2.inRange(hsv, self.RED_LOWER1, self.RED_UPPER1) + \
                   cv2.inRange(hsv, self.RED_LOWER2, self.RED_UPPER2)
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        green_mask = cv2.inRange(hsv, self.GREEN_LOWER, self.GREEN_UPPER)
        
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.KERNEL)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.KERNEL)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, self.KERNEL)
        
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        if red_pixels > 50 and red_pixels > yellow_pixels and red_pixels > green_pixels:
            signal_key = 'red'
        elif yellow_pixels > 50 and yellow_pixels > red_pixels and yellow_pixels > green_pixels:
            signal_key = 'yellow'
        elif green_pixels > 50 and green_pixels > red_pixels and green_pixels > yellow_pixels:
            signal_key = 'green'
        else:
            signal_key = 'none'
        
        return signal_key, self.signal_names[signal_key], self.signal_colors[signal_key]
    
    def _circularity(self, contour):
        """Calculate circularity of a contour (0-1, where 1 is perfect circle)."""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return circularity
    
    def _shape_ratio(self, contour):
        """Get width/height ratio of contour bounding box."""
        x, y, w, h = cv2.boundingRect(contour)
        if h == 0:
            return 0
        return w / h
    
    def _count_corners(self, contour):
        """Approximate contour and count corners."""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return len(approx)
    
    def detect_signs(self, frame):
        """Detect traffic signs (Stop, Yield, Speed Limit)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        
        # Detect red masks for Stop and Yield
        red_mask = cv2.inRange(hsv, self.SIGN_RED_LOWER1, self.SIGN_RED_UPPER1) + \
                   cv2.inRange(hsv, self.SIGN_RED_LOWER2, self.SIGN_RED_UPPER2)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, self.KERNEL_SIGN)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.KERNEL_SIGN)
        
        # Detect white mask for Speed Limit
        white_mask = cv2.inRange(hsv, self.SIGN_WHITE_LOWER, self.SIGN_WHITE_UPPER)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, self.KERNEL_SIGN)
        
        detections = []
        
        # Process red signs (Stop and Yield)
        contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_red:
            area = cv2.contourArea(contour)
            if area < 500:  # Minimum area
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (signs should be reasonably large)
            if w < 20 or h < 20 or w > width * 0.8 or h > height * 0.8:
                continue
            
            corners = self._count_corners(contour)
            circularity = self._circularity(contour)
            
            # Stop sign: 8 corners (octagon), high circularity
            if 7 <= corners <= 9 and circularity > 0.4:
                detections.append({
                    'type': 'stop',
                    'name': self.sign_names['stop'],
                    'box': (x, y, w, h),
                    'color': self.sign_colors['stop'],
                    'confidence': min(0.95, circularity)
                })
            
            # Yield sign: 3 corners (triangle), lower circularity
            elif corners == 3 and circularity > 0.2:
                detections.append({
                    'type': 'yield',
                    'name': self.sign_names['yield'],
                    'box': (x, y, w, h),
                    'color': self.sign_colors['yield'],
                    'confidence': min(0.85, circularity + 0.2)
                })
        
        # Process white signs (Speed Limit)
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours_white:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Speed limit: rectangular, white, roughly square
            if w < 20 or h < 20 or w > width * 0.8 or h > height * 0.8:
                continue
            
            ratio = self._shape_ratio(contour)
            corners = self._count_corners(contour)
            
            # Speed limit: 4 corners, roughly square (0.8-1.2 ratio)
            if corners == 4 and 0.7 < ratio < 1.3:
                detections.append({
                    'type': 'speed_limit',
                    'name': self.sign_names['speed_limit'],
                    'box': (x, y, w, h),
                    'color': self.sign_colors['speed_limit'],
                    'confidence': 0.80
                })
        
        return detections if detections else [{'type': 'none', 'name': self.sign_names['none']}]
    
    def detect_all(self, frame):
        """Detect both traffic lights and signs."""
        light_signal, light_text, light_color = self.detect_light(frame)
        signs = self.detect_signs(frame)
        
        return {
            'light': {
                'type': light_signal,
                'text': light_text,
                'color': light_color
            },
            'signs': signs
        }
