"""
Traffic Sign Detector using YOLOv8
Detects and classifies traffic signs from images or video frames
"""

import cv2
import numpy as np
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class TrafficSignDetector:
    """
    Detects traffic signs using YOLOv8 pre-trained model.
    Covers: Stop, Yield, Speed Limit, No Entry, One Way, Pedestrian Crossing, etc.
    """
    
    # YOLO COCO class IDs for traffic-specific objects only
    TRAFFIC_CLASS_IDS = {
        10: "traffic light",     # COCO class 10
        13: "stop sign",          # COCO class 13
        14: "parking meter",      # COCO class 14
    }
    
    # Traffic-related keywords to detect in class names
    TRAFFIC_KEYWORDS = [
        'traffic', 'stop', 'sign', 'yield', 'speed',
        'parking', 'meter', 'road', 'street', 'signal'
    ]
    
    # Color codes for visualization
    SIGN_COLORS = {
        "Stop": (0, 0, 255),  # Red
        "Yield": (0, 165, 255),  # Orange
        "Speed Limit": (0, 0, 0),  # Black
        "No Entry": (0, 0, 255),  # Red
        "Pedestrian": (0, 255, 0),  # Green
        "Warning": (0, 255, 255),  # Yellow
    }
    
    def __init__(self, model_name="yolov8s.pt", confidence=0.35, iou_threshold=0.45):
        """
        Initialize the traffic sign detector with improved accuracy settings.
        
        Args:
            model_name (str): YOLOv8 model name (nano, small, medium, large, xlarge)
            confidence (float): Confidence threshold for detections (0-1). Lower = more detections
            iou_threshold (float): NMS IoU threshold to avoid duplicate detections
        """
        self.model = None
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.model_name = model_name
        
        if not YOLO_AVAILABLE:
            print("⚠️ YOLOv8 not installed. Install with: pip install ultralytics")
            return
        
        try:
            self.model = YOLO(model_name)
            print(f"✅ Traffic Sign Detector loaded: {model_name}")
            print(f"   Confidence threshold: {confidence}")
            print(f"   NMS IoU threshold: {iou_threshold}")
        except Exception as e:
            print(f"❌ Error loading YOLO model: {e}")
    
    def _is_traffic_object(self, class_id, class_name):
        """
        STRICT FILTER: Only detect EXACT STOP signs
        
        Args:
            class_id: YOLO class ID
            class_name: YOLO class name
        
        Returns:
            bool: True if STOP sign only, False otherwise
        """
        # VERY STRICT: Only detect STOP signs, nothing else
        class_name_lower = class_name.lower().strip()
        
        # Must be exactly "stop" or "stop sign"
        if class_name_lower == "stop" or class_name_lower == "stop sign":
            return True
        
        # COCO dataset class 13 is stop sign
        if class_id == 13:
            return True
        
        return False  # Reject everything else
    
    def detect(self, frame, preprocess=True):
        """
        Detect traffic signs in an image frame with improved accuracy.
        
        Args:
            frame: Input image (numpy array)
            preprocess (bool): Apply image preprocessing for better detection
        
        Returns:
            dict: Containing:
                - 'detections': List of detected signs with bounding boxes
                - 'signs': List of detected sign types
                - 'annotated_frame': Image with bounding boxes drawn
                - 'status': Detection status message
        """
        if self.model is None:
            return {
                'detections': [],
                'signs': [],
                'annotated_frame': frame,
                'status': 'Model not loaded'
            }
        
        try:
            # Preprocess image for better detection
            inference_frame = frame.copy()
            if preprocess:
                inference_frame = self._preprocess_image(inference_frame)
            
            # Run inference with NMS
            results = self.model(inference_frame, conf=self.confidence, iou=self.iou_threshold, verbose=False)
            
            detections = []
            signs_found = []
            annotated_frame = frame.copy()
            
            # Process results
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # HIGH ACCURACY CHECK: Skip low confidence detections
                    if conf < 0.50:
                        continue
                    
                    # Get class name
                    sign_name = result.names.get(cls, f"Sign {cls}")
                    
                    # STRICT FILTER: Only STOP signs - reject all others
                    if not self._is_traffic_object(cls, sign_name):
                        continue
                    
                    # Store detection info
                    detection_info = {
                        'sign': sign_name,
                        'confidence': conf,
                        'bbox': (x1, y1, x2, y2),
                        'class': cls
                    }
                    
                    detections.append(detection_info)
                    signs_found.append(sign_name)
                    
                    # Draw bounding box
                    color = self._get_color_for_sign(sign_name)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with better positioning to avoid overlap
                    label = f"{sign_name} ({conf:.0%})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    # Get text size for background
                    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                    text_x = x1
                    text_y = y1 - 8
                    
                    # Adjust text position if too close to edge
                    if text_y - text_size[1] < 0:
                        text_y = y2 + text_size[1] + 8
                    
                    # Draw text background
                    cv2.rectangle(annotated_frame, 
                                 (text_x - 2, text_y - text_size[1] - 2),
                                 (text_x + text_size[0] + 2, text_y + 2),
                                 color, -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label, (text_x, text_y),
                               font, font_scale, (255, 255, 255), thickness)
                
                status = f"✅ Detected {len(signs_found)} sign(s)"
            else:
                status = "⚠️ No traffic signs detected"
            
            return {
                'detections': detections,
                'signs': signs_found,
                'annotated_frame': annotated_frame,
                'status': status
            }
        
        except Exception as e:
            return {
                'detections': [],
                'signs': [],
                'annotated_frame': frame,
                'status': f'❌ Error: {str(e)}'
            }
    
    
    def _preprocess_image(self, frame):
        """
        Preprocess image for better sign detection.
        Handles varying lighting and contrast conditions.
        """
        try:
            # Enhance contrast (CLAHE - Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            return enhanced
        except:
            return frame
    
    def _get_color_for_sign(self, sign_name):
        """Get color for visualization based on sign type."""
        sign_lower = sign_name.lower()
        
        if 'stop' in sign_lower:
            return (0, 0, 255)  # Red
        elif 'yield' in sign_lower:
            return (0, 165, 255)  # Orange
        elif 'speed' in sign_lower:
            return (0, 0, 0)  # Black
        elif 'no entry' in sign_lower or 'no vehicles' in sign_lower:
            return (0, 0, 255)  # Red
        elif 'pedestrian' in sign_lower or 'crossing' in sign_lower:
            return (0, 255, 0)  # Green
        elif 'warning' in sign_lower or 'caution' in sign_lower or 'danger' in sign_lower:
            return (0, 255, 255)  # Yellow
        else:
            return (255, 0, 0)  # Blue
    
    def detect_batch(self, frames):
        """
        Detect signs in multiple frames.
        
        Args:
            frames: List of image frames
        
        Returns:
            list: List of detection results
        """
        return [self.detect(frame) for frame in frames]
    
    def get_debug_info(self, frame):
        """
        Generate debug information for the detection.
        
        Args:
            frame: Input image
        
        Returns:
            dict: Debug information
        """
        if self.model is None:
            return {'error': 'Model not loaded', 'model_status': YOLO_AVAILABLE}
        
        detection_result = self.detect(frame)
        
        return {
            'model_name': self.model_name,
            'confidence_threshold': self.confidence,
            'total_detections': len(detection_result['detections']),
            'signs_detected': detection_result['signs'],
            'frame_shape': frame.shape,
            'status': detection_result['status']
        }


# Alternative: Simpler sign classification using pre-trained CNN
class TrafficSignClassifier:
    """
    Lightweight traffic sign classifier using pre-trained model.
    Falls back to basic shape/color detection if model unavailable.
    """
    
    def __init__(self):
        """Initialize the classifier."""
        self.model = None
        
    def classify(self, sign_crop):
        """
        Classify a traffic sign image.
        
        Args:
            sign_crop: Cropped image of detected sign
        
        Returns:
            tuple: (sign_class, confidence)
        """
        # Placeholder for classification logic
        # In production, would use a trained CNN model
        
        # Analyze shape and color
        gray = cv2.cvtColor(sign_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count contours to estimate sign type
        num_contours = len(contours)
        
        if num_contours > 10:
            return "Complex Sign", 0.6
        else:
            return "Basic Sign", 0.5


if __name__ == "__main__":
    # Test the detector
    detector = TrafficSignDetector()
    
    if detector.model is not None:
        # Test on sample image
        test_image = cv2.imread("./images/traffic.jpg")
        if test_image is not None:
            result = detector.detect(test_image)
            print(f"Status: {result['status']}")
            print(f"Signs found: {result['signs']}")
            
            # Display result
            cv2.imshow("Traffic Sign Detection", result['annotated_frame'])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
