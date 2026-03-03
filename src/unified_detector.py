"""
Unified Traffic Detection System
Combines traffic light detection and traffic sign detection
"""

import cv2
import numpy as np
from signal_detector import TrafficDetector
try:
    from sign_detector import TrafficSignDetector
    SIGN_DETECTOR_AVAILABLE = True
except ImportError:
    SIGN_DETECTOR_AVAILABLE = False
    TrafficSignDetector = None


class UnifiedTrafficDetector:
    """
    Unified detector for both traffic lights and traffic signs.
    Provides a single interface for all traffic detection tasks.
    """
    
    def __init__(self, enable_lights=True, enable_signs=True, sign_confidence=0.35):
        """
        Initialize unified detector.
        
        Args:
            enable_lights (bool): Enable traffic light detection
            enable_signs (bool): Enable traffic sign detection
            sign_confidence (float): Confidence threshold for sign detection (higher = faster)
        """
        self.light_detector = None
        self.sign_detector = None
        self.enable_lights = enable_lights
        self.enable_signs = enable_signs
        
        if enable_lights:
            self.light_detector = TrafficDetector()
            print("✅ Traffic Light Detector initialized")
        
        if enable_signs and SIGN_DETECTOR_AVAILABLE:
            self.sign_detector = TrafficSignDetector(confidence=sign_confidence)
            print("✅ Traffic Sign Detector initialized")
    
    def detect_all(self, frame):
        """
        Detect both traffic lights and traffic signs in a frame.
        
        Args:
            frame: Input image (numpy array)
        
        Returns:
            dict: Comprehensive detection results
        """
        results = {
            'lights': None,
            'signs': None,
            'annotated_frame': frame.copy(),
            'summary': {}
        }
        
        # Detect traffic lights
        if self.enable_lights and self.light_detector is not None:
            try:
                signal_key, signal_text, color = self.light_detector.detect_light(frame)
                results['lights'] = {
                    'signal': signal_key,
                    'text': signal_text,
                    'color': color
                }
                
                results['summary']['traffic_light'] = {
                    'detected': signal_key.upper(),
                    'status': f"🚦 {signal_text}"
                }
                
                # Add light annotation to frame (single display only)
                annotated = frame.copy()
                cv2.rectangle(annotated, (10, 50), (200, 120), color, -1)
                cv2.putText(annotated, signal_text, (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                results['annotated_frame'] = annotated
            except Exception as e:
                print(f"❌ Light detection error: {e}")
        
        # Detect traffic signs
        if self.enable_signs and self.sign_detector is not None:
            try:
                sign_result = self.sign_detector.detect(frame)
                results['signs'] = sign_result
                
                annotated = results['annotated_frame']  # Use annotated frame from lights
                
                # Add sign annotations on top of light annotations
                for detection in sign_result.get('detections', []):
                    # Handle both 'bbox' (from YOLOv8) and 'box' (from HSV fallback) formats
                    if 'bbox' in detection:
                        x1, y1, x2, y2 = detection['bbox']
                        x, y, w, h = x1, y1, (x2 - x1), (y2 - y1)
                    elif 'box' in detection:
                        x, y, w, h = detection['box']
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    else:
                        continue
                    
                    name = detection.get('sign', detection.get('name', 'Unknown'))
                    color = detection.get('color', (0, 255, 0))
                    confidence = detection.get('confidence', 0)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    label = f"{name} ({confidence:.0%})" if 0 < confidence < 1 else name
                    cv2.putText(annotated, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                results['annotated_frame'] = annotated
                
                results['summary']['traffic_signs'] = {
                    'count': len(sign_result['signs']),
                    'signs': sign_result['signs'],
                    'status': sign_result['status']
                }
            except Exception as e:
                print(f"❌ Sign detection error: {e}")
                if 'traffic_light' not in results['summary']:
                    results['summary']['traffic_signs'] = {
                        'count': 0,
                        'signs': [],
                        'status': f"⚠️ Detection issue: {e}"
                    }
        
        return results
    
    def detect_lights_only(self, frame):
        """
        Detect only traffic lights.
        
        Args:
            frame: Input image
        
        Returns:
            dict: Traffic light detection results with annotated frame
        """
        if self.light_detector is None:
            return {'error': 'Traffic light detector not enabled'}
        
        signal_key, signal_text, color = self.light_detector.detect_light(frame)
        
        # Create annotated frame
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Draw signal text at top
        cv2.putText(annotated, f"Light: {signal_text}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw colored box
        cv2.rectangle(annotated, (10, 50), (200, 120), color, -1)
        cv2.putText(annotated, signal_text, (30, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        return {
            'signal': signal_key,
            'text': signal_text,
            'color': color,
            'message': f"🚦 {signal_text}",
            'annotated_image': annotated
        }
    
    def detect_signs_only(self, frame):
        """
        Detect only traffic signs.
        
        Args:
            frame: Input image
        
        Returns:
            dict: Traffic sign detection results with annotated frame
        """
        if self.sign_detector is None:
            # Fallback: use HSV-based sign detection
            return self._detect_signs_hsv(frame)
        
        try:
            return self.sign_detector.detect(frame)
        except:
            # Fallback to HSV-based detection
            return self._detect_signs_hsv(frame)
    
    def _detect_signs_hsv(self, frame):
        """
        Fallback HSV-based traffic sign detection.
        """
        if self.light_detector is None:
            return {'error': 'No detector available', 'signs': [], 'annotated_frame': frame}
        
        detections = self.light_detector.detect_signs(frame)
        
        annotated = frame.copy()
        signs_found = []
        
        for detection in detections:
            x, y, w, h = detection['box']
            name = detection['name']
            color = detection['color']
            
            signs_found.append(name)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            # Draw label
            cv2.putText(annotated, name, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        status = f"✅ Detected {len(signs_found)} sign(s)" if signs_found else "⚠️ No signs detected"
        
        return {
            'detections': detections,
            'signs': signs_found,
            'annotated_frame': annotated,
            'status': status
        }
    
    def get_status(self):
        """Get detector status."""
        return {
            'lights_enabled': self.enable_lights and self.light_detector is not None,
            'signs_enabled': self.enable_signs and self.sign_detector is not None,
            'light_detector': self.light_detector.__class__.__name__ if self.light_detector else None,
            'sign_detector': self.sign_detector.__class__.__name__ if self.sign_detector else None
        }


if __name__ == "__main__":
    import os
    
    # Test unified detector
    detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=True)
    
    # Test on sample image
    test_image_path = "./images/traffic.jpg"
    if os.path.exists(test_image_path):
        frame = cv2.imread(test_image_path)
        
        print("\n🔍 Running unified detection...")
        result = detector.detect_all(frame)
        
        print("\n📊 Results:")
        if result['lights']:
            print(f"  Traffic Light: {result['summary'].get('traffic_light', {}).get('detected')}")
        
        if result['signs']:
            print(f"  Traffic Signs Detected: {result['summary'].get('traffic_signs', {}).get('count')}")
            print(f"  Signs: {result['summary'].get('traffic_signs', {}).get('signs')}")
        
        # Display
        cv2.imshow("Unified Traffic Detection", result['annotated_frame'])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
