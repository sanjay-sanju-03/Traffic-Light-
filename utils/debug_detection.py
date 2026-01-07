"""Debug traffic signal detection - shows color masks"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
from signal_detector import TrafficSignalDetector

def debug_image(image_path):
    """Show color detection masks for debugging"""
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    image = cv2.resize(image, (500, 700))
    
    # Initialize detector
    detector = TrafficSignalDetector()
    
    # Detect signal
    signal_key, signal_text, color = detector.detect(image)
    
    # Get masks for debugging
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    red_mask = cv2.inRange(hsv, detector.RED_LOWER1, detector.RED_UPPER1) + \
               cv2.inRange(hsv, detector.RED_LOWER2, detector.RED_UPPER2)
    yellow_mask = cv2.inRange(hsv, detector.YELLOW_LOWER, detector.YELLOW_UPPER)
    green_mask = cv2.inRange(hsv, detector.GREEN_LOWER, detector.GREEN_UPPER)
    
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    green_pixels = cv2.countNonZero(green_mask)
    
    print(f"\n📊 Detection Results for: {image_path}")
    print(f"{'='*50}")
    print(f"🔴 Red pixels:    {red_pixels}")
    print(f"🟡 Yellow pixels: {yellow_pixels}")
    print(f"🟢 Green pixels:  {green_pixels}")
    print(f"{'='*50}")
    print(f"✅ Detected: {signal_text}")
    print(f"{'='*50}\n")
    
    # Display result
    cv2.putText(image, signal_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow("Original", image)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Yellow Mask", yellow_mask)
    cv2.imshow("Green Mask", green_mask)
    
    print("Press any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_detection.py <image_path>")
        print("Example: python debug_detection.py ../images/red.jpg")
        sys.exit(1)
    
    debug_image(sys.argv[1])
