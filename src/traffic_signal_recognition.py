# Traffic Signal Recognition using OpenCV
# Detects RED, YELLOW, GREEN traffic lights from image files

import cv2
import sys
from signal_detector import TrafficSignalDetector

def main(image_path="traffic.jpg", debug=False):
    """
    Detect traffic signal in image file.
    
    Args:
        image_path: Path to the image file
        debug: If True, display color masks
    """
    # Read image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Image '{image_path}' not found")
        return False
    
    # Resize for consistency
    image = cv2.resize(image, (500, 700))
    
    # Initialize detector
    detector = TrafficSignalDetector()
    
    # Detect signal
    signal_key, signal_text, color = detector.detect(image)
    
    # Display result
    cv2.putText(image, signal_text, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    cv2.imshow("Traffic Signal Recognition", image)
    
    # Debug mode: show color masks
    if debug:
        masks = detector.get_debug_masks(image)
        cv2.imshow("Red Mask", masks['red'])
        cv2.imshow("Yellow Mask", masks['yellow'])
        cv2.imshow("Green Mask", masks['green'])
    
    print(f"Detected: {signal_text}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    # Accept image path from command line or use default
    image_path = sys.argv[1] if len(sys.argv) > 1 else "traffic.jpg"
    debug = "--debug" in sys.argv
    main(image_path, debug)
