# Real-Time Traffic Signal Recognition using Webcam
# OpenCV + HSV Color Detection

import cv2
from signal_detector import TrafficSignalDetector

def main(camera_id=0, display_fps=True, exit_key='q'):
    """
    Real-time traffic signal detection from webcam.
    
    Args:
        camera_id: Camera index (default 0 for primary camera)
        display_fps: Show FPS counter
        exit_key: Key to exit (default 'q')
    """
    # Start webcam
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"Error: Camera {camera_id} not accessible")
        return False
    
    # Initialize detector
    detector = TrafficSignalDetector()
    
    # FPS tracking
    frame_count = 0
    
    print(f"Press '{exit_key}' to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read from camera")
            break
        
        frame_count += 1
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (640, 480))
        
        # Detect traffic signal
        signal_key, signal_text, color = detector.detect(frame)
        
        # Display result
        cv2.putText(frame, signal_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        
        # Optional: Display FPS
        if display_fps and frame_count % 10 == 0:
            print(f"Processed {frame_count} frames | Last signal: {signal_text}")
        
        cv2.imshow("Real-Time Traffic Signal Recognition", frame)
        
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord(exit_key):
            print(f"Exiting... (processed {frame_count} frames)")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    return True

if __name__ == "__main__":
    main()
