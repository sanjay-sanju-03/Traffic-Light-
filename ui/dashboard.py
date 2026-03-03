"""Enhanced Dashboard with Traffic Lights and Signs Detection"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from unified_detector import UnifiedTrafficDetector

class TrafficDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("🚦 Unified Traffic Detection System - Lights & Signs")
        self.root.geometry("900x750")
        self.root.resizable(False, False)
        
        style = ttk.Style()
        style.theme_use('clam')
        
        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#e74c3c"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize unified detector - STOP SIGNS ONLY with HIGH accuracy
        try:
            self.detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=True, sign_confidence=0.55)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not initialize fully: {e}\nLights detection available")
            self.detector = UnifiedTrafficDetector(enable_lights=True, enable_signs=False)
        
        self.create_ui()
    
    def create_ui(self):
        """Create dashboard interface"""
        # Header
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=100)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        
        title = tk.Label(header_frame, text="🚦 Traffic Detection System",
                        font=("Arial", 24, "bold"), bg=self.primary_color, fg="white")
        title.pack(pady=10)
        
        subtitle = tk.Label(header_frame, text="Complete Detection: Traffic Lights (Red/Yellow/Green) + Signs (40+ types)",
                           font=("Arial", 10), bg=self.primary_color, fg="#ecf0f1")
        subtitle.pack()
        
        # Main content
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Options
        buttons_frame = tk.Frame(content_frame, bg=self.bg_color)
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        self.create_option_frame(buttons_frame, "📷 Webcam Detection", 
                                "Real-time: Lights + Signs", self.start_webcam, self.accent_color).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.create_option_frame(buttons_frame, "📁 Image Upload",
                                "Analyze: Lights + Signs", self.upload_image, self.success_color).pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Info frame
        info_frame = tk.LabelFrame(content_frame, text="ℹ️ Detection Modes", font=("Arial", 10, "bold"),
                                   bg=self.bg_color, fg=self.primary_color, padx=15, pady=15)
        info_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        info_text = """
📷 WEBCAM MODE - Real-time Detection:
  🚦 Traffic Lights: RED, YELLOW, GREEN
  🛑 Traffic Signs: STOP, YIELD, SPEED LIMIT + 40+ more
  (Press 'q' to exit)

📸 IMAGE UPLOAD MODE - Complete Analysis:
  🚦 Traffic Lights: Detect color (Red/Yellow/Green)
  🛑 Traffic Signs: Identify all signs in image
  (Upload image → Get full detection results)
        """
        
        info_label = tk.Label(info_frame, text=info_text.strip(), font=("Arial", 9),
                             bg=self.bg_color, fg="#34495e", justify=tk.LEFT)
        info_label.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg=self.primary_color, height=50)
        status_frame.pack(fill=tk.X, padx=0, pady=0)
        
        self.status_label = tk.Label(status_frame, text="Ready", font=("Arial", 10),
                                     bg=self.primary_color, fg="#ecf0f1")
        self.status_label.pack(pady=10)
    
    def create_option_frame(self, parent, title, desc, command, color):
        """Create option card"""
        frame = tk.Frame(parent, bg="white", relief=tk.RIDGE, borderwidth=2)
        
        title_label = tk.Label(frame, text=title, font=("Arial", 14, "bold"),
                              bg="white", fg=self.primary_color)
        title_label.pack(pady=10)
        
        desc_label = tk.Label(frame, text=desc, font=("Arial", 10),
                             bg="white", fg="#7f8c8d", wraplength=200, justify=tk.CENTER)
        desc_label.pack(pady=10)
        
        button = tk.Button(frame, text="Start →", font=("Arial", 11, "bold"),
                          bg=color, fg="white", padx=30, pady=10, border=0, cursor="hand2", command=command)
        button.pack(pady=15)
        
        return frame
    
    def update_status(self, message):
        """Update status"""
        self.status_label.config(text=message)
        self.root.update()
    
    def start_webcam(self):
        """Start webcam detection"""
        self.update_status("Starting webcam...")
        thread = threading.Thread(target=self._run_webcam, daemon=True)
        thread.start()
    
    def _run_webcam(self):
        """Run webcam with maximum performance optimization"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot access camera")
                self.update_status("Camera error")
                return
            
            # Aggressive camera optimization
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Get fresh frames only
            cap.set(cv2.CAP_PROP_FPS, 60)  # 60 FPS for ultra-smooth video
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)  # Lower resolution at source
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus lag
            
            self.update_status("📷 Webcam running @ 60 FPS... (Press 'q' to exit)")
            print("\n🚦 Traffic Detection - STOP SIGNS ONLY @ 60 FPS")
            print("Press 'q' to exit\n")
            
            frame_count = 0
            last_result = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                try:
                    # Moderate resolution for good detection at 60 FPS
                    proc_frame = cv2.resize(frame, (480, 360))
                    
                    # Detect frequently for responsive detection
                    if frame_count % 2 == 0:
                        try:
                            result = self.detector.detect_all(proc_frame)
                            last_result = result
                        except Exception as e:
                            result = last_result if last_result else {'annotated_frame': proc_frame}
                    else:
                        # Reuse last detection for smooth playback
                        result = last_result if last_result else {'annotated_frame': proc_frame}
                    
                    annotated = result.get('annotated_frame', proc_frame)
                    
                    # Minimal logging - only on detection changes
                    if frame_count % 120 == 0:
                        summary = result.get('summary', {})
                        light_info = summary.get('traffic_light', {})
                        print(f"🚦 {light_info.get('detected', '?')} | FPS: {int(frame_count/max(1, frame_count//30))}")
                    
                    # Display
                    cv2.imshow("🚦 Traffic Detection (Press 'q' to exit)", annotated)
                    
                except Exception as e:
                    cv2.imshow("🚦 Traffic Detection (Press 'q' to exit)", proc_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            self.update_status("Webcam closed")
            print(f"✅ Processed {frame_count} frames")
        
        except Exception as e:
            messagebox.showerror("Error", f"Webcam error: {str(e)}")
            self.update_status("Error occurred")
            print(f"❌ Error: {e}")
    
    def upload_image(self):
        """Upload and analyze image"""
        self.update_status("Opening file dialog...")
        
        file_path = filedialog.askopenfilename(
            title="Select image for traffic detection",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            self.update_status(f"Processing: {file_path.split('/')[-1]}...")
            thread = threading.Thread(target=self._process_image, args=(file_path,), daemon=True)
            thread.start()
    
    def _process_image(self, image_path):
        """Process image with BOTH traffic light and sign detection"""
        try:
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                self.update_status("Error loading image")
                return
            
            self.update_status("Processing traffic lights and signs...")
            print(f"\n📸 Analyzing: {image_path}")
            
            # Detect BOTH lights AND signs together
            result = self.detector.detect_all(frame)
            
            # Get detection results
            summary = result.get('summary', {})
            annotated_frame = result.get('annotated_frame', frame)
            
            light_info = summary.get('traffic_light', {})
            sign_info = summary.get('traffic_signs', {})
            
            # Display results
            print(f"\n{'='*60}")
            print(f"📊 DETECTION RESULTS:")
            print(f"{'='*60}")
            print(f"\n🚦 Traffic Light:")
            print(f"   Status: {light_info.get('status', 'N/A')}")
            print(f"   Detected: {light_info.get('detected', 'None')}")
            
            print(f"\n🛑 Traffic Signs ({sign_info.get('count', 0)} detected):")
            if sign_info.get('signs'):
                for i, sign in enumerate(sign_info.get('signs', []), 1):
                    print(f"   {i}. {sign}")
            else:
                print("   No signs detected")
            
            print(f"\n{sign_info.get('status', '')}")
            print(f"{'='*60}\n")
            
            # Show image with detections
            cv2.imshow("🚦 Traffic Detection Results: Lights + Signs", annotated_frame)
            self.update_status(f"✅ Detected - Light: {light_info.get('detected', 'None')} | Signs: {sign_info.get('count', 0)}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing error: {str(e)}")
            self.update_status("Error processing image")
            print(f"❌ Error: {e}")

def main():
    root = tk.Tk()
    app = TrafficDashboard(root)
    root.mainloop()

if __name__ == "__main__":
    main()
