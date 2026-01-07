"""
Traffic Signal Recognition Dashboard
Provides GUI for users to choose between webcam or image upload
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
from signal_detector import TrafficSignalDetector


class TrafficSignalDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Signal Recognition System")
        self.root.geometry("900x700")
        self.root.resizable(False, False)
        
        # Set style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#f0f0f0"
        self.primary_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        
        self.root.configure(bg=self.bg_color)
        
        # Variables
        self.selected_image_path = None
        self.detector = TrafficSignalDetector()
        
        # Create UI
        self.create_ui()
    
    def create_ui(self):
        """Create dashboard UI"""
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.primary_color, height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        
        title = tk.Label(
            header_frame,
            text="🚦 Traffic Signal Recognition",
            font=("Arial", 24, "bold"),
            bg=self.primary_color,
            fg="white"
        )
        title.pack(pady=15)
        
        subtitle = tk.Label(
            header_frame,
            text="Choose an option to detect traffic signals",
            font=("Arial", 10),
            bg=self.primary_color,
            fg="#ecf0f1"
        )
        subtitle.pack()
        
        # Main content frame
        content_frame = tk.Frame(self.root, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Buttons frame
        buttons_frame = tk.Frame(content_frame, bg=self.bg_color)
        buttons_frame.pack(fill=tk.BOTH, expand=True)
        
        # Webcam option
        webcam_frame = self.create_option_frame(
            buttons_frame,
            "📷 Webcam Detection",
            "Use your webcam to detect traffic signals in real-time",
            self.start_webcam,
            self.accent_color
        )
        webcam_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Upload option
        upload_frame = self.create_option_frame(
            buttons_frame,
            "📁 Image Upload",
            "Upload an image to detect traffic signals",
            self.upload_image,
            self.success_color
        )
        upload_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Information frame
        info_frame = tk.LabelFrame(
            content_frame,
            text="ℹ️ Information",
            font=("Arial", 10, "bold"),
            bg=self.bg_color,
            fg=self.primary_color,
            padx=15,
            pady=15
        )
        info_frame.pack(fill=tk.BOTH, expand=True, pady=20)
        
        info_text = """
• Detects RED, YELLOW, and GREEN traffic signals
• Uses HSV color space for accurate detection
• Real-time processing with webcam
• Supports JPG, PNG, and BMP image formats
• Press 'q' to exit webcam mode or close window
        """
        
        info_label = tk.Label(
            info_frame,
            text=info_text.strip(),
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#34495e",
            justify=tk.LEFT
        )
        info_label.pack()
        
        # Status frame
        status_frame = tk.Frame(self.root, bg=self.primary_color, height=40)
        status_frame.pack(fill=tk.X, padx=0, pady=0)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 10),
            bg=self.primary_color,
            fg="#ecf0f1"
        )
        self.status_label.pack(pady=10)
    
    def create_option_frame(self, parent, title, description, command, color):
        """Create an option card frame"""
        frame = tk.Frame(parent, bg="white", relief=tk.RIDGE, borderwidth=2)
        
        # Title
        title_label = tk.Label(
            frame,
            text=title,
            font=("Arial", 14, "bold"),
            bg="white",
            fg=self.primary_color
        )
        title_label.pack(pady=10)
        
        # Description
        desc_label = tk.Label(
            frame,
            text=description,
            font=("Arial", 10),
            bg="white",
            fg="#7f8c8d",
            wraplength=200,
            justify=tk.CENTER
        )
        desc_label.pack(pady=10)
        
        # Button
        button = tk.Button(
            frame,
            text="Start →",
            font=("Arial", 11, "bold"),
            bg=color,
            fg="white",
            padx=30,
            pady=10,
            border=0,
            cursor="hand2",
            command=command
        )
        button.pack(pady=15)
        
        return frame
    
    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)
        self.root.update()
    
    def start_webcam(self):
        """Start webcam detection in a thread"""
        self.update_status("Starting webcam...")
        
        thread = threading.Thread(target=self._run_webcam, daemon=True)
        thread.start()
    
    def _run_webcam(self):
        """Run webcam detection"""
        try:
            from webcam import main as webcam_detection
            self.update_status("Webcam running... (Press 'q' to exit)")
            result = webcam_detection(camera_id=0, display_fps=True, exit_key='q')
            if result is False:
                self.update_status("Webcam not available")
            else:
                self.update_status("Webcam closed")
        except Exception as e:
            messagebox.showerror("Error", f"Webcam error: {str(e)}")
            self.update_status("Error occurred")
    
    def upload_image(self):
        """Open file dialog to upload image"""
        self.update_status("Opening file dialog...")
        
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select a traffic signal image",
            filetypes=file_types
        )
        
        if file_path:
            self.selected_image_path = file_path
            self.update_status(f"Processing: {file_path.split('/')[-1]}...")
            
            thread = threading.Thread(
                target=self._process_image,
                args=(file_path,),
                daemon=True
            )
            thread.start()
    
    def _process_image(self, image_path):
        """Process uploaded image"""
        try:
            import cv2
            from signal_detector import TrafficSignalDetector
            import numpy as np
            from PIL import Image as PILImage
            
            # Convert path to absolute and normalize for Windows
            image_path = os.path.abspath(image_path)
            
            # Check if file exists
            if not os.path.exists(image_path):
                messagebox.showerror("Error", f"File not found: {image_path}")
                self.update_status("Error: File not found")
                return
            
            # Try reading with PIL first (more reliable on Windows)
            try:
                pil_image = PILImage.open(image_path)
                pil_image = pil_image.convert('RGB')
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            except Exception as pil_error:
                # Fallback to cv2
                image = cv2.imread(image_path)
                if image is None:
                    messagebox.showerror("Error", f"Could not read image. File may be corrupted or unsupported format:\n{image_path}")
                    self.update_status("Error: Invalid image file")
                    return
            
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
            print(f"Detected: {signal_text}")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            self.update_status(f"Detected: {signal_text}")
        except Exception as e:
            messagebox.showerror("Error", f"Image processing error: {str(e)}")
            self.update_status("Error processing image")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = TrafficSignalDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()
