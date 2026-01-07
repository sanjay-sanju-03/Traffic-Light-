"""Create test images for traffic signal detection"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2
import numpy as np

# Get images directory
images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(images_dir, exist_ok=True)

# Create a red traffic signal image
red_img = np.zeros((500, 350, 3), dtype=np.uint8)
red_img[:, :] = (200, 200, 200)  # Light gray background
# Draw red circle
cv2.circle(red_img, (175, 150), 60, (0, 0, 255), -1)
cv2.imwrite(os.path.join(images_dir, 'red.jpg'), red_img)
print("✓ Created red.jpg")

# Create a yellow traffic signal image
yellow_img = np.zeros((500, 350, 3), dtype=np.uint8)
yellow_img[:, :] = (200, 200, 200)  # Light gray background
# Draw yellow circle
cv2.circle(yellow_img, (175, 250), 60, (0, 255, 255), -1)
cv2.imwrite(os.path.join(images_dir, 'yellow.jpg'), yellow_img)
print("✓ Created yellow.jpg")

# Create a green traffic signal image
green_img = np.zeros((500, 350, 3), dtype=np.uint8)
green_img[:, :] = (200, 200, 200)  # Light gray background
# Draw green circle
cv2.circle(green_img, (175, 350), 60, (0, 255, 0), -1)
cv2.imwrite(os.path.join(images_dir, 'green.jpg'), green_img)
print("✓ Created green.jpg")

print(f"\nTest images created successfully in: {images_dir}")
