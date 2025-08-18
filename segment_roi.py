import cv2
import numpy as np

def segment_roi(image):
    if image is None:
        print("Error: Could not load image")
        exit()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Create mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contours[0]], -1, 255, thickness=cv2.FILLED)
    
    # Create segmented image
    segmented = cv2.bitwise_and(image, image, mask=mask)
    
    # Create overlay
    overlay = image.copy()
    overlay[mask == 0] = [255, 255, 255]  # White background
    
    return segmented, overlay