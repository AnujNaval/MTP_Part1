import cv2
import numpy as np

def segment_blood_vessels(image):
    if image is None:
        print("Error: Could not load image")
        exit()
    
    # Extract green channel
    green_channel = image[:, :, 1]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green = clahe.apply(green_channel)
    
    # Apply median blur
    blurred = cv2.medianBlur(enhanced_green, 5)
    
    # Morphological black-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding
    _, vessel_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    return vessel_mask