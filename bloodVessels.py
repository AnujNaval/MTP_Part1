import cv2
import numpy as np
import os

def segment_blood_vessels(image):
    if image is None:
        print("Error: Could not load image")
        exit()
    
    # Extract green channel
    green_channel = image[:, :, 1]
    
    #cv2.imshow("Green Channel",green_channel)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    # cv2.imshow("Enhanced", enhanced)
    blurred = cv2.medianBlur(enhanced, 5)
    cv2.imshow("Blurred", blurred)
    
    # Morphological black-hat
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    
    # Thresholding
    _, vessel_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)
    
    # Remove small noise
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    return vessel_mask

if __name__ == "__main__":
    image_filename = "retina-1.jpg"
    image_path = os.path.join("dataset", image_filename)

    image = cv2.imread(image_path)

    vessel_mask = segment_blood_vessels(image)

    cv2.imshow("Vessel Mask", vessel_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()