import cv2
import numpy as np

def segment_optic_disk(image):
    if image is None:
        print("Error: Could not load image")
        exit()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Thresholding
    _, thresh = cv2.threshold(enhanced, 235, 255, cv2.THRESH_BINARY)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center, radius = (int(x), int(y)), int(radius * 0.9)
        
        mask = np.zeros_like(gray)
        cv2.circle(mask, center, radius, 255, -1)
        
        optic_disc_segment = cv2.bitwise_and(image, image, mask=mask)
        highlighted_image = image.copy()
        cv2.circle(highlighted_image, center, radius, (0, 255, 0), 3)
    else:
        print("Optic disc not found - using fallback")
        # Create placeholder images
        highlighted_image = image.copy()
        optic_disc_segment = np.zeros_like(image)
        # Add error message
        cv2.putText(highlighted_image, "Optic Disk Not Found", (50, image.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(optic_disc_segment, "Optic Disk Not Found", (50, image.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return highlighted_image, optic_disc_segment