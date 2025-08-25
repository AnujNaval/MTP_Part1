import cv2
import numpy as np
import os

def segment_optic_disk(image):
    if image is None:
        print("Error: Could not load image")
        return None, None
    
    # Extract green channel
    green_channel = image[:, :, 1]
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)

    # Median Blur
    blurred = cv2.medianBlur(enhanced, 15)
    # cv2.imshow("Blurred-Median", blurred)
    
    # Thresholding
    _, thresh = cv2.threshold(blurred, 160, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center, radius = (int(x), int(y)), int(radius * 0.9)
        
        mask = np.zeros_like(green_channel)
        cv2.circle(mask, center, radius, 255, -1)
        
        optic_disc_segment = cv2.bitwise_and(image, image, mask=mask)
        highlighted_image = image.copy()
        cv2.circle(highlighted_image, center, radius, (0, 255, 0), 3)
    else:
        print("Optic disc not found")
        highlighted_image = image.copy()
        optic_disc_segment = np.zeros_like(image)
    
    return highlighted_image, optic_disc_segment

if __name__ == "__main__":
    input_folder = "dataset"
    output_folder = "highlighted_output"
    os.makedirs(output_folder, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            highlighted, segmented = segment_optic_disk(image)
            
            if highlighted is not None:
                output_path = os.path.join(output_folder, f"highlighted_{filename}")
                cv2.imwrite(output_path, highlighted)
                print(f"Saved: {output_path}")
    
    print("Processing complete!")
