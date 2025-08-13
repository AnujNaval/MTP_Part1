import cv2
import numpy as np

# Load image
image = cv2.imread("retina_images/retina-1.jpeg")
if image is None:
    print("Error: Could not load image")
    exit()
else:
    print("Image Loaded Successfully")

# Step 1: Extract green channel (vessels are most visible here)
green_channel = image[:, :, 1]

# Step 2: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_green = clahe.apply(green_channel)

# Step 3: Apply median blur to reduce noise
blurred = cv2.medianBlur(enhanced_green, 5)

# Step 4: Use morphological 'black-hat' to enhance vessels
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)

# Step 5: Apply threshold to segment vessels
_, vessel_mask = cv2.threshold(blackhat, 15, 255, cv2.THRESH_BINARY)

# Step 6: Optional - remove small noise using morphological opening
kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)

# Step 7: Mask the vessels onto the original image
vessels_segmented = cv2.bitwise_and(image, image, mask=vessel_mask)

# Show results
cv2.imshow("Original", image)
cv2.imshow("Green Channel", green_channel)
cv2.imshow("Enhanced Green", enhanced_green)
cv2.imshow("Vessel Mask", vessel_mask)
cv2.imshow("Segmented Vessels", vessels_segmented)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the output
# cv2.imwrite("vessel_mask.png", vessel_mask)
# cv2.imwrite("segmented_vessels.png", vessels_segmented)
