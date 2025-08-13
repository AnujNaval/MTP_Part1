import cv2
import numpy as np

image = cv2.imread("retina_images/retina-1.jpeg")

if image is None:
    print("Error: Could not load image")
    exit()
else:
    print("Image loaded successfully")
    
# cv2.imshow("Original", image)

# Convert to grayscale: Many segmentation methods (thresholding, contouring) work best/simplest on 1-channel intensity images; also reduces noise/complexity.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray", gray)

# Apply Gaussian blur to reduce noise: Suppresses small noise/dust and smooths edges so thresholding and contour detection are more stable.
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow("Blur", blur)

# Threshold to separate bright retina from dark background
_, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
# cv2.imshow("Thresh", thresh)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
mask = np.zeros_like(gray)

# Draw the largest contour as filled white on mask: The retina (circular disc) is usually the largest external object; we’ll pick it.
cv2.drawContours(mask, [contours[0]], -1, 255, thickness=cv2.FILLED)

# Create masked image: Copy original pixels where mask==255, set others to 0 (black)
segmented = cv2.bitwise_and(image, image, mask=mask)

# Overlay mask on the original for visual confirmation: Make a copy and color the background (where mask==0) white.
overlay = image.copy()
overlay[mask == 0] = [255, 255, 255]  # White where it's masked out

# Show results
cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Segmented ROI", segmented)
cv2.imshow("ROI Highlighted", overlay)

# cv2.imwrite("output_images/segmented_image.jpeg", segmented)
# cv2.imwrite("output_images/overlay.jpeg", overlay)

cv2.waitKey(0)
cv2.destroyAllWindows()