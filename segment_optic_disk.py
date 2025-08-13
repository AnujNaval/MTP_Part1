import cv2
import numpy as np

# Load retina image
image = cv2.imread("retina_images/retina-4.jpeg")

if image is None:
    print("Error: Could not load image")
    exit()
else:
    print("Image loaded successfully")

# Convert to grayscale to simplify processing
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply median blur to reduce noise
blurred = cv2.medianBlur(gray, 5)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blurred)

# Threshold the image to segment bright regions (optic disc is one of the brightest)
_, thresh = cv2.threshold(enhanced, 235, 255, cv2.THRESH_BINARY)

# Use morphological operations to close small holes and connect nearby bright regions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Find contours of bright regions
contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    # Select the largest contour assuming it is the optic disc
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a minimum enclosing circle around the largest contour
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center, radius = (int(x), int(y)), int(radius * 0.9)  # Slightly smaller circle for better focus

    # Create an empty mask and draw the circle representing the optic disc
    mask = np.zeros_like(gray)
    cv2.circle(mask, center, radius, 255, -1)

    # Extract the optic disc region using the circular mask
    optic_disc_segment = cv2.bitwise_and(image, image, mask=mask)

    # Create a copy of the original image to draw the highlighted circle
    highlighted_image = image.copy()
    
    # Draw a green circle to highlight the optic disc on the original image
    cv2.circle(highlighted_image, center, radius, (0, 255, 0), 3)  # thickness=3

    # Save and show the results
    # cv2.imwrite("retina_images/optic_disc_segmented_circle.jpeg", optic_disc_segment)
    # cv2.imwrite("retina_images/optic_disc_highlighted_original.jpeg", highlighted_image)

    cv2.imshow("Optic Disc Highlighted on Original", highlighted_image)
    cv2.imshow("Segmented Optic Disc (Circle)", optic_disc_segment)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Optic disc not found")