# retina_analysis.py
import cv2
import numpy as np
import os
from segment_roi import segment_roi
from segment_optic_disk import segment_optic_disk
from segment_blood_vessels import segment_blood_vessels

# Create output directory if it doesn't exist
os.makedirs("output_images", exist_ok=True)

def create_composite(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print("\nProcessing image:", os.path.basename(image_path))
    
    # Perform all segmentations
    roi_segmented, roi_overlay = segment_roi(image)
    od_highlighted, od_segment = segment_optic_disk(image)
    vessel_mask = segment_blood_vessels(image)
    
    # Convert vessel mask to 3-channel for display
    vessel_mask_bgr = cv2.cvtColor(vessel_mask, cv2.COLOR_GRAY2BGR)
    
    # Get dimensions
    h, w = image.shape[:2]
    composite = np.zeros((2 * h, 3 * w, 3), dtype=np.uint8)
    
    # Add titles to images
    def add_title(img, title):
        titled_img = img.copy()
        cv2.putText(titled_img, title, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return titled_img
    
    # Arrange images in grid
    composite[0:h, 0:w] = add_title(image, "Original")
    composite[0:h, w:2*w] = add_title(roi_segmented, "ROI Segmented")
    composite[0:h, 2*w:3*w] = add_title(roi_overlay, "ROI Overlay")
    composite[h:2*h, 0:w] = add_title(od_highlighted, "Optic Disk Highlighted")
    composite[h:2*h, w:2*w] = add_title(od_segment, "Optic Disk Segment")
    composite[h:2*h, 2*w:3*w] = add_title(vessel_mask_bgr, "Blood Vessel")
    
    # Save composite image
    output_path = f"output_images/{os.path.splitext(os.path.basename(image_path))[0]}_composite.jpg"
    cv2.imwrite(output_path, composite)
    print(f"Composite image saved to {output_path}")
    
    # Display results
    cv2.imshow("Retina Analysis Results", composite)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Specify the image filename to process
    image_filename = "retina-4.jpeg"  # Change this to your desired image filename
    image_path = os.path.join("retina_images", image_filename)
    
    # Process the specified image
    create_composite(image_path)