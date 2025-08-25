import cv2
import numpy as np
import os

def segment_blood_vessels(image):
    print("Inside segmentation function")
    green_channel = image[:, :, 1]
    print("Green channel extracted")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    print("CLAHE applied")

    blurred = cv2.GaussianBlur(enhanced, (7, 7), 0)
    print("Blur applied")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, kernel)
    print("Black-hat applied")

    vessel_mask = cv2.adaptiveThreshold(blackhat, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 15, -2)
    print("Adaptive threshold applied")

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    vessel_mask = cv2.morphologyEx(vessel_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    print("Noise removed")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(vessel_mask)
    print(f"Connected components found: {num_labels}")

    sizes = stats[1:, cv2.CC_STAT_AREA]
    small_labels = np.where(sizes < 50)[0] + 1  # Labels of small objects
    mask = np.isin(labels, small_labels)
    vessel_mask[mask] = 0

    return vessel_mask


if __name__ == "__main__":
    input_folder = "dataset"
    output_folder = "blood_vessels"
    os.makedirs(output_folder, exist_ok=True)
    print("Output Folder made")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            print(f"Running code for {filename}")
            vessel_mask = segment_blood_vessels(image)
            print(f"Vessel Mask found for {filename}")
            
            if vessel_mask is not None:
                output_path = os.path.join(output_folder, f"vessels_{filename}")
                cv2.imwrite(output_path, vessel_mask)
                print(f"Saved: {output_path}")
    
    print("Processing complete!")