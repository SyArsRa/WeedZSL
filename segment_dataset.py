# THis file extracts crops from RGB images and saves them as separate files.
# It uses a vegetation mask to isolate the chilli crops from the background.
# This is done so that when running evaluate_embeddings.py, the crops can be used for evaluation with having to 
# segmented again and again.
import os
import cv2
import numpy as np
from segmentation import generate_exg_vegetation_mask

# -------------------- Save Component as RGB Crop -------------------- #
def save_rgb_components(crop_rgb, mask, output_dir, img_name):
    base = os.path.splitext(img_name)[0]
    chilli_rgb = cv2.bitwise_and(crop_rgb, crop_rgb, mask=mask)
    cv2.imwrite(os.path.join(output_dir, f"{base}_chilli.png"), chilli_rgb)

# -------------------- Main Pipeline -------------------- #
def main(img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]

    for img_file in image_files:
        image_path = os.path.join(img_dir, img_file)
        image = cv2.imread(image_path)

        mask = generate_exg_vegetation_mask(image)

        mask = (mask > 127).astype(np.uint8)
        save_rgb_components(image, mask, output_dir, img_file)

if __name__ == "__main__":
    main(
        img_dir=r'datasets\custom_dataset\chilli/',
        output_dir=r'datasets\custom_dataset\new_components_rgb/'
    )
