import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── Hyperparameters ─────────────────────────────────────────────────────────────
IMAGE_PATH = r"D:\FYP\Dataset\weed\DJI_202412191545_004\DJI_20241219154909_0003_D.JPG"
OUTPUT_DIR = "segmentation_results"
COMPONENTS_DIR = "temporary_vegetation_components"
RESIZE_DIM = 1000
EXG_THRESH = 30.0
MIN_AREA = 15
TARGET_SEGMENT_SIZE = 256

# ─── IMAGE PREPROCESSING ─────────────────────────────────────────────────────────
def resize_image_with_padding(image: np.ndarray, target_size: int = RESIZE_DIM) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

# ─── EXCESS GREEN INDEX ───────────────────────────────────────────────────────────
def compute_excess_green_index(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image")
    r, g, b = cv2.split(image.astype(np.float32))
    exg = 2 * g - r - b
    return exg

# ─── MASK CREATION ─────────────────────────────────────────────────────────────────
def create_vegetation_mask(exg: np.ndarray, threshold: float = EXG_THRESH) -> np.ndarray:
    mask = (exg > threshold).astype(np.uint8) * 255
    return mask

# ─── CONNECTED COMPONENTS ──────────────────────────────────────────────────────────
def find_connected_components(mask: np.ndarray, min_area: int = MIN_AREA):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, w, h))
    return boxes, labels

# ─── SEGMENT EXTRACTION ───────────────────────────────────────────────────────────
def center_and_resize_segment(seg: np.ndarray, target_size: int = TARGET_SEGMENT_SIZE) -> np.ndarray:
    h, w = seg.shape[:2]
    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=seg.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off:y_off+h, x_off:x_off+w] = seg
    resized = cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return resized

def extract_crop_segments(image: np.ndarray, mask: np.ndarray, boxes) -> list:
    segments = []
    for (x, y, w, h) in boxes:
        region = image[y:y+h, x:x+w]
        mask_region = mask[y:y+h, x:x+w]
        mask_3c = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2RGB) / 255.0
        crop = (region * mask_3c).astype(np.uint8)
        segment = center_and_resize_segment(crop)
        segments.append(segment)
    return segments

# ─── VISUALIZATION ────────────────────────────────────────────────────────────────
def visualize_detection_results(image: np.ndarray, boxes, mask: np.ndarray = None) -> np.ndarray:
    vis = image.copy()
    if mask is not None:
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = mask
        vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, f"Plant {idx+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis

# ─── FULL PIPELINE ───────────────────────────────────────────────────────────────
def process_field_image(image_path: str = IMAGE_PATH) -> dict:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = resize_image_with_padding(img_rgb)
    exg = compute_excess_green_index(resized)
    mask = create_vegetation_mask(exg)
    boxes, labels = find_connected_components(mask)
    segments = extract_crop_segments(resized, mask, boxes)
    visualization = visualize_detection_results(resized, boxes, mask)
    return {
        "original": img_rgb,
        "resized": resized,
        "exg": exg,
        "mask": mask,
        "boxes": boxes,
        "labels": labels,
        "segments": segments,
        "visualization": visualization
    }

# ─── SAVE RESULTS ────────────────────────────────────────────────────────────────
def save_results(results: dict, output_dir: str = OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "visualization.jpg"),
                cv2.cvtColor(results["visualization"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, "mask.jpg"), results["mask"])
    seg_dir = os.path.join(output_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for i, seg in enumerate(results["segments"], start=1):
        cv2.imwrite(os.path.join(seg_dir, f"segment_{i:03d}.jpg"),
                    cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))

# ─── MAIN EXECUTION ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    img_path = sys.argv[1]
    print(f"Processing {img_path}...")
    results = process_field_image(img_path)
    save_results(results)
    print(f"Detected {len(results['boxes'])} plants. Results saved to '{OUTPUT_DIR}'.")
