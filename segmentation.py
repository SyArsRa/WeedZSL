import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── Hyperparameters ─────────────────────────────────────────────────────────────
IMAGE_PATH = r"D:\FYP\Dataset\weed\DJI_202412191545_004\DJI_20241219154909_0003_D.JPG"
SEGMENTATION_OUTPUT_DIR = "segmentation_results"
COMPONENTS_DIR = "temp_veg_components"  # Directory for temporary vegetation components
RESIZE_DIM = 1000
EXG_THRESH = 25  # From infer.py
MIN_AREA_THRESHOLD = 50  # From infer.py
TARGET_SEGMENT_SIZE = 256


# ─── VEGETATION DETECTION FUNCTIONS ───────────────────────────────────────────────
def generate_exg_vegetation_mask(bgr_img, exg_thresh=EXG_THRESH):
    """Generate vegetation mask using Excess Green Index (ExG)"""
    B, G, R = cv2.split(bgr_img.astype(np.float32))
    ExG = 2 * G - R - B
    _, veg = cv2.threshold(ExG, exg_thresh, 255, cv2.THRESH_BINARY)
    return veg.astype(np.uint8)


def segment_plants_by_exg_and_cc(
    img_bgr,
    exg_thresh=EXG_THRESH,
):
    """Localize plants by finding leaves via ExG, filtering, and clustering."""
    # 1) Vegetation mask
    veg = generate_exg_vegetation_mask(img_bgr, exg_thresh)

    # 3) Connected components on veg mask
    num_labels, labels_img, stats, cents = cv2.connectedComponentsWithStats(veg)

    # Return comprehensive results
    return {
        "veg_mask": veg,
        "labels_img": labels_img,  # Raw CC labels
        "stats": stats,  # Stats for raw CCs
        "centroids": cents,  # Centroids for raw CCs
        "num_labels": num_labels,  # Number of raw CCs (incl background)
    }


# --- Step 1: Modified extract_vegetation_components ---
def extract_vegetation_components(img, results, min_area_threshold=MIN_AREA_THRESHOLD):
    """Extract significant vegetation components as separate images"""
    component_count = 0
    component_map = {}  # Map original component index to new index

    os.makedirs(COMPONENTS_DIR, exist_ok=True)  # Ensure directory exists

    for i in range(
        1, results["num_labels"]
    ):  # Iterate original component indices (skip background 0)
        x, y, w, h, area = results["stats"][i]

        if area < min_area_threshold:
            continue

        component_mask = (results["labels_img"] == i).astype(np.uint8)
        component_img = cv2.bitwise_and(img, img, mask=component_mask)

        # Crop to the bounding box
        component_crop = component_img[y : y + h, x : x + w]

        if component_crop.size == 0:
            continue

        component_count += 1
        component_filename = f"component_{component_count:03d}.png"
        component_path = os.path.join(COMPONENTS_DIR, component_filename)
        cv2.imwrite(component_path, component_crop)

        # Store mapping from original component index to new component index
        component_map[i] = component_count

    return component_count, component_map


# --- Step 3: Modified create_visualization ---
def create_visualization(
    img,
    results,
    min_area_threshold=MIN_AREA_THRESHOLD,
    classification_results=None,
    selected_classes=None,
):
    """
    Create comprehensive visualizations for the segmentation results.
    Returns multiple visualizations in a single figure for debugging purposes.
    """
    counts = {"crops": 0, "weeds": 0}

    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # 2. Vegetation mask
    axes[1].imshow(results["veg_mask"], cmap="gray")
    axes[1].set_title("Vegetation Mask (ExG)")
    axes[1].axis("off")

    # 3. Connected Components visualization
    cc_vis = np.zeros_like(img)
    # Generate a random color for each component (skip background 0)
    for i in range(1, results["num_labels"]):
        area = results["stats"][i][4]
        if area < min_area_threshold:
            continue
        mask = (results["labels_img"] == i).astype(np.uint8)
        # Random color for this component
        color = np.random.randint(0, 255, 3).tolist()
        cc_vis[mask > 0] = color
    axes[2].imshow(cc_vis)
    axes[2].set_title("Connected Components")
    axes[2].axis("off")

    # 4. Bounding Boxes
    bbox_vis = img.copy()
    for i in range(1, results["num_labels"]):
        x, y, w, h, area = results["stats"][i]
        if area < min_area_threshold:
            continue
        cv2.rectangle(
            bbox_vis, (x, y), (x + w, y + h), (0, 255, 0), 2
        )  # Green bounding boxes
    axes[3].imshow(cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Bounding Boxes")
    axes[3].axis("off")

    # 6. Weed vs Crop classification
    weed_crop_vis = img.copy()
    if classification_results is not None and selected_classes is not None:
        for original_idx, (class_name, confidence, _) in classification_results.items():
            x, y, w, h = results["stats"][original_idx][:4]
            area = results["stats"][original_idx][4]

            if area < min_area_threshold:
                continue

            # Determine color: green for selected crops, red for weeds
            if class_name in selected_classes:
                color = (255, 255, 255)  # Pastel purple for crops
                counts["crops"] += 1
            else:
                color = (0, 0, 255)  # Red for weeds
                counts["weeds"] += 1

            # Draw bounding box
            cv2.rectangle(weed_crop_vis, (x, y), (x + w, y + h), color, 2)

            # Add class name and co     nfidence as text
            # label = f"{class_name[:10]} ({confidence:.2f})"
            # cv2.putText(weed_crop_vis, label, (x, y-5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # if color == (0, 255, 0):
            #    cv2.putText(weed_crop_vis, label, (x, y-5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    axes[5].imshow(cv2.cvtColor(weed_crop_vis, cv2.COLOR_BGR2RGB))
    axes[5].set_title(
        f'Weed vs Crop (Crops: {counts["crops"]}, Weeds: {counts["weeds"]})'
    )
    axes[5].axis("off")

    # Adjust layout and save the figure
    plt.tight_layout()

    return {
        "fig": fig,
        "counts": counts,
        "weed_crop_visualization": weed_crop_vis,  # Keep this for backward compatibility
    }


# ─── IMAGE PREPROCESSING ─────────────────────────────────────────────────────────
def resize_image_with_padding(
    image: np.ndarray, target_size: int = RESIZE_DIM
) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size, target_size, 3), dtype=image.dtype)
    y_off = (target_size - new_h) // 2
    x_off = (target_size - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + w] = resized
    return canvas


# ─── EXCESS GREEN INDEX ───────────────────────────────────────────────────────────
def compute_excess_green_index(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be RGB image")
    r, g, b = cv2.split(image.astype(np.float32))
    exg = 2 * g - r - b
    return exg


# ─── MASK CREATION ─────────────────────────────────────────────────────────────────
def create_vegetation_mask(
    exg: np.ndarray, threshold: float = EXG_THRESH
) -> np.ndarray:
    mask = (exg > threshold).astype(np.uint8) * 255
    return mask


# ─── CONNECTED COMPONENTS ──────────────────────────────────────────────────────────
def find_connected_components(mask: np.ndarray, min_area: int = MIN_AREA_THRESHOLD):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            boxes.append((x, y, w, h))
    return boxes, labels


# ─── SEGMENT EXTRACTION ───────────────────────────────────────────────────────────
def center_and_resize_segment(
    seg: np.ndarray, target_size: int = TARGET_SEGMENT_SIZE
) -> np.ndarray:
    h, w = seg.shape[:2]
    max_dim = max(h, w)
    canvas = np.zeros((max_dim, max_dim, 3), dtype=seg.dtype)
    y_off = (max_dim - h) // 2
    x_off = (max_dim - w) // 2
    canvas[y_off : y_off + h, x_off : x_off + w] = seg
    resized = cv2.resize(
        canvas, (target_size, target_size), interpolation=cv2.INTER_AREA
    )
    return resized


def extract_crop_segments(image: np.ndarray, mask: np.ndarray, boxes) -> list:
    segments = []
    for x, y, w, h in boxes:
        region = image[y : y + h, x : x + w]
        mask_region = mask[y : y + h, x : y + w]
        mask_3c = cv2.cvtColor(mask_region, cv2.COLOR_GRAY2RGB) / 255.0
        crop = (region * mask_3c).astype(np.uint8)
        segment = center_and_resize_segment(crop)
        segments.append(segment)
    return segments


# ─── VISUALIZATION ────────────────────────────────────────────────────────────────
def visualize_detection_results(
    image: np.ndarray, boxes, mask: np.ndarray = None
) -> np.ndarray:
    vis = image.copy()
    if mask is not None:
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = mask
        vis = cv2.addWeighted(vis, 0.8, overlay, 0.2, 0)
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"Plant {idx+1}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
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
        "visualization": visualization,
    }


# ─── SAVE RESULTS ────────────────────────────────────────────────────────────────
def save_results(results: dict, output_dir: str = SEGMENTATION_OUTPUT_DIR) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(output_dir, "visualization.jpg"),
        cv2.cvtColor(results["visualization"], cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(os.path.join(output_dir, "mask.jpg"), results["mask"])
    seg_dir = os.path.join(output_dir, "segments")
    os.makedirs(seg_dir, exist_ok=True)
    for i, seg in enumerate(results["segments"], start=1):
        cv2.imwrite(
            os.path.join(seg_dir, f"segment_{i:03d}.jpg"),
            cv2.cvtColor(seg, cv2.COLOR_RGB2BGR),
        )
