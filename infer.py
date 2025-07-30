import json
import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

from embedding_models.embeddings_slip import get_similar_classes_slip
from embedding_models.embedding_clip import get_similar_classes_clip
from embedding_models.embeddings_gemini import get_similar_classes_gemini
from embedding_models.embeddings_imagebind import get_similar_classes_imagebind
from embedding_models.embeddings_openclip import get_similar_classes_openclip
from embedding_models.embeddings_llama import get_similar_classes_llama

from classification_models.MobileNet import MobileNetV2
from classification_models.ResNet18 import ResNet18
from classification_models.ShuffleNet import ShuffleNet
from classification_models.ShuffleNet_SE import ShuffleNetV2WithSE
from classification_models.ShuffleNet_SEPCONV import ShuffleNetV2WithSepConv
from classification_models.ShuffleNet_SEPCONV_SE import ShuffleNetV2WithSepConvAndSE
from classification_models.SqueezeNet import SqueezeNet

from class_mapping import ID_TO_NAME, NAME_TO_ID
# --- Hyperparameters ---
IMAGE_PATH = r"D:\FYP\Dataset\weed\DJI_202412191545_004\DJI_20241219154909_0003_D.JPG"
MODEL_PATH = "D:\weed_new\models\shufflenet.pt"
RESIZE_DIM = 1000
OUTPUT_DIR = "results_output"
COMPONENTS_DIR = "temp_veg_components"
CLASSIFIED_DIR = "classified_plants_output"
MIN_AREA_THRESHOLD = 50
EXG_THRESH = 25
LEAF_AREA_RANGE = (25, 2500)
SOLIDITY_THRESH = 0.7
ASPECT_RATIO_RANGE = (0.15, 6.0)
MAIN_CROP = "Wheat"
EMBEDDING_MODEL = "slip" # Options: "slip", "clip", "gemini", "imagebind", "openclip", "llama"
CLASSIFICATION_MODEL = "shufflenet" # Options: "shufflenet", "mobilenet", "resnet18", "squeezenet", "shufflenet_se", "shufflenet_sepconv", "shufflenet_sepconv_se"
DEVICE = None # "cuda" if torch.cuda.is_available() else "cpu"

inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.25, 0.25, 0.25]),
])
# ─── VEGETATION DETECTION FUNCTIONS ───────────────────────────────────────────────
def generate_exg_vegetation_mask(bgr_img, exg_thresh=EXG_THRESH):
    """Generate vegetation mask using Excess Green Index (ExG)"""
    B, G, R = cv2.split(bgr_img.astype(np.float32))
    ExG = 2*G - R - B
    _, veg = cv2.threshold(ExG, exg_thresh, 255, cv2.THRESH_BINARY)
    return veg.astype(np.uint8)





def segment_plants_by_exg_and_cc(img_bgr,
                                       exg_thresh=EXG_THRESH,
                                       apply_morph_clean=False): # Optional cleaning
    """Localize plants by finding leaves via ExG, filtering, and clustering."""
    # 1) Vegetation mask
    veg = generate_exg_vegetation_mask(img_bgr, exg_thresh)


    # 3) Connected components on veg mask
    num_labels, labels_img, stats, cents = cv2.connectedComponentsWithStats(veg)

    # Return comprehensive results
    return {
        'veg_mask': veg,
        'labels_img': labels_img, # Raw CC labels
        'stats': stats,           # Stats for raw CCs
        'centroids': cents,       # Centroids for raw CCs
        'num_labels': num_labels, # Number of raw CCs (incl background)
    }

# --- Step 1: Modified extract_vegetation_components ---
def extract_vegetation_components(img, results, min_area_threshold=MIN_AREA_THRESHOLD):
    """Extract significant vegetation components as separate images"""
    component_count = 0
    component_map = {}  # Map original component index to new index

    os.makedirs(COMPONENTS_DIR, exist_ok=True)  # Ensure directory exists

    for i in range(1, results['num_labels']):  # Iterate original component indices (skip background 0)
        x, y, w, h, area = results['stats'][i]

        if area < min_area_threshold:
            continue

        component_mask = (results['labels_img'] == i).astype(np.uint8)
        component_img = cv2.bitwise_and(img, img, mask=component_mask)

        # Crop to the bounding box
        component_crop = component_img[y:y+h, x:x+w]

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
def create_visualization(img, results, min_area_threshold=MIN_AREA_THRESHOLD, classification_results=None, selected_classes=None):
    """
    Create comprehensive visualizations for the segmentation results.
    Returns multiple visualizations in a single figure for debugging purposes.
    """
    counts = {
        "crops": 0,
        "weeds": 0
    }
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Vegetation mask
    axes[1].imshow(results['veg_mask'], cmap='gray')
    axes[1].set_title('Vegetation Mask (ExG)')
    axes[1].axis('off')
    
    # 3. Connected Components visualization
    cc_vis = np.zeros_like(img)
    # Generate a random color for each component (skip background 0)
    for i in range(1, results['num_labels']):
        area = results['stats'][i][4]
        if area < min_area_threshold:
            continue
        mask = (results['labels_img'] == i).astype(np.uint8)
        # Random color for this component
        color = np.random.randint(0, 255, 3).tolist()
        cc_vis[mask > 0] = color
    axes[2].imshow(cc_vis)
    axes[2].set_title('Connected Components')
    axes[2].axis('off')
    
    # 4. Bounding Boxes
    bbox_vis = img.copy()
    for i in range(1, results['num_labels']):
        x, y, w, h, area = results['stats'][i]
        if area < min_area_threshold:
            continue
        cv2.rectangle(bbox_vis, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green bounding boxes
    axes[3].imshow(cv2.cvtColor(bbox_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Bounding Boxes')
    axes[3].axis('off')
    
    
    
    # 6. Weed vs Crop classification
    weed_crop_vis = img.copy()
    if classification_results is not None and selected_classes is not None:
        for original_idx, (class_name, confidence, _) in classification_results.items():
            x, y, w, h = results['stats'][original_idx][:4]
            area = results['stats'][original_idx][4]
            
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
            cv2.rectangle(weed_crop_vis, (x, y), (x+w, y+h), color, 2)
            
            # Add class name and co     nfidence as text
            #label = f"{class_name[:10]} ({confidence:.2f})"
            #cv2.putText(weed_crop_vis, label, (x, y-5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            #if color == (0, 255, 0):
            #    cv2.putText(weed_crop_vis, label, (x, y-5),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    axes[5].imshow(cv2.cvtColor(weed_crop_vis, cv2.COLOR_BGR2RGB))
    axes[5].set_title(f'Weed vs Crop (Crops: {counts["crops"]}, Weeds: {counts["weeds"]})')
    axes[5].axis('off')
    
    # Adjust layout and save the figure
    plt.tight_layout()
    
    return {
        'fig': fig,
        'counts': counts,
        'weed_crop_visualization': weed_crop_vis  # Keep this for backward compatibility
    }

# ─── CLASSIFICATION FUNCTIONS ─────────────────────────────────────────────────────
def load_classification_model(model_path, device, classification_model_name):
    """Load the classification model from checkpoint"""
    if classification_model_name == "shufflenet":
        model = ShuffleNet(num_classes=83, device=device)
    elif classification_model_name == "mobilenet":
        model = MobileNetV2(num_classes=83, device=device)
    elif classification_model_name == "resnet18":
        model = ResNet18(num_classes=83, device=device)
    elif classification_model_name == "squeezenet":
        model = SqueezeNet(num_classes=83, device=device)
    elif classification_model_name == "shufflenet_se":
        model = ShuffleNetV2WithSE(num_classes=83)
    elif classification_model_name == "shufflenet_sepconv":
        model = ShuffleNetV2WithSepConv(num_classes=83)
    elif classification_model_name == "shufflenet_sepconv_se":
        model = ShuffleNetV2WithSepConvAndSE(num_classes=83)
    else:
        raise ValueError(f"Unknown classification model: {classification_model_name}")

    try:
        ckpt = torch.load(model_path, map_location=device)
        # Adjust key if necessary based on how the checkpoint was saved
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "state_dict" in ckpt:
             model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt)  # Assume ckpt is the state_dict itself
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {model_path}")
        raise
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        raise
    model.eval()
    return model

# Modified classify_image function
def classify_image(model, img, device):
    """
    Classify a single image using the loaded model.
    Returns predicted index, class name, and confidence.
    """
    # Convert PIL Image to tensor
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    tensor = inference_transforms(img).unsqueeze(0).to(device)

    class_name = "Unknown"  # Default
    idx = -1
    confidence = 0.0

    try:
        with torch.no_grad():
            out = model(tensor)
            probs = F.softmax(out, dim=1)
            confidence, idx_tensor = torch.max(probs, dim=1)
            idx = int(idx_tensor.item())
            confidence = float(confidence.item())

            if idx in ID_TO_NAME:
                class_name = ID_TO_NAME[idx]
            else:
                print(f"Warning: Predicted index {idx} not in ID_TO_NAME mapping.")
                class_name = f"Unknown_{idx}"

    except Exception as e:
        print(f"Error during classification: {e}")
        idx = -1
        class_name = "Error"
        confidence = 0.0

    return idx, class_name, confidence

# Modified classify_components
def classify_components(model_path, component_map, device='cpu', classification_model_name=CLASSIFICATION_MODEL):
    """
    Classify all component images.
    Returns:
    - counts: dictionary of class_name -> count
    - results: dictionary of original_component_idx -> (class_name, confidence, filename)
    """
    device = torch.device(device)
    model = load_classification_model(model_path, device, classification_model_name)
    counts = {}
    results = {}
    
    # Create reverse mapping from new component index to original index
    reverse_map = {new_idx: orig_idx for orig_idx, new_idx in component_map.items()}
    
    # Process each component image
    for filename in os.listdir(COMPONENTS_DIR):
        if not filename.startswith("component_"):
            continue
            
        # Parse the component number from filename (e.g., "component_001.png" -> 1)
        try:
            component_number = int(filename.split('_')[1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Skipping file with invalid format: {filename}")
            continue
            
        # Get the original component index
        if component_number not in reverse_map:
            print(f"Warning: Component {component_number} not found in component map")
            continue
            
        original_idx = reverse_map[component_number]
        image_path = os.path.join(COMPONENTS_DIR, filename)
        
        try:
            image = Image.open(image_path)
            idx, class_name, confidence = classify_image(model, image, device)
            
            # Record classification result
            if class_name is not None:
                counts[class_name] = counts.get(class_name, 0) + 1
                results[original_idx] = (class_name, confidence, filename)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    
    return counts, results

def plot_classification_results(counts, output_dir):
    """Plot the distribution of classified plants and save to output_dir"""
    if not counts:
        print("No classification results to plot.")
        return

    plt.figure(figsize=(max(12, len(counts) * 0.5), 6))
    # Sort by count descending
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    classes, vals = zip(*sorted_counts)

    plt.bar(classes, vals)
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.title('Plant Classification Distribution')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classification_distribution.png'))
    plt.close()

# ─── INTEGRATED MAIN FUNCTION ───────────────────────────────────────────────────
def process_aerial_image(image_path=IMAGE_PATH,
                         model_path=MODEL_PATH,
                         resize_dim=RESIZE_DIM,
                         output_dir=OUTPUT_DIR,
                         components_dir=COMPONENTS_DIR,
                         min_area_threshold=MIN_AREA_THRESHOLD,
                         exg_thresh=EXG_THRESH,
                         main_crop=MAIN_CROP,
                         embedding_model_name=EMBEDDING_MODEL,
                         classification_model_name=CLASSIFICATION_MODEL,
                         device=DEVICE):
    """
    Complete pipeline: Load, detect, extract, classify, visualize.
    """
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)

    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load and resize image
    print(f"Processing image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image: {image_path}")
        return None
    img_resized = cv2.resize(img, (resize_dim, resize_dim))

    # 2. Detect vegetation and leaves, cluster into plants
    print("Running vegetation detection and clustering...")
    results = segment_plants_by_exg_and_cc(
        img_resized,
        exg_thresh=exg_thresh
    )

    # 3. Extract significant components as individual images
    print("Extracting vegetation components...")
    num_components, component_map = extract_vegetation_components(
        img_resized, results, min_area_threshold
    )
    print(f"Extracted {num_components} vegetation components to {components_dir}")

    # 4. Classify components
    print(f"Classifying components using model: {model_path}")
    counts, classification_results = classify_components(
        model_path, component_map, device, classification_model_name
    )

    # Determine selected classes (Crops)
    selected_classes = []
    if main_crop:
        # Use Gemini (or placeholder) if main_crop isn't directly in the list (e.g., "Wheat")
        all_detected_classes = list(counts.keys())
        if main_crop not in NAME_TO_ID.keys():
            print(f"Main crop '{main_crop}' specified. Finding relevant classes using {embedding_model_name}...")
            
            # Select the appropriate compare_classes function based on EMBEDDING_MODEL
            if embedding_model_name == "slip":
                compare_classes_func = get_similar_classes_slip
            elif embedding_model_name == "clip":
                compare_classes_func = get_similar_classes_clip
            elif embedding_model_name == "gemini":
                compare_classes_func = get_similar_classes_gemini
            elif embedding_model_name == "imagebind":
                compare_classes_func = get_similar_classes_imagebind
            elif embedding_model_name == "openclip":
                compare_classes_func = get_similar_classes_openclip
            elif embedding_model_name == "llama":
                compare_classes_func = get_similar_classes_llama
            else:
                print(f"Warning: Unknown embedding model '{embedding_model_name}'. Defaulting to SLIP.")
                compare_classes_func = get_similar_classes_slip

            try:
                selected_classes = compare_classes_func(all_detected_classes, main_crop)
                if not isinstance(selected_classes, list) or not all(isinstance(item, str) for item in selected_classes):
                    print("Warning: API call did not return a valid list of strings. Defaulting to main_crop.")
                    selected_classes = [main_crop] if main_crop in all_detected_classes else []
            except Exception as api_err:
                print(f"Error during API call: {api_err}. Defaulting to main_crop.")
                selected_classes = [main_crop] if main_crop in all_detected_classes else []
        else:
            print(f"Using '{main_crop}' as the only crop class.")
            selected_classes = [main_crop]
    else:
        print("Warning: No 'main_crop' specified. Weed/Crop visualization will not be generated.")

    # 5. Generate visualizations using the classification results
    print("Creating visualizations...")
    viz = create_visualization(
        img_resized,
        results,
        min_area_threshold,
        classification_results,
        selected_classes
    )

    # Save the comprehensive debugging visualization figure
    debug_viz_path = os.path.join(output_dir, "debug_visualizations.png")
    viz['fig'].savefig(debug_viz_path, dpi=300, bbox_inches='tight')
    plt.close(viz['fig'])
    print(f"Saved comprehensive debug visualizations to: {debug_viz_path}")

    # Also save individual weed vs crop visualization for backward compatibility
    if viz['weed_crop_visualization'] is not None:
        weed_crop_path = os.path.join(output_dir, "05_weed_vs_crop.png")
        cv2.imwrite(weed_crop_path, viz['weed_crop_visualization'])
        print(f"Saved weed vs crop visualization to: {weed_crop_path}")

    # Plot classification distribution
    plot_classification_results(counts, output_dir)

    # Print summary statistics
    print("\nClassification summary:")
    if counts:
        sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        for cls, cnt in sorted_counts:
            print(f"- {cls}: {cnt}")
    else:
        print("No components were successfully classified.")

    return {
        "num_components": num_components,
        "class_counts": counts,
        "visualizations": ["debug_visualizations.png", "05_weed_vs_crop.png", "classification_distribution.png"]
    }
# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # --- Clean previous run ---
    print(f"Cleaning up previous run directories...")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(COMPONENTS_DIR, ignore_errors=True) # Clean temp dir
    shutil.rmtree(CLASSIFIED_DIR, ignore_errors=True) # Clean classified output

    # --- Run Pipeline ---
    print("\nStarting aerial image processing pipeline...")
    pipeline_results = process_aerial_image(embedding_model_name=EMBEDDING_MODEL, classification_model_name=CLASSIFICATION_MODEL)
