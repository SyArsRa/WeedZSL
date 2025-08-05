import os
import shutil
import cv2
import torch
import matplotlib.pyplot as plt

from segmentation import segment_plants_by_exg_and_cc, extract_vegetation_components, create_visualization

#from embedding_models.embeddings_slip import get_similar_classes_slip
#from embedding_models.embedding_clip import get_similar_classes_clip
from embedding_models.embeddings_gemini import get_similar_classes_gemini
#from embedding_models.embeddings_imagebind import get_similar_classes_imagebind
#from embedding_models.embeddings_openclip import get_similar_classes_openclip
#from embedding_models.embeddings_llama import get_similar_classes_llama

from classification import classify_components, plot_classification_results # Import moved functions

from class_mapping import NAME_TO_ID
# --- Hyperparameters ( Change as Needed ) ---
IMAGE_PATH = r"wheat\DJI_202412191545_004\DJI_20241219154909_0003_D.JPG"
MODEL_PATH = "models_weights\shufflenet.pt"
RESIZE_DIM = 1000
OUTPUT_DIR = "results_output"
COMPONENTS_DIR = "temp_veg_components"
CLASSIFIED_DIR = "classified_plants_output"
MIN_AREA_THRESHOLD = 50
EXG_THRESH = 30
ASPECT_RATIO_RANGE = (0.15, 6.0)
MAIN_CROP = "Wheat"
EMBEDDING_MODEL = "gemini" # Options: "slip", "clip", "gemini", "imagebind", "openclip", "llama"
CLASSIFICATION_MODEL = "shufflenet" # Options: "shufflenet", "mobilenet", "resnet18", "squeezenet", "shufflenet_se", "shufflenet_sepconv", "shufflenet_sepconv_se"
DEVICE = None # "cuda" if torch.cuda.is_available() else "cpu"


# ─── INTEGRATED MAIN FUNCTION ───────────────────────────────────────────────────
def process_image(image_path=IMAGE_PATH,
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
        weed_crop_path = os.path.join(output_dir, "weed_vs_crop.png")
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
        "visualizations": ["debug_visualizations.png", "weed_vs_crop.png", "classification_distribution.png"]
    }

# ─── MAIN EXECUTION ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    # --- Clean previous run ---
    print(f"Cleaning up previous run directories...")
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    shutil.rmtree(COMPONENTS_DIR, ignore_errors=True) # Clean temp dir
    shutil.rmtree(CLASSIFIED_DIR, ignore_errors=True) # Clean classified output

    # --- Run Pipeline ---
    print("\nStarting pipeline...")
    pipeline_results = process_image(embedding_model_name=EMBEDDING_MODEL, classification_model_name=CLASSIFICATION_MODEL)