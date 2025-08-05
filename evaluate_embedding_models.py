import glob
import traceback
import json
import os
import shutil
import cv2
import numpy as np
import torch

from PIL import Image
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd



from classification import classify_components, load_classification_model, inference_transforms
from segmentation import vegetation_mask_exg, letterbox_resize
from class_mapping import ID_TO_NAME, NAME_TO_ID
from embedding_models.embeddings_gemini import get_similar_classes_gemini
from embedding_models.embedding_clip import get_similar_classes_clip
from embedding_models.embeddings_imagebind import get_similar_classes_imagebind
from embedding_models.embeddings_llama import get_similar_classes_llama
from embedding_models.embeddings_openclip import get_similar_classes_openclip
from embedding_models.embeddings_slip import get_similar_classes_slip

# Hyperparameters
DATASET_PATH = r"unseen"
MODEL_PATH = r"models\sesc.pt"
CLASSIFICATION_MODEL_NAME = "shufflenet_sepconv_se"
EMBEDDING_MODEL_NAME = "gemini"



    
# ─── VISUALIZATION UTILITIES ─────────────────────────────────────────────────────
def create_debug_visualizations(img, class_name, confidence, metric, debug_dir='debug_images'):
    """Create and save debug visualizations for single image classification"""
    os.makedirs(f"{debug_dir}/{class_name}/{random.randint(0, 10000)}", exist_ok=True)

    # 1. Original image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    orig_path = os.path.join(debug_dir, 'original.png')
    plt.savefig(orig_path, bbox_inches='tight')
    plt.close()
    
    # 2. Classification result visualization
    class_vis = img.copy()
    h, w = img.shape[:2]
    
    # Determine if this class is selected
    is_selected = metric == 'TP'
    color = (0, 255, 0) if is_selected else (0, 0, 255)  # Green if selected, red otherwise
    
    # Draw border around entire image
    cv2.rectangle(class_vis, (0, 0), (w-1, h-1), color, 5)
    
    # Add text with classification result
    text = f"{class_name} ({confidence:.2f})"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 30
    
    # Add background rectangle for text
    cv2.rectangle(class_vis, (text_x - 5, text_y - text_size[1] - 5), 
                 (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
    cv2.putText(class_vis, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB))
    plt.title('Classification Result')
    plt.axis('off')
    class_path = os.path.join(debug_dir, 'classification.png')
    plt.savefig(class_path, bbox_inches='tight')
    plt.close()
    
    # 3. Combined visualization
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB))
    plt.title(f'Classification: {class_name} ({confidence:.2f})')
    plt.axis('off')
    
    plt.tight_layout()
    combined_path = os.path.join(debug_dir, 'combined_debug.png')
    plt.savefig(combined_path, bbox_inches='tight')
    plt.close()
    
    return {
        'original': orig_path,
        'classification': class_path,
        'combined': combined_path
    }

 





# ─── SELECTION & MARKING ─────────────────────────────────────────────────────────
def select_classes(detected, main_crop, threshold, embedding_model_name):
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
            print(f"Warning: Unknown embedding model '{embedding_model_name}'. Defaulting to Gemini.")
            compare_classes_func = get_similar_classes_gemini

        try:
            sel = compare_classes_func(list(detected), main_crop, threshold=threshold)
            if isinstance(sel, list): 
                return sel
        except Exception as api_err:
            print(f"Error during API call: {api_err}. Defaulting to simple string matching.")
            pass
    
    # Fallback: simple string matching
    return [c for c in detected if main_crop.lower() in c.lower()]

def evaluate_classification(class_name, selected_classes, label, main_crop):
    """Evaluate classification result with only TP and FP."""

    label = label == main_crop
    if class_name in selected_classes:
        if label == 1:
            return 'TP'
        else:
            return 'FP'
    else:
        if label == 1:
            return 'FN'
        else:
            return 'TN'

 

def save_class_distribution_barplot_for_main_crop(main_crop, all_outputs, save_path="output/class_distribution_main_crop", selected=[]):
    """
    Saves a bar plot of class frequencies (label == 1 only).
    Green bars = selected classes, blue bars = others.
    """
    # Filter only label == 1
    label_1_outputs = [all_outputs[output] for output in all_outputs if all_outputs[output]["label"] == True]
    
    # Count frequency of each class in label == 1
    global_counts = {}
    for output in label_1_outputs:
        cls = output["class"]
        global_counts[cls] = global_counts.get(cls, 0) + 1

    # Determine selected classes (thresholded)
    values = list(global_counts.values())
    if len(values) == 0:
        print("No classes with label==1 found.")
        return

    # Sort classes
    sorted_classes = sorted(global_counts, key=global_counts.get, reverse=True)
    sorted_counts = [global_counts[cls] for cls in sorted_classes]
    sorted_colors = ['green' if cls in selected else 'blue' for cls in sorted_classes]

    # Plot
    plt.figure(figsize=(20, 10))
    plt.bar(sorted_classes, sorted_counts, color=sorted_colors)
    
    #add frequency as text to each bar
    for i, count in enumerate(sorted_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=90)
    plt.xlabel("Class")
    plt.ylabel("Frequency (label == 1)")
    plt.title("Main Crop Class Frequency (Green = Selected, Blue = Not Selected)")
    plt.tight_layout()
    plt.savefig(save_path+f"_{main_crop}.png", bbox_inches='tight')
    plt.close()

# ─── MAIN BATCH PIPELINE ────────────────────────────────────────────────────────
def process_batch(
    images_data,
    model_path,
    output_dir='results',
    debug_dir='debug_images',
    classification_model_name="shufflenet"
):
    """
    Process a batch of images for classification without segmentation.
    Each image is treated as a single component.
    """
    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = load_classification_model(model_path, device, classification_model_name)

    # Storage
    per_image_data = {}
    global_counts = {}

    # Create a temporary directory for images
    temp_image_dir = "temp_eval_images"
    os.makedirs(temp_image_dir, exist_ok=True)

    # Save images to the temporary directory
    print("Saving images to temporary directory...")
    component_map = {}
    temp_per_image_data = {}
    for i, (img_path, label) in enumerate(images_data):
        if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        name = os.path.splitext(os.path.basename(img_path))[0]
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            continue

        img_bgr = letterbox_resize(img, target_size=(1000, 1000), color=(0, 0, 0))
        img = vegetation_mask_exg(img_bgr, 28)
        img = cv2.bitwise_and(img_bgr, img_bgr, mask=img)
        
        temp_file_path = os.path.join(temp_image_dir, f"component_{i}.png")
        cv2.imwrite(temp_file_path, img)
        component_map[i] = i # Map original index to new index
        temp_per_image_data[i] = {'label': label, 'image': img, 'original_name': name} # Store original image and label, and original name

    # Classify components using the shared function
    print("Classifying images using classify_components...")
    _, classification_results = classify_components(
        model_path, component_map, device, classification_model_name, temp_image_dir
    )

    # Populate per_image_data with classification results
    final_per_image_data = {}
    global_counts = {}
    for original_idx, (class_name, confidence, filename) in classification_results.items():
        if original_idx in temp_per_image_data:
            original_data = temp_per_image_data[original_idx]
            final_per_image_data[original_data['original_name']] = {
                'image': original_data['image'],
                'class_name': class_name,
                'confidence': confidence,
                'class_idx': NAME_TO_ID.get(class_name, -1), # Get ID from name
                'label': original_data['label'],
            }
            global_counts[class_name] = global_counts.get(class_name, 0) + 1 # Update global counts based on classified images

    # Clean up temporary directory
    shutil.rmtree(temp_image_dir)
    
    return global_counts, final_per_image_data, debug_dir, output_dir

def threshold_selection(main_crop, threshold, global_counts, per_image_data, debug_dir, output_dir, embedding_model_name="gemini"):
    # Select classes globally
    detected = list(global_counts.keys())
    selected = select_classes(detected, main_crop, threshold, embedding_model_name) if main_crop else detected
    print(f"Selected classes: {selected}")

    # Metrics accumulator
    metrics = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    summary = {}
    metric_colors = {
    'TP': (0, 255, 0),    # Green
    'FP': (255, 255, 0),  # Yellow  
    'FN': (255, 0, 0),    # Red
    'TN': (0, 0, 255)     # Blue
    }
    # Second pass: evaluate and create visualizations
    print("\nSecond pass: Evaluating and creating visualizations...")
    plt.ioff()

    counter = 0
    for name, data in per_image_data.items():
        img = data['image']
        class_name = data['class_name']
        confidence = data['confidence']
        label = data['label']

        # Evaluate classification (always do this for metrics)
        metric_result = evaluate_classification(class_name, selected, label, main_crop)
        metrics[metric_result] += 1

        # Store summary for this image (always do this)
        summary[name] = {
            'class': class_name,
            'confidence': float(confidence),  # Convert to Python float
            'selected': metric_result in ['TP', 'FP'],
            'metric': metric_result,
            'label': label == main_crop
        }

        # Only create visualizations for every 10th sample (10%)
        #turnn off for now
        if False:
            print(f"Creating visualizations for {name}...")

            # Create debug visualizations
            debug_paths = create_debug_visualizations(
                img, class_name, confidence, 
                metric_result in ['TP', 'FP'], 
                debug_dir=debug_dir
            )

            # Pre-compute values
            h, w = img.shape[:2]
            color = metric_colors[metric_result]
            border_color = (0, 255, 0) if metric_result in ['TP', 'FP'] else (0, 0, 255)

            # Create metrics visualization
            metrics_vis = img.copy()
            cv2.rectangle(metrics_vis, (0, 0), (w-1, h-1), color, 5)
            cv2.putText(metrics_vis, f"{metric_result}: {class_name}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Create class visualization
            class_vis = img.copy()
            cv2.rectangle(class_vis, (0, 0), (w-1, h-1), border_color, 5)
            
            # Convert images to RGB once for matplotlib
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            class_vis_rgb = cv2.cvtColor(class_vis, cv2.COLOR_BGR2RGB)
            metrics_vis_rgb = cv2.cvtColor(metrics_vis, cv2.COLOR_BGR2RGB)
            
            # Create final summary visualization using subplots for efficiency
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
            
            axes[0, 0].imshow(img_rgb)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(class_vis_rgb)
            axes[0, 1].set_title(f'Classification: {class_name}\n(Confidence: {confidence:.3f})')
            axes[0, 1].axis('off')
            
            axes[1, 0].imshow(metrics_vis_rgb)
            axes[1, 0].set_title(f'Evaluation: {metric_result}')
            axes[1, 0].axis('off')
            
            # Add statistics text
            axes[1, 1].axis('off')
            stats_text = (f"Image: {name}\n"
                        f"Predicted: {class_name}\n"
                        f"Confidence: {confidence:.3f}\n"
                        f"Selected: {'Yes' if metric_result in ['TP', 'FP'] else 'No'}\n"
                        f"Label: {label}\n"
                        f"Metric: {metric_result}")
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center')
            
            # Hide unused subplots
            axes[0, 2].axis('off')
            axes[1, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(debug_dir, 'final_summary.png'), bbox_inches='tight')
            plt.close(fig)  # Explicitly close figure to free memory
            
            # Save CV2 images
            cv2.imwrite(os.path.join(debug_dir, 'metrics_visualization.png'), metrics_vis)
            cv2.imwrite(os.path.join(debug_dir, f"vis_{name}.jpeg"), class_vis)
        
        counter += 1# Save summary & metrics

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    # Create overall summary visualization
    plt.figure(figsize=(16, 12))
    
    # Plot class distribution
    plt.subplot(2, 2, 1)
    classes = list(global_counts.keys())
    counts = [global_counts[cls] for cls in classes]
    colors = ['green' if cls in selected else 'red' for cls in classes]
    
    plt.bar(classes, counts, color=colors)
    plt.title('Class Distribution (Green = Selected, Red = Not Selected)')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    
    # Plot metrics
    plt.subplot(2, 2, 2)
    metrics_keys = ['TP', 'FP', 'FN', 'TN']
    metrics_values = [metrics[k] for k in metrics_keys]
    metrics_colors = ['green', 'yellow', 'red', 'blue']
    
    plt.bar(metrics_keys, metrics_values, color=metrics_colors)
    plt.title('Overall Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Count')
    
    # Calculate and display performance metrics
    plt.subplot(2, 2, 3)
    plt.axis('off')
    
    total = sum(metrics.values())
    if total > 0:
        accuracy = (metrics['TP'] + metrics['TN']) / total
        if metrics['TP'] + metrics['FP'] > 0:
            precision = metrics['TP'] / (metrics['TP'] + metrics['FP'])
        else:
            precision = 0
        if metrics['TP'] + metrics['FN'] > 0:
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        else:
            recall = 0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0
        
        perf_text = f"Performance Metrics:\n\n"
        perf_text += f"Total Images: {total}\n"
        perf_text += f"Accuracy: {accuracy:.3f}\n"
        perf_text += f"Precision: {precision:.3f}\n"
        perf_text += f"Recall: {recall:.3f}\n"
        perf_text += f"F1 Score: {f1_score:.3f}\n\n"
        perf_text += f"Selected Classes: {len(selected)}\n"
        perf_text += f"Total Classes: {len(classes)}\n"
        
        plt.text(0.1, 0.5, perf_text, fontsize=12, verticalalignment='center')
    
    # Show selected classes
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    selected_text = "Selected Classes:\n\n"
    for cls in selected:
        count = global_counts.get(cls, 0)
        selected_text += f"• {cls}: {count} images\n"
    
    plt.text(0.1, 0.5, selected_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overall_summary.png'), bbox_inches='tight')
    plt.close()

    save_class_distribution_barplot_for_main_crop(main_crop,summary, "output/main_crop_distribution.png", selected=selected)
    print(f"\nProcessing complete!")
    print(f"Total images processed: {len(per_image_data)}")
    print(f"Selected classes: {selected}")
    print(f"Metrics: {metrics}")


    return summary, metrics

def load_images_with_binary_labels(dataset_root):
    image_data = []

    # Traverse all class folders
    for class_entry in os.scandir(dataset_root):
        if class_entry.is_dir():
            class_name = class_entry.name
            image_paths = glob.glob(os.path.join(class_entry.path, "*.*"))

            # Label as True if this is the target class
            label = class_name

            for img_path in image_paths:
                image_data.append((img_path, label))

    return image_data

def evaluate_and_plot_f1(path, model_path, classification_model_name="shufflenet", embedding_model_name="gemini"):
    all_results = []
    crops = os.listdir(path)
    image_data = load_images_with_binary_labels(
            #dataset_root=r"unseen",
            dataset_root=path,
        )

    global_counts, per_image_data, debug_dir, output_dir = process_batch(
            images_data=image_data,
            model_path=model_path,
            output_dir="output",
            debug_dir="debug",
            classification_model_name=classification_model_name
    )

    def run_crop_evaluation(crop_name: str):
        print(f"\n===== Processing Crop: {crop_name} =====")

        for threshold in np.arange(0.00, 0.11, 0.01):
            print(f"\n--- Running for Threshold: {threshold:.2f} ({crop_name}) ---")
            summary, metrics = threshold_selection(
                crop_name, threshold, global_counts, per_image_data, debug_dir, output_dir
            )

            TP = metrics.get("TP", 0)
            FP = metrics.get("FP", 0)
            FN = metrics.get("FN", 0)
            TN = metrics.get("TN", 0)

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

            total = TP + FP + FN + TN
            accuracy = (TP + TN) / total if total > 0 else 0

            all_results.append({
                "Crop": crop_name,
                "Threshold": threshold,
                "F1 Score": f1,
                "Accuracy": accuracy,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN
            })
            break

    # Run for each crop
    for crop in crops:
        run_crop_evaluation(crop)

    # Plot F1 vs Threshold
    df = pd.DataFrame(all_results)

    sns.set(style="whitegrid", context="talk")
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="Threshold", y="F1 Score", hue="Crop", marker="o", palette="Set2")

    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.xticks(np.arange(0.0, 0.11, 0.01))
    plt.ylim(0, 1)
    plt.legend(title="Crop")
    plt.tight_layout()
    plt.show()

    return df  # Return dataframe in case user wants to save or inspect results


df = evaluate_and_plot_f1(
    path=DATASET_PATH,
    model_path=MODEL_PATH,
    classification_model_name=CLASSIFICATION_MODEL_NAME,
    embedding_model_name=EMBEDDING_MODEL_NAME
)
# Save the results to a CSV file
df.to_csv("f1_results.csv", index=False)