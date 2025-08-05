# Helper functions for classification tasks

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from classification_models.MobileNet import MobileNetV2
from classification_models.ResNet18 import ResNet18
from classification_models.ShuffleNet import ShuffleNet
from classification_models.ShuffleNet_SE import ShuffleNetV2WithSE
from classification_models.ShuffleNet_SEPCONV import ShuffleNetV2WithSepConv
from classification_models.ShuffleNet_SEPCONV_SE import ShuffleNetV2WithSepConvAndSE
from classification_models.SqueezeNet import SqueezeNet

from class_mapping import ID_TO_NAME, NAME_TO_ID

# --- Hyperparameters ---
COMPONENTS_DIR = "temp_veg_components" # This is still needed here for classify_components

inference_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.25, 0.25, 0.25]),
])

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
def classify_components(model_path, component_map, device='cpu', classification_model_name="shufflenet", components_dir=COMPONENTS_DIR):
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
    for filename in os.listdir(components_dir):
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
        image_path = os.path.join(components_dir, filename)
        
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
