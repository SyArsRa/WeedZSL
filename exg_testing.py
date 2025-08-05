import cv2
import numpy as np
import os
import glob
from pathlib import Path

from segmentation import generate_exg_vegetation_mask

# Default paths for evaluation
IMAGE_DIR = r"PhenoBench-v110\PhenoBench\train\images"
MASK_DIR = r"PhenoBench-v110\PhenoBench\train\plant_visibility"

# Default ExG thresholds to test
THRESHOLDS_TO_TEST = [5, 10, 20, 30, 40]

def pixel_accuracy(mask1, mask2):
    """Calculate pixel-based accuracy between two binary masks."""
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    
    # Convert to binary if needed
    mask1 = (mask1 > 0).astype(int)
    mask2 = (mask2 > 0).astype(int)
    
    # Calculate accuracy
    correct_pixels = np.sum(mask1 == mask2)
    total_pixels = mask1.size
    
    return correct_pixels / total_pixels

def comprehensive_metrics(pred_mask, true_mask):
    """Calculate comprehensive set of segmentation metrics."""
    # Ensure masks are the same shape
    assert pred_mask.shape == true_mask.shape, "Masks must have the same shape"
    
    # Convert to binary
    pred = (pred_mask > 0).astype(int)
    true = (true_mask > 0).astype(int)
    
    # Calculate confusion matrix components
    tp = np.sum((pred == 1) & (true == 1))  # True positives
    tn = np.sum((pred == 0) & (true == 0))  # True negatives
    fp = np.sum((pred == 1) & (true == 0))  # False positives
    fn = np.sum((pred == 0) & (true == 1))  # False negatives
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
 
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'iou': iou,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn
    }

def find_matching_files(image_dir, mask_dir):
    """Find matching image and mask files."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    mask_extensions = ['.png', '.PNG']
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))
    
    mask_files = []
    for ext in mask_extensions:
        mask_files.extend(glob.glob(os.path.join(mask_dir, f"*{ext}")))
    
    # Create dictionaries for easier matching
    image_dict = {}
    for img_path in image_files:
        basename = Path(img_path).stem
        image_dict[basename] = img_path
    
    mask_dict = {}
    for mask_path in mask_files:
        basename = Path(mask_path).stem
        mask_dict[basename] = mask_path
    
    # Find matches
    matches = []
    for basename in image_dict.keys():
        if basename in mask_dict:
            matches.append((image_dict[basename], mask_dict[basename]))
    
    return matches

def evaluate_vegetation_detection(image_dir, mask_dir, exg_thresh, output_dir="evaluation_results"):
    """Evaluate vegetation detection against ground truth masks."""
    
    # Find matching files
    matches = find_matching_files(image_dir, mask_dir)
    
    if not matches:
        print("No matching image-mask pairs found!")
        return None, None
    
    print(f"Found {len(matches)} matching image-mask pairs")
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_metrics = []
    
    for i, (image_path, mask_path) in enumerate(matches):
        print(f"Processing {i+1}/{len(matches)}: {Path(image_path).name}")
        
        # Load image and ground truth mask
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue
            
        gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt_mask is None:
            print(f"Could not load mask: {mask_path}")
            continue
        
        # Resize image to match mask if needed
        if img.shape[:2] != gt_mask.shape:
            img = cv2.resize(img, (gt_mask.shape[1], gt_mask.shape[0]))
        
        # Generate prediction using ExG
        pred_mask = generate_exg_vegetation_mask(img, exg_thresh)
        
        # Calculate metrics
        metrics = comprehensive_metrics(pred_mask, gt_mask)
        metrics['image_name'] = Path(image_path).name
        all_metrics.append(metrics)
        
        # Save visualization
        if i < 10:  # Save first 10 for visualization
            # Create comparison image
            comparison = np.zeros((img.shape[0], img.shape[1] * 3, 3), dtype=np.uint8)
            
            # Original image
            comparison[:, :img.shape[1]] = img
            
            # Ground truth (green)
            gt_colored = np.zeros_like(img)
            gt_colored[gt_mask > 0] = [0, 255, 0]
            comparison[:, img.shape[1]:img.shape[1]*2] = gt_colored
            
            # Prediction (red) and overlap (yellow)
            pred_colored = np.zeros_like(img)
            pred_colored[pred_mask > 0] = [0, 0, 255]
            # Overlap in yellow
            overlap = (pred_mask > 0) & (gt_mask > 0)
            pred_colored[overlap] = [0, 255, 255]
            comparison[:, img.shape[1]*2:] = pred_colored
            
            # Add text labels
            #font = cv2.FONT_HERSHEY_TRIPLEX
            #cv2.putText(comparison, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            #cv2.putText(comparison, 'Ground Truth', (img.shape[1] + 10, 30), font, 1, (255, 255, 255), 2)
            #cv2.putText(comparison, 'Prediction', (img.shape[1]*2 + 10, 30), font, 1, (255, 255, 255), 2)
            
            # Add metrics text
            #text_y = 60
            #cv2.putText(comparison, f"Acc: {metrics['accuracy']:.3f}", (10, text_y), font, 0.7, (255, 255, 255), 2)
            #cv2.putText(comparison, f"IoU: {metrics['iou']:.3f}", (10, text_y + 30), font, 0.7, (255, 255, 255), 2)
            #cv2.putText(comparison, f"Dice: {metrics['dice']:.3f}", (10, text_y + 60), font, 0.7, (255, 255, 255), 2)
            #cv2.putText(comparison, f"Prec: {metrics['precision']:.3f}", (10, text_y + 90), font, 0.7, (255, 255, 255), 2)
            #cv2.putText(comparison, f"Rec: {metrics['recall']:.3f}", (10, text_y + 120), font, 0.7, (255, 255, 255), 2)
            
            output_path = os.path.join(output_dir, f"comparison_{i+1:03d}_{Path(image_path).stem}.png")
            cv2.imwrite(output_path, comparison)
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {}
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'iou', 'dice']
        
        for key in metric_keys:
            values = [m[key] for m in all_metrics]
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Print results
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Total images processed: {len(all_metrics)}")
        print(f"ExG threshold used: {exg_thresh}")
        print("\nAverage Metrics:")
        print("-" * 40)
        
        for key in metric_keys:
            print(f"{key.upper():12}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f} "
                  f"(min: {avg_metrics[key]['min']:.4f}, max: {avg_metrics[key]['max']:.4f})")
        
        # Save detailed results to file
        results_file = os.path.join(output_dir, "detailed_results.txt")
        with open(results_file, 'w') as f:
            f.write("Vegetation Detection Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total images processed: {len(all_metrics)}\n")
            f.write(f"ExG threshold used: {exg_thresh}\n\n")
            
            f.write("Per-image results:\n")
            f.write("-" * 30 + "\n")
            for m in all_metrics:
                f.write(f"{m['image_name']}: Acc={m['accuracy']:.3f}, IoU={m['iou']:.3f}, "
                       f"Dice={m['dice']:.3f}, Prec={m['precision']:.3f}, Rec={m['recall']:.3f}\n")
            
            f.write("\nAverage Metrics:\n")
            f.write("-" * 20 + "\n")
            for key in metric_keys:
                f.write(f"{key}: {avg_metrics[key]['mean']:.4f} ± {avg_metrics[key]['std']:.4f}\n")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Visualization images saved to: {output_dir}")
        
        return avg_metrics, all_metrics
    
    else:
        print("No valid results obtained!")
        return None, None

if __name__ == "__main__":
    # Set your paths here
    image_dir = IMAGE_DIR
    mask_dir = MASK_DIR

    # Test different ExG thresholds
    thresholds = THRESHOLDS_TO_TEST

    best_threshold = None
    best_score = 0
    threshold_results = {}
    
    for thresh in thresholds:
        print(f"\n{'='*60}")
        print(f"Testing ExG threshold: {thresh}")
        print('='*60)
        
        output_dir = f"evaluation_results_exg_{thresh}"
        avg_metrics, _ = evaluate_vegetation_detection(image_dir, mask_dir, thresh, output_dir)
        
        if avg_metrics:
            # Store results for this threshold
            threshold_results[thresh] = avg_metrics
            
            # Use IoU as the primary metric for comparison
            iou_score = avg_metrics['iou']['mean']
            if iou_score > best_score:
                best_score = iou_score
                best_threshold = thresh
    
    # Print summary of all thresholds
    print(f"\n{'='*80}")
    print("SUMMARY OF ALL THRESHOLDS")
    print('='*80)
    print(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10} {'Dice':<10}")
    print('-' * 80)
    
    for thresh in thresholds:
        if thresh in threshold_results:
            metrics = threshold_results[thresh]
            print(f"{thresh:<10} {metrics['accuracy']['mean']:<10.4f} {metrics['precision']['mean']:<10.4f} "
                  f"{metrics['recall']['mean']:<10.4f} {metrics['f1_score']['mean']:<10.4f} "
                  f"{metrics['iou']['mean']:<10.4f} {metrics['dice']['mean']:<10.4f}")
    
    if best_threshold:
        print(f"\n{'='*60}")
        print(f"BEST THRESHOLD: {best_threshold} (IoU: {best_score:.4f})")
        print('='*60)
        
        # Print detailed stats for best threshold
        best_metrics = threshold_results[best_threshold]
        print(f"\nDetailed metrics for best threshold ({best_threshold}):")
        print("-" * 50)
        for key in ['accuracy', 'precision', 'recall', 'f1_score', 'iou', 'dice']:
            m = best_metrics[key]
            print(f"{key.upper():12}: {m['mean']:.4f} ± {m['std']:.4f} "
                  f"(min: {m['min']:.4f}, max: {m['max']:.4f})")
    
    # Save threshold comparison to file
    comparison_file = "threshold_comparison.txt"
    with open(comparison_file, 'w') as f:
        f.write("ExG Threshold Comparison Results\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Threshold':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'IoU':<10} {'Dice':<10}\n")
        f.write('-' * 80 + "\n")
        
        for thresh in thresholds:
            if thresh in threshold_results:
                metrics = threshold_results[thresh]
                f.write(f"{thresh:<10} {metrics['accuracy']['mean']:<10.4f} {metrics['precision']['mean']:<10.4f} "
                       f"{metrics['recall']['mean']:<10.4f} {metrics['f1_score']['mean']:<10.4f} "
                       f"{metrics['iou']['mean']:<10.4f} {metrics['dice']['mean']:<10.4f}\n")
        
        if best_threshold:
            f.write(f"\nBest threshold: {best_threshold} (IoU: {best_score:.4f})\n")
    
    print(f"\nThreshold comparison saved to: {comparison_file}")