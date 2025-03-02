import os
import re
import time
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score

# NOTE: exclude BIRD class
CLASS_MAPPING = {
    0: "AIRPLANE",
    1: "DRONE",
    2: "HELICOPTER"
}

def view_df_summary(df):
    print("Dataset Summary:")
    print(f"Total number of objects: {len(df)}")
    print(f"Total number of unique images: {df['image_file'].nunique()}")

    print("\nClass distribution:")
    print(df['class'].value_counts())

    print("\nImage dimensions summary:")
    print(df[['image_width', 'image_height']].describe())

def get_matching_files(images_path, labels_path):
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]

    image_files = [f for f in image_files if 'BIRD' not in f]
    label_files = [f for f in label_files if 'BIRD' not in f]

    image_map = {os.path.splitext(f)[0]: f for f in image_files}
    label_map = {os.path.splitext(f)[0]: f for f in label_files}

    common_keys = set(image_map.keys()) & set(label_map.keys())

    matched_files = [(image_map[k], label_map[k]) for k in common_keys]
    return matched_files

def extract_info_from_filename(filename):
    match = re.match(r'V_([A-Z]+)_(\d+_\d+)_', filename)
    if match:
        class_name, image_id = match.groups()
        return class_name, image_id
    return None, None

def read_label_file(label_path):
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append({
                'class_id': int(class_id),
                'class_name': CLASS_MAPPING[int(class_id)],
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height
            })
        return boxes
    except Exception as e:
        print(f"Error reading {label_path}: {e}")
        return []

def create_dataset(images_path, labels_path):
    matched_files = get_matching_files(images_path, labels_path)

    data = []
    for img_file, label_file in matched_files:
        class_name, image_id = extract_info_from_filename(img_file)

        img_path = os.path.join(images_path, img_file)
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        label_path = os.path.join(labels_path, label_file)
        boxes = read_label_file(label_path)

        for box in boxes:
            if box["class_name"] != class_name:
                print(f"Found mismatch between filename class and detected class: {image_id}")

            data.append({
                'image_file': img_file,
                'label_file': label_file,
                'image_id': image_id,
                'class': box['class_name'],
                'x_center': box['x_center'],
                'y_center': box['y_center'],
                'width': box['width'],
                'height': box['height'],
                'image_width': img_width,
                'image_height': img_height,
                'image_path': img_path,
                'label_path': label_path
            })

    return pd.DataFrame(data)

# Metrics

def calculate_iou(pred_box, target_box):
    """Calculate IoU between single predicted and target box"""
    # Convert center format to corners
    pred_x1 = pred_box[0] - pred_box[2] / 2
    pred_y1 = pred_box[1] - pred_box[3] / 2
    pred_x2 = pred_box[0] + pred_box[2] / 2
    pred_y2 = pred_box[1] + pred_box[3] / 2

    target_x1 = target_box[0] - target_box[2] / 2
    target_y1 = target_box[1] - target_box[3] / 2
    target_x2 = target_box[0] + target_box[2] / 2
    target_y2 = target_box[1] + target_box[3] / 2

    # Calculate intersection
    x1 = max(pred_x1, target_x1)
    y1 = max(pred_y1, target_y1)
    x2 = min(pred_x2, target_x2)
    y2 = min(pred_y2, target_y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    union = pred_area + target_area - intersection

    return intersection / (union + 1e-6)

def calculate_map(pred_boxes, pred_scores, target_boxes, iou_threshold=0.5):
    """Calculate mAP for the predictions"""
    aps = []

    for class_id in range(3):  # For each class (AIRPLANE, DRONE, HELICOPTER)
        precisions = []
        recalls = []

        # Get predictions for this class
        class_preds = [i for i, score in enumerate(pred_scores) if torch.argmax(score) == class_id]
        class_targets = [i for i, box in enumerate(target_boxes) if box[0] == class_id]

        if len(class_targets) == 0:
            continue

        # Calculate precision and recall at different confidence thresholds
        for threshold in np.arange(0, 1, 0.1):
            tp = fp = fn = 0

            for pred_idx in class_preds:
                pred_box = pred_boxes[pred_idx]
                max_iou = 0
                best_target_idx = -1

                # Find best matching target box
                for target_idx in class_targets:
                    target_box = target_boxes[target_idx][1:]  # Remove class_id
                    iou = calculate_iou(pred_box, target_box)
                    if iou > max_iou:
                        max_iou = iou
                        best_target_idx = target_idx

                if max_iou >= iou_threshold and pred_scores[pred_idx][class_id] >= threshold:
                    tp += 1
                else:
                    fp += 1

            fn = len(class_targets) - tp

            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)

            precisions.append(precision)
            recalls.append(recall)

        if len(precisions) > 0:
            ap = np.trapz(precisions, recalls)
            aps.append(ap)

    return np.mean(aps) if len(aps) > 0 else 0

def evaluate_model(model, dataset, num_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32 (4 bytes)

    ious = []
    f1_scores = []
    inference_times = []
    pred_boxes_all = []
    pred_scores_all = []
    target_boxes_all = []

    print("\nEvaluating model on", num_samples, "samples...")

    with torch.no_grad():
        for i in range(num_samples):
            image, target_boxes = dataset[i]
            image = image.unsqueeze(0).to(device)
            target_boxes = target_boxes.to(device)

            # Measure inference time
            start_time = time.time()
            pred_scores, pred_boxes = model(image)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Store predictions and targets
            pred_boxes_all.append(pred_boxes[0])
            pred_scores_all.append(pred_scores[0])
            target_boxes_all.append(target_boxes[0])

            # Calculate IoU for valid boxes
            if target_boxes[0, 0] != -1:  # If there is a valid target box
                iou = calculate_iou(pred_boxes[0, 0].cpu(), target_boxes[0, 1:].cpu())
                ious.append(iou)

            # Calculate F1 Score
            pred_class = torch.argmax(pred_scores[0, 0]).cpu()
            true_class = target_boxes[0, 0].cpu()
            if true_class != -1:
                f1_scores.append(pred_class == true_class)

    # Calculate mAP
    map_score = calculate_map(pred_boxes_all, pred_scores_all, target_boxes_all)

    # Print results
    print("\nModel Evaluation Metrics:")
    print("-" * 50)
    print(f"Model Size: {model_size_mb:.2f} MB ({total_params:,} parameters)")
    print(f"Average Inference Latency: {np.mean(inference_times)*1000:.2f} ms")
    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"mAP: {map_score:.4f}")
    print(f"F1 Score: {np.mean(f1_scores):.4f}")

    return {
        'model_size': model_size_mb,
        'inference_latency': np.mean(inference_times),
        'mean_iou': np.mean(ious),
        'map': map_score,
        'f1_score': np.mean(f1_scores)
    }

