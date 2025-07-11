import cv2
import numpy as np
import torch
from torchvision import transforms

def visualize_predictions(image, model):
    """
    Process image through model and create prediction visualization
    Returns original image and prediction visualization side-by-side
    """
    # Preprocess image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    
    # Run prediction
    with torch.no_grad():
        pred = model(img_tensor)
        pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # Create color-coded prediction visualization
    color_map = {
        0: [0, 0, 0],        # background: black
        1: [255, 0, 0],      # crack: red
        2: [0, 255, 0],      # pothole: green
        3: [0, 0, 255]       # rut: blue
    }
    h, w = pred_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        color_mask[pred_mask == class_idx] = color
    
    # Convert original from BGR to RGB
    original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create side-by-side comparison
    comparison = np.hstack((original_rgb, color_mask))
    return comparison