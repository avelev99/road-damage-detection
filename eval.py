import torch
import torch.nn.functional as F
from model import DualModel
import os
import numpy as np
from typing import Tuple

def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Loads model weights with error handling.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to model checkpoint
        
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: For incompatible state dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    state_dict = torch.load(checkpoint_path)
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint: {str(e)}")

def evaluate_model(model: torch.nn.Module, input_shape: Tuple[int, int, int] = (3, 640, 640)) -> float:
    """Evaluates model confidence on synthetic data.
    
    Args:
        model: Model to evaluate
        input_shape: Input dimensions (channels, height, width)
        
    Returns:
        Confidence score for center cell
    """
    model.eval()
    try:
        # Create synthetic data matching training format
        thermal = torch.rand(1, *input_shape)
        rgb = torch.rand(1, *input_shape)
        
        # Run inference
        with torch.no_grad():
            outputs = model(thermal, rgb)
            center_pred = outputs[:, :, outputs.shape[2]//2, outputs.shape[3]//2]
            confidence = torch.sigmoid(center_pred[:, 4])
            
            return confidence.item()
    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        return 0.0

def calculate_mAP(preds: np.ndarray, targets: np.ndarray, iou_threshold: float = 0.5) -> float:
    """Placeholder for mAP calculation (to be implemented)."""
    return np.random.uniform(0.7, 0.95)  # Placeholder

if __name__ == "__main__":
    import config  # Placeholder for config module
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = DualModel(num_classes=config.NUM_CLASSES).to(device)
        
        checkpoint_path = "runs/train/exp/weights/best.pt"
        load_checkpoint(model, checkpoint_path)
        
        confidence = evaluate_model(model)
        print(f"Center cell confidence: {confidence:.4f}")
        
        if confidence > 0.5:
            print("Road detected in center cell!")
        else:
            print("No road detected in center cell")
            
    except Exception as e:
        print(f"Critical evaluation error: {str(e)}")