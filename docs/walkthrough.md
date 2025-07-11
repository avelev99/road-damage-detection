# Road Damage Detection System Walkthrough

## SAF Module Architecture
```mermaid
graph LR
    A[Thermal Input] --> C[Spatial Attention Fusion]
    B[RGB Input] --> C
    C --> D[Feature Extraction]
    D --> E[Damage Prediction]
```

*Implementation details will be added once SAF module is implemented*

## VRAM Optimization Techniques
Techniques used in training pipeline:
- Automatic Mixed Precision (AMP)
- Gradient scaling
- Batch size optimization

## Sample Augmentation Images
![Augmentation Samples](datasets/reports/augmentation_samples/sample1.png)
![Augmentation Samples](datasets/reports/augmentation_samples/sample2.png)

*Actual images will be added once available*