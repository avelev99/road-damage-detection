import os
import cv2
import numpy as np
from model import RoadDamageModel  # Assuming model.py exists
from utils import visualize_predictions  # Assuming utils.py exists

# Configuration
TEST_IMAGES_DIR = 'datasets/raw/dataset/test/images'
OUTPUT_DIR = 'assets/demos'
IMAGE_PATHS = [
    'i148.png',
    'i1040.jpg',
    'img1.jpg'  # Example from datasets/processed/val/images
]

def create_demo_gifs():
    """Generate animated GIFs showing prediction evolution"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Validate model checkpoint
    checkpoint = 'model.ckpt'
    if not os.path.exists(checkpoint):
        print(f"Error: Model checkpoint not found at {checkpoint}")
        return
    
    model = RoadDamageModel.load_from_checkpoint(checkpoint)
    
    for img_name in IMAGE_PATHS:
        img_path = os.path.join(TEST_IMAGES_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        # Load and process image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get model prediction (assuming model supports intermediate outputs)
        prediction = model.predict(image)
        
        # Generate visualization frames with smooth transitions
        frames = []
        
        # Start with original image
        frames.append(visualize_predictions(image, None))
        
        # Add intermediate prediction states (simulated if not available)
        for alpha in np.linspace(0, 1, 10):
            # Blend between original and prediction
            blended_pred = None if alpha == 0 else prediction * alpha
            blended_viz = visualize_predictions(image, blended_pred)
            frames.append(blended_viz)
        
        # Add final prediction
        frames.append(visualize_predictions(image, prediction))
        
        # Create animated GIF with smooth looping
        gif_path = os.path.join(OUTPUT_DIR, f'demo_{os.path.splitext(img_name)[0]}.gif')
        create_gif(frames, gif_path, duration=100, loop=0)
        print(f"Created animated demo: {gif_path}")

def create_gif(images, output_path, duration=100, loop=0):
    """Create optimized GIF with looping support
    
    Args:
        images: List of numpy array images
        output_path: Output file path
        duration: Frame duration in milliseconds
        loop: Number of loops (0 = infinite)
    """
    import imageio
    with imageio.get_writer(
        output_path,
        mode='I',
        duration=duration/1000,  # Convert to seconds
        loop=loop,  # Infinite looping
        subrectangles=True  # Optimize frame differences
    ) as writer:
        for img in images:
            # Convert to uint8 if needed
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            writer.append_data(img)

if __name__ == '__main__':
    create_demo_gifs()