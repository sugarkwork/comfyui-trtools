import sys
import os
import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from PIL import Image

import logging

# Configure logging to see errors
logging.basicConfig(level=logging.DEBUG)

# Mock folder_paths before importing nodes
sys.modules["folder_paths"] = MagicMock()
import folder_paths

# Setup a local models directory for testing
TEST_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_models"))
if not os.path.exists(TEST_MODELS_DIR):
    os.makedirs(TEST_MODELS_DIR)

# Mock get_folder_paths to return our test directory
folder_paths.get_folder_paths.return_value = [TEST_MODELS_DIR]

# Now import the node
import nodes
from nodes import TRTUpscaler

def test_upscaler():
    print("Initializing TRTUpscaler...")
    upscaler = TRTUpscaler()

    # Load sample image
    image_path = "sample.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert("RGB")
    
    # ComfyUI passes images as float32 tensors [B, H, W, C] normalized to 0-1
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0) # Add batch dimension
    
    # Parameters
    model_name = "4x_foolhardy_Remacri" # This should trigger download -> convert
    tile_size = 512
    tile_overlap = 32
    use_fp16 = True

    print(f"Starting upscale with model: {model_name}")
    print("This may take a while if the model needs to be downloaded and converted...")
    
    try:
        result_tuple = upscaler.upscale(
            image=img_tensor,
            model_name=model_name,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            use_fp16=use_fp16
        )
        
        result_tensor = result_tuple[0]
        
        # Convert back to image and save
        output_tensor = result_tensor.squeeze(0).cpu().numpy()
        output_image = (output_tensor * 255).clip(0, 255).astype(np.uint8)
        output_pil = Image.fromarray(output_image)
        
        output_filename = "sample_upscaled.png"
        output_pil.save(output_filename)
        print(f"Success! Upscaled image saved to {output_filename}")
        print(f"Input shape: {img_tensor.shape}")
        print(f"Output shape: {result_tensor.shape}")
        
    except Exception as e:
        print(f"An error occurred during upscaling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upscaler()
