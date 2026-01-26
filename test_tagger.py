import sys
import os
import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Mock folder_paths
sys.modules["folder_paths"] = MagicMock()
import folder_paths

# Setup local models directory
TEST_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_models"))
if not os.path.exists(TEST_MODELS_DIR):
    os.makedirs(TEST_MODELS_DIR)
folder_paths.get_folder_paths.return_value = [TEST_MODELS_DIR]

import nodes
from nodes import TRTagger

def test_tagger():
    print("Initializing TRTagger...")
    tagger = TRTagger()
    
    # Load sample image
    image_path = "sample.jpg"
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    print(f"Loading {image_path}...")
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).unsqueeze(0) 

    # Parameters
    model_name = "wd-eva02-large-tagger-v3" 
    threshold = 0.35
    character_threshold = 0.85
    use_fp16 = True

    print(f"Starting tagging with model: {model_name}")
    
    try:
        tags_tuple = tagger.tag(
            image=img_tensor,
            model_name=model_name,
            threshold=threshold,
            character_threshold=character_threshold,
            use_fp16=use_fp16
        )
        
        tags = tags_tuple[0]
        print("\n--- Result Tags ---")
        print(tags)
        print("-------------------\n")
        
    except Exception as e:
        print(f"An error occurred during tagging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tagger()
