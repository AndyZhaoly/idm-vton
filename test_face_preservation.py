"""
Test script for face preservation module
"""

import sys
sys.path.insert(0, 'gradio_demo')
sys.path.insert(0, '.')

import torch
from PIL import Image
import numpy as np
from preprocess.humanparsing.run_parsing import Parsing
from face_preservation import (
    FacePreservation, 
    visualize_mask, 
    get_head_mask, 
    paste_head_onto_generated
)

def test_face_preservation():
    print("Testing Face Preservation Module...")
    print("-" * 50)
    
    # Set device (IDM-VTON uses cuda:2)
    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize parsing model
    print("\n1. Initializing parsing model...")
    gpu_id = 2 if torch.cuda.is_available() else -1
    parsing_model = Parsing(gpu_id)
    print("   Parsing model loaded!")
    
    # Initialize face preservation
    print("\n2. Initializing face preservation...")
    face_preservation = FacePreservation(
        parsing_model=parsing_model,
        include_neck=True,
        dilate_kernel_size=5,
        feather_amount=10
    )
    print("   Face preservation initialized!")
    
    # Load test image
    print("\n3. Loading test image...")
    test_img_path = 'gradio_demo/example/human/00034_00.jpg'
    test_img = Image.open(test_img_path).convert('RGB')
    print(f"   Image size: {test_img.size}")
    
    # Get head mask
    print("\n4. Generating head mask...")
    head_mask = face_preservation.get_mask(test_img)
    print(f"   Mask shape: {head_mask.shape}")
    print(f"   Mask unique values: {np.unique(head_mask)}")
    
    # Visualize mask
    print("\n5. Creating mask visualization...")
    mask_viz = visualize_mask(test_img, head_mask, alpha=0.5, color=(0, 255, 0))
    
    # Save outputs
    print("\n6. Saving outputs...")
    mask_viz.save('test_face_mask_viz.png')
    print("   Saved: test_face_mask_viz.png")
    
    # Test paste function (simulate generated image with some changes)
    print("\n7. Testing face paste (simulating)...")
    # Create a "generated" version with slight changes
    generated = test_img.copy()
    # Simulate some try-on effect by slightly modifying colors
    generated_np = np.array(generated).astype(np.float32)
    generated_np = np.clip(generated_np * 0.9, 0, 255).astype(np.uint8)  # Darken slightly
    generated = Image.fromarray(generated_np)
    
    # Paste original face onto "generated" image
    result = paste_head_onto_generated(test_img, generated, head_mask, feather_amount=10)
    result.save('test_face_paste_result.png')
    print("   Saved: test_face_paste_result.png")
    
    # Test full pipeline
    print("\n8. Testing full face preservation pipeline...")
    result_full = face_preservation(test_img, generated)
    result_full.save('test_full_pipeline.png')
    print("   Saved: test_full_pipeline.png")
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("Check the output images:")
    print("  - test_face_mask_viz.png: Green overlay showing detected head region")
    print("  - test_face_paste_result.png: Original face pasted onto generated image")
    print("  - test_full_pipeline.png: Result from full pipeline")
    print("=" * 50)

if __name__ == "__main__":
    test_face_preservation()
