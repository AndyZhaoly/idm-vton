"""
Face Preservation Module for IDM-VTON

This module uses the existing SCHP human parsing model to extract head regions
(face, hair, hat, neck) and paste them from the original image onto the
generated try-on image to preserve identity.
"""

import numpy as np
import cv2
from PIL import Image
import torch
from typing import Tuple, Optional


# SCHP Label mapping (from preprocess/humanparsing/utils/miou.py)
# 0: Background, 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses, 5: Upper-clothes, 
# 6: Dress, 7: Coat, 8: Socks, 9: Pants, 10: Jumpsuits, 11: Scarf, 12: Skirt, 
# 13: Face, 14: Left-arm, 15: Right-arm, 16: Left-leg, 17: Right-leg, 
# 18: Left-shoe, 19: Right-shoe

HEAD_LABELS = [1, 13]  # Hat, Face (excludes Hair)
NECK_LABELS = [11]  # Scarf could be part of neck area


def get_head_mask(parsing_result: np.ndarray, include_neck: bool = True, 
                   dilate_kernel_size: int = 5) -> np.ndarray:
    """
    Extract head mask from parsing result.
    
    Args:
        parsing_result: Parsing result array from SCHP model (H, W) with class labels
        include_neck: Whether to include neck/scarf region
        dilate_kernel_size: Kernel size for morphological dilation to slightly expand mask
        
    Returns:
        Binary mask (H, W) with 255 for head region, 0 otherwise
    """
    mask = np.zeros_like(parsing_result, dtype=np.uint8)
    
    # Add head labels
    for label in HEAD_LABELS:
        mask = np.where(parsing_result == label, 255, mask)
    
    # Add neck labels if requested
    if include_neck:
        for label in NECK_LABELS:
            mask = np.where(parsing_result == label, 255, mask)
    
    # Dilate slightly to ensure smooth boundaries
    if dilate_kernel_size > 0:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def get_head_mask_from_parsing_model(parsing_model, input_image: Image.Image,
                                      include_neck: bool = True,
                                      dilate_kernel_size: int = 5) -> np.ndarray:
    """
    Get head mask by running the parsing model on input image.
    
    Args:
        parsing_model: The SCHP parsing model instance (from preprocess.humanparsing.run_parsing)
        input_image: PIL Image of the person
        include_neck: Whether to include neck/scarf region
        dilate_kernel_size: Kernel size for dilation
        
    Returns:
        Binary mask as numpy array
    """
    # Run parsing model (expects 384x512 image based on app.py)
    input_resized = input_image.resize((384, 512))
    parsing_result, _ = parsing_model(input_resized)
    parsing_array = np.array(parsing_result)
    
    # Get head mask
    head_mask = get_head_mask(parsing_array, include_neck, dilate_kernel_size)
    
    # Resize mask back to original image size
    head_mask_resized = cv2.resize(head_mask, input_image.size, interpolation=cv2.INTER_NEAREST)
    
    return head_mask_resized


def feather_mask(mask: np.ndarray, feather_amount: int = 10) -> np.ndarray:
    """
    Apply feathering (Gaussian blur) to mask edges for smooth blending.
    
    Args:
        mask: Binary mask (H, W)
        feather_amount: Amount of feathering/blur
        
    Returns:
        Feathered mask with values in [0, 255]
    """
    if feather_amount <= 0:
        return mask
    
    # Apply Gaussian blur for feathering
    feathered = cv2.GaussianBlur(mask.astype(np.float32), 
                                  (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
    
    # Normalize to 0-255
    feathered = np.clip(feathered, 0, 255).astype(np.uint8)
    
    return feathered


def paste_head_onto_generated(
    original_img: Image.Image,
    generated_img: Image.Image,
    head_mask: np.ndarray,
    feather_amount: int = 10
) -> Image.Image:
    """
    Paste the head region from original image onto generated image.
    
    Args:
        original_img: Original person image (PIL Image)
        generated_img: Generated try-on image (PIL Image)
        head_mask: Binary mask for head region
        feather_amount: Amount of feathering for smooth blending
        
    Returns:
        Result image with preserved face (PIL Image)
    """
    # Ensure images are RGB
    original_img = original_img.convert('RGB')
    generated_img = generated_img.convert('RGB')
    
    # Ensure sizes match
    if original_img.size != generated_img.size:
        generated_img = generated_img.resize(original_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays
    original_np = np.array(original_img).astype(np.float32)
    generated_np = np.array(generated_img).astype(np.float32)
    
    # Feather the mask for smooth blending
    feathered_mask = feather_mask(head_mask, feather_amount)
    
    # Normalize mask to 0-1
    mask_normalized = feathered_mask.astype(np.float32) / 255.0
    
    # Expand mask to 3 channels
    mask_3ch = np.stack([mask_normalized] * 3, axis=-1)
    
    # Blend images: result = generated * (1 - mask) + original * mask
    result_np = generated_np * (1 - mask_3ch) + original_np * mask_3ch
    
    # Convert back to uint8
    result_np = np.clip(result_np, 0, 255).astype(np.uint8)
    
    return Image.fromarray(result_np)


def preserve_face_in_tryon(
    original_img: Image.Image,
    generated_img: Image.Image,
    parsing_model,
    include_neck: bool = True,
    dilate_kernel_size: int = 5,
    feather_amount: int = 10
) -> Tuple[Image.Image, np.ndarray]:
    """
    Main function to preserve face in try-on generation.
    
    Args:
        original_img: Original person image
        generated_img: Generated try-on image from IDM-VTON
        parsing_model: SCHP parsing model instance
        include_neck: Whether to include neck region
        dilate_kernel_size: Size for mask dilation
        feather_amount: Feathering amount for smooth blending
        
    Returns:
        Tuple of (result_image, head_mask)
    """
    # Get head mask from original image
    head_mask = get_head_mask_from_parsing_model(
        parsing_model, original_img, include_neck, dilate_kernel_size
    )
    
    # Paste head onto generated image
    result_img = paste_head_onto_generated(
        original_img, generated_img, head_mask, feather_amount
    )
    
    return result_img, head_mask


def visualize_mask(image: Image.Image, mask: np.ndarray, 
                   alpha: float = 0.5, color: Tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
    """
    Visualize mask overlay on image for debugging.
    
    Args:
        image: Original image
        mask: Binary mask
        alpha: Transparency of overlay
        color: RGB color for mask overlay
        
    Returns:
        Image with mask overlay
    """
    img_np = np.array(image.convert('RGB'))
    mask_3ch = np.stack([mask / 255.0] * 3, axis=-1)
    
    # Create colored mask
    colored_mask = np.zeros_like(img_np)
    colored_mask[:, :] = color
    
    # Blend
    overlay = img_np * (1 - mask_3ch * alpha) + colored_mask * (mask_3ch * alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return Image.fromarray(overlay)


# For backward compatibility and direct use in app.py
class FacePreservation:
    """
    Wrapper class for face preservation that can be integrated into IDM-VTON pipeline.
    """
    
    def __init__(self, parsing_model, include_neck: bool = True, 
                 dilate_kernel_size: int = 5, feather_amount: int = 10):
        """
        Initialize FacePreservation.
        
        Args:
            parsing_model: SCHP parsing model instance
            include_neck: Whether to include neck in head mask
            dilate_kernel_size: Dilation kernel size
            feather_amount: Feathering amount for blending
        """
        self.parsing_model = parsing_model
        self.include_neck = include_neck
        self.dilate_kernel_size = dilate_kernel_size
        self.feather_amount = feather_amount
    
    def __call__(self, original_img: Image.Image, generated_img: Image.Image) -> Image.Image:
        """
        Apply face preservation.
        
        Args:
            original_img: Original person image
            generated_img: Generated try-on image
            
        Returns:
            Image with preserved face
        """
        result_img, _ = preserve_face_in_tryon(
            original_img, generated_img, self.parsing_model,
            self.include_neck, self.dilate_kernel_size, self.feather_amount
        )
        return result_img
    
    def get_mask(self, image: Image.Image) -> np.ndarray:
        """Get head mask for visualization/debugging."""
        return get_head_mask_from_parsing_model(
            self.parsing_model, image, self.include_neck, self.dilate_kernel_size
        )