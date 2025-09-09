import cv2

import cv2
import numpy as np


def global_histogram_equalization(input_image):
    """Applies global histogram equalization to grayscale or color images."""
    # If the image is grayscale, apply equalizeHist directly
    if len(input_image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(input_image)

    # If the image is color (3 channels), apply histogram equalization to each channel
    elif len(input_image.shape) == 3:  # Color image (assumed to be BGR format)
        channels = cv2.split(input_image)  # Split into B, G, R channels
        eq_channels = [cv2.equalizeHist(channel) for channel in channels]  # Equalize each channel
        return cv2.merge(eq_channels)  # Merge the equalized channels back together

    return input_image  # If image doesn't fit the expected format, return unchanged


def adaptive_histogram_equalization(input_image, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(input_image)
