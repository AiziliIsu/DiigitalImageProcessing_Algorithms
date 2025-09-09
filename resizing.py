# resizing.py
import cv2
import numpy as np

def load_image(image_path):
    """Loads an image from the provided path."""
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV default) to RGB
    return None

def resize_image(image, width, height, interpolation="bilinear"):
    """Resize the image using bilinear or nearest neighbor interpolation."""
    if interpolation == "bilinear":
        interpolation_method = cv2.INTER_LINEAR
    elif interpolation == "nearest":
        interpolation_method = cv2.INTER_NEAREST
    else:
        raise ValueError("Unsupported interpolation method. Choose 'bilinear' or 'nearest'.")

    resized_image = cv2.resize(image, (width, height), interpolation=interpolation_method)
    return resized_image
