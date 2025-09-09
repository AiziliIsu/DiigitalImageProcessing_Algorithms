import cv2
from utils import apply_padding

# Blurring (without padding options)
def apply_blur(image, kernel_size):
    """Applies Gaussian blur to an image."""
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image