import cv2
from utils import apply_padding

def apply_sharpening(image, kernel_size):
    """Applies sharpening using a Laplacian filter."""
    # Apply Laplacian to detect edges
    laplacian = cv2.Laplacian(image, ddepth=cv2.CV_64F, ksize=kernel_size)

    # Convert Laplacian result back to the same scale as the original image
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Sharpen by adding the Laplacian to the original image
    sharpened_image = cv2.addWeighted(image, 1.5, laplacian_abs, -0.5, 0)

    return sharpened_image

