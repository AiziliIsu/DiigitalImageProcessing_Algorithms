
import cv2
import numpy as np
from skimage.util import random_noise
from utils import apply_padding

# Noise Simulation
def simulate_noise(image, noise_type):
    if noise_type == "Gaussian":
        noisy_image = random_noise(image, mode='gaussian')
    elif noise_type == "Salt and Pepper":
        noisy_image = random_noise(image, mode='s&p')
    elif noise_type == "Speckle":
        noisy_image = random_noise(image, mode='speckle')
    else:  # Poisson Noise
        noisy_image = random_noise(image, mode='poisson')
    noisy_image = np.array(255 * noisy_image, dtype='uint8')
    return noisy_image

# Noise Removal Function
def apply_noise_removal(image, noise_type, kernel_size):
    """Applies noise removal filter depending on the noise type."""

    # Apply the appropriate noise removal filter based on the noise type
    if noise_type == "Gaussian":
        # Gaussian blur requires a sigmaX value, which can be based on the kernel size or explicitly specified
        sigmaX = 0  # Automatically calculated from the kernel size
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX)

    elif noise_type == "Salt and Pepper":
        # Median blur does not require sigmaX, just the kernel size
        return cv2.medianBlur(image, kernel_size)

    else:
        # Mean/average filter as a fallback for other noise types
        return cv2.blur(image, (kernel_size, kernel_size))

