import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Function to load an image from either a URL or local file path without converting to grayscale
def load_image(image_path):
    """Loads an image from either a URL or local file path without converting to grayscale."""
    # Check if the input is a local file path
    if os.path.exists(image_path):
        try:
            img = Image.open(image_path)
            return np.array(img)  # Preserve color image
        except Exception as e:
            st.error("Failed to load image from the local file path.")
            return None
    else:
        # Otherwise, assume it's a URL and try to load it from the web
        try:
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
            return np.array(img)  # Preserve color image
        except Exception as e:
            st.error("Failed to load image from the URL.")
            return None

# Function to resize and display images side by side
def display_side_by_side(*images):
    """Resizes images to have the same height, then concatenates them side by side."""
    # Find the minimum height among all images
    min_height = min(image.shape[0] for image in images)

    # Resize all images to the same height
    resized_images = [cv2.resize(image, (int(image.shape[1] * min_height / image.shape[0]), min_height)) for image in images]

    # Concatenate the resized images along the width axis
    concatenated_image = np.concatenate(resized_images, axis=1)

    return concatenated_image

# Function to plot histograms for a list of images
def plot_histograms(images, titles):
    """Plots histograms of the given images."""
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        axs[i].hist(img.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
        axs[i].set_title(title)
    st.pyplot(fig)  # Display the plot in Streamlit

# Padding utility function to apply different types of padding
def apply_padding(image, padding_type, kernel_size):
    """Applies padding to an image depending on the padding type."""
    if padding_type == "Zero Padding":
        padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_CONSTANT, value=0)
    elif padding_type == "Replication Padding":
        padded_image = cv2.copyMakeBorder(image, kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2, cv2.BORDER_REPLICATE)
    else:
        padded_image = image  # No padding
    return padded_image

# Function for applying blur (Gaussian Blur or Mean Filter)
def apply_blur(image, kernel_size, padding_type):
    """Applies Gaussian blur to an image with padding options."""
    padded_image = apply_padding(image, padding_type, kernel_size)
    blurred_image = cv2.GaussianBlur(padded_image, (kernel_size, kernel_size), 0)
    return blurred_image

# Function for sharpening using a Laplacian filter
def apply_sharpening(image, kernel_size, padding_type):
    """Applies sharpening using a Laplacian filter with padding options."""
    padded_image = apply_padding(image, padding_type, kernel_size)
    laplacian = cv2.Laplacian(padded_image, cv2.CV_64F)
    sharpened_image = cv2.convertScaleAbs(laplacian)
    return sharpened_image

# Function for edge detection (Sobel or Prewitt operators)
def apply_edge_detection(image, operator, kernel_size, padding_type):
    """Applies Sobel or Prewitt operator for edge detection."""
    padded_image = apply_padding(image, padding_type, kernel_size)
    if operator == "Sobel":
        edges = cv2.Sobel(padded_image, cv2.CV_64F, 1, 1, ksize=kernel_size)
    else:  # Prewitt operator
        kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        edges = cv2.filter2D(padded_image, -1, kernelx) + cv2.filter2D(padded_image, -1, kernely)
    return edges

# Noise Simulation (without skimage)
def simulate_noise(image, noise_type):
    """Simulates noise on an image (Gaussian, Salt and Pepper, Speckle, Poisson)."""
    if noise_type == "Gaussian":
        row, col, ch = image.shape
        mean = 0
        var = 10
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (row, col, ch)).astype(np.uint8)
        noisy_image = cv2.add(image, gaussian)
    elif noise_type == "Salt and Pepper":
        noisy_image = np.copy(image)
        salt_pepper_ratio = 0.02
        num_salt = np.ceil(salt_pepper_ratio * image.size * 0.5)
        num_pepper = np.ceil(salt_pepper_ratio * image.size * 0.5)

        # Add salt noise (white pixels)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 255

        # Add pepper noise (black pixels)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[coords[0], coords[1], :] = 0
    elif noise_type == "Speckle":
        noisy_image = image + image * np.random.randn(*image.shape)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    else:  # Poisson Noise
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy_image = np.random.poisson(image * vals) / float(vals)
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

# Noise Removal Function
def apply_noise_removal(image, noise_type, padding_type, kernel_size):
    """Applies noise removal filter depending on the noise type."""
    padded_image = apply_padding(image, padding_type, kernel_size)
    if noise_type == "Gaussian":
        return cv2.GaussianBlur(padded_image, (kernel_size, kernel_size), 0)
    elif noise_type == "Salt and Pepper":
        return cv2.medianBlur(padded_image, kernel_size)
    else:
        return cv2.blur(padded_image, (kernel_size, kernel_size))
