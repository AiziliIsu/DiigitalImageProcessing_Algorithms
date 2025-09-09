import streamlit as st
import cv2
from exact_histogram_matching import exact_histogram_matching
from histogram_equalization import global_histogram_equalization
from blurring import apply_blur
from sharpening import apply_sharpening
from noise import simulate_noise, apply_noise_removal
from utils import load_image, display_side_by_side, plot_histograms, apply_padding
from padding import padding_menu
from run_length_encoding import run_length_encoding, rle_decode
from huffman_encoding import huffman_encoding, huffman_decoding
from arithmetic_encoding import arithmetic_encoding, arithmetic_decoding
from collections import defaultdict
from arithmetic_encoding import arithmetic_encoding, arithmetic_decoding

import numpy as np

# app.py
import streamlit as st
from resizing import load_image, resize_image  # Import the resizing functions

# Sidebar menu for switching between algorithms
menu_option = st.sidebar.selectbox(
    "Choose an option:",
    ("Histogram Equalization", "Exact Histogram Matching", "Blurring", "Sharpening",
     "Noise Simulation", "Noise Removal", "Padding Options", "Run Length Encoding (RLE)",
     "Huffman Encoding", "Arithmetic Encoding", "Log Transformation", "Power Law Transformation",
     "Contrast Stretching", "Image Acquisition", "Image Sampling and Quantization", "Image Scaling", "Image Resizing")
)

# Image Resizing Option (Separate from other menu items)
if menu_option == "Image Resizing":
    st.title("Image Resizing: Bilinear or Nearest Neighbor")

    # Input fields for image URL or path, width, height, and interpolation method
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Resizing")
    resize_width = st.number_input("Resize Width", min_value=1, step=1, value=100)
    resize_height = st.number_input("Resize Height", min_value=1, step=1, value=100)
    interpolation_option = st.selectbox("Choose Interpolation Method", ["Bilinear", "Nearest Neighbor"])

    if img_url_or_path:
        input_image = load_image(img_url_or_path)

        if input_image is not None:
            st.image(input_image, caption="Original Image", use_column_width=True)

            # Map user choice to interpolation method
            interpolation = "bilinear" if interpolation_option == "Bilinear" else "nearest"

            # Resize the image
            resized_image = resize_image(input_image, resize_width, resize_height, interpolation)

            # Display resized image
            st.image(resized_image, caption=f"Resized Image ({interpolation_option})", use_column_width=True)

            # Display original and resized size comparison
            original_size = input_image.shape[:2]  # Height, Width
            resized_size = resized_image.shape[:2]
            st.write(f"**Original Size:** {original_size[1]}x{original_size[0]}")
            st.write(f"**Resized Size:** {resized_size[1]}x{resized_size[0]}")

# For Histogram Equalization (Already Implemented)
if menu_option == "Histogram Equalization":
    st.title("Histogram Equalization")
    img_url_or_path = st.text_input("Enter the Input Image URL or Local Path for Equalization")
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            equalized_image = global_histogram_equalization(input_image)
            st.image(display_side_by_side(input_image, equalized_image), caption="Input and Equalized Image", use_column_width=True)
            st.write("Histograms of Input and Equalized Image:")
            plot_histograms([input_image, equalized_image], ["Input", "Equalized"])

# For Exact Histogram Matching (Already Implemented)
elif menu_option == "Exact Histogram Matching":
    st.title("Exact Histogram Matching")
    input_img_url_or_path = st.text_input("Enter the Input Image URL or Local Path for Matching")
    ref_img_url_or_path = st.text_input("Enter the Reference Image URL or Local Path for Matching")
    if input_img_url_or_path and ref_img_url_or_path:
        input_image = load_image(input_img_url_or_path)
        reference_image = load_image(ref_img_url_or_path)
        if input_image is not None and reference_image is not None:
            matched_image, _ = exact_histogram_matching(input_image, reference_image)
            st.image(display_side_by_side(input_image, reference_image, matched_image), caption="Input, Reference, and Matched Image", use_column_width=True)
            st.write("Histograms of Input, Reference, and Matched Image:")
            plot_histograms([input_image, reference_image, matched_image], ["Input", "Reference", "Matched"])

# For Blurring
elif menu_option == "Blurring":
    st.title("Blurring")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Blurring")
    kernel_size = st.slider("Kernel Size", 3, 15, step=2)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            blurred_image = apply_blur(input_image, kernel_size)
            st.image(display_side_by_side(input_image, blurred_image), caption="Input and Blurred Image", use_column_width=True)

# For Sharpening
elif menu_option == "Sharpening":
    st.title("Sharpening")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Sharpening")
    kernel_size = st.slider("Kernel Size", 3, 15, step=2)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            sharpened_image = apply_sharpening(input_image, kernel_size)
            st.image(display_side_by_side(input_image, sharpened_image), caption="Input and Sharpened Image", use_column_width=True)




# For Noise Simulation
elif menu_option == "Noise Simulation":
    st.title("Noise Simulation")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Noise Simulation")
    noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt and Pepper", "Speckle", "Poisson"])
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            noisy_image = simulate_noise(input_image, noise_type)
            st.image(display_side_by_side(input_image, noisy_image), caption="Input and Noisy Image", use_column_width=True)

# For Noise Removal
elif menu_option == "Noise Removal":
    st.title("Noise Removal")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Noise Removal")
    noise_type = st.selectbox("Noise Type", ["Gaussian", "Salt and Pepper"])
    kernel_size = st.slider("Kernel Size", 3, 15, step=2)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            denoised_image = apply_noise_removal(input_image, noise_type, kernel_size)
            st.image(display_side_by_side(input_image, denoised_image), caption="Input and Denoised Image", use_column_width=True)


# For Padding Options
elif menu_option == "Padding Options":
    padding_menu()



# For Run Length Encoding (RLE)
if menu_option == "Run Length Encoding (RLE)":
    st.title("Run Length Encoding (RLE)")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for RLE")
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            # Perform Run Length Encoding
            encoded_image = run_length_encoding(input_image)
            decoded_image = rle_decode(encoded_image, input_image.shape)

            # Display the original and decoded images side by side for comparison
            st.image(display_side_by_side(input_image, decoded_image), caption="Original and Decoded Image",
                     use_column_width=True)

            # Calculate compression metrics
            original_size = input_image.nbytes / 1024  # Convert from bytes to KB
            compressed_size = len(
                encoded_image) * 2 / 1024  # Each RLE tuple has two elements (value, count), converted to KB
            compression_factor = original_size / compressed_size
            compression_ratio = 1 - (1 / compression_factor)  # R = 1 - 1/C
            redundant_data_percentage = compression_ratio * 100  # Percentage of redundant data

            # Display compression metrics
            st.write(f"**Original Size:** {original_size:.2f} KB")
            st.write(f"**Compressed Size:** {compressed_size:.2f} KB")
            st.write(f"**Compression Factor (C):** {compression_factor:.2f}")
            st.write(f"**Compression Ratio (R):** {compression_ratio:.3f}")
            st.write(f"Thus, **{redundant_data_percentage:.1f}%** of the data in the original image is redundant.")



# For Huffman Encoding (Optimized)
elif menu_option == "Huffman Encoding":
    st.title("Huffman Encoding")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Huffman Encoding")
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            # Perform Huffman Encoding
            encoded_image, huffman_codes, _ = huffman_encoding(input_image)

            # Display a limited portion of the encoded image to prevent performance issues
            display_limit = 100  # Limit to the first 100 bits (or symbols)
            encoded_preview = encoded_image[:display_limit] + "..." if len(
                encoded_image) > display_limit else encoded_image

            st.write(f"Huffman Encoded Output (first {display_limit} symbols):")
            st.write(encoded_preview)

            # Decode the image back for comparison
            decoded_image = huffman_decoding(encoded_image, huffman_codes, input_image.shape)

            # Display the original and decoded images side by side
            st.image(display_side_by_side(input_image, decoded_image), caption="Original and Decoded Image",
                     use_column_width=True)

            # Calculate compression metrics
            original_size = input_image.nbytes / 1024  # Convert from bytes to KB
            compressed_size = len(encoded_image) // 8 / 1024  # Convert bits to bytes and then to KB
            compression_factor = original_size / compressed_size
            compression_ratio = 1 - (1 / compression_factor)  # R = 1 - 1/C
            redundant_data_percentage = compression_ratio * 100  # Percentage of redundant data

            # Display compression metrics
            st.write(f"**Original Size:** {original_size:.2f} KB")
            st.write(f"**Compressed Size:** {compressed_size:.2f} KB")
            st.write(f"**Compression Factor (C):** {compression_factor:.2f}")
            st.write(f"**Compression Ratio (R):** {compression_ratio:.3f}")
            st.write(f"Thus, **{redundant_data_percentage:.1f}%** of the data in the original image is redundant.")



# After decoding the image
elif menu_option == "Arithmetic Encoding":
    st.title("Arithmetic Encoding")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Arithmetic Encoding")
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            # Perform Arithmetic Encoding
            encoded_value, frequency = arithmetic_encoding(input_image)

            # Display the encoded value or compression stats without decoding
            original_size = input_image.nbytes / 1024  # Convert bytes to KB
            compressed_size = len(str(encoded_value))  # Size of the compressed encoded value
            compression_factor = original_size / compressed_size
            compression_ratio = 1 - (1 / compression_factor)

            # Display compression stats
            st.write(f"**Original Size:** {original_size:.2f} KB")
            st.write(f"**Compressed Size:** {compressed_size:.2f} KB")
            st.write(f"**Compression Factor (C):** {compression_factor:.2f}")
            st.write(f"**Compression Ratio (R):** {compression_ratio:.3f}")

import numpy as np

def log_transformation(image, constant=1):
    """Apply log transformation to an image."""
    image = np.array(image, dtype=float)
    c = constant  # Scaling factor
    log_image = c * np.log(1 + image)  # Apply log transformation
    log_image = np.uint8(255 * (log_image / np.max(log_image)))  # Normalize to 255
    return log_image

# In app.py for Log Transformation
if menu_option == "Log Transformation":
    st.title("Log Transformation")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Log Transformation")
    constant = st.slider("Log Scaling Factor", 1.0, 5.0, step=0.1)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            log_image = log_transformation(input_image, constant)
            st.image(display_side_by_side(input_image, log_image), caption="Original and Log Transformed Image", use_column_width=True)

def power_law_transformation(image, gamma=1.0):
    """Apply power law transformation (gamma correction) to an image."""
    image = np.array(image, dtype=float)
    gamma_corrected = np.power(image / 255.0, gamma)  # Gamma correction
    gamma_corrected = np.uint8(255 * gamma_corrected)  # Normalize to 255
    return gamma_corrected

# In app.py for Power Law Transformation
if menu_option == "Power Law Transformation":
    st.title("Power Law Transformation (Gamma Correction)")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Gamma Correction")
    gamma = st.slider("Gamma Value", 0.1, 3.0, step=0.1)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            gamma_image = power_law_transformation(input_image, gamma)
            st.image(display_side_by_side(input_image, gamma_image), caption=f"Original and Gamma-Corrected Image (γ={gamma})", use_column_width=True)

def contrast_stretching(image, low_in, high_in):
    """Apply contrast stretching to an image."""
    image = np.array(image, dtype=float)
    stretched_image = 255 * (image - low_in) / (high_in - low_in)  # Apply contrast stretching
    stretched_image = np.clip(stretched_image, 0, 255)  # Clip values to valid range
    return np.uint8(stretched_image)

# In app.py for Contrast Stretching
if menu_option == "Contrast Stretching":
    st.title("Contrast Stretching")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Contrast Stretching")
    low_in = st.slider("Lower Intensity Bound", 0, 255, step=1)
    high_in = st.slider("Upper Intensity Bound", 0, 255, step=1, value=255)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            contrast_image = contrast_stretching(input_image, low_in, high_in)
            st.image(display_side_by_side(input_image, contrast_image), caption="Original and Contrast-Stretched Image", use_column_width=True)


def image_sampling(image, sampling_factor):
    """Reduce the spatial resolution of the image by downsampling."""
    return image[::sampling_factor, ::sampling_factor]

def image_quantization(image, levels):
    """Reduce the number of intensity levels."""
    image = np.array(image, dtype=float)
    quantized_image = np.floor(image / (256 / levels)) * (256 / levels)
    return np.uint8(quantized_image)

# In app.py for Sampling and Quantization
if menu_option == "Image Sampling and Quantization":
    st.title("Image Sampling and Quantization")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Sampling and Quantization")
    sampling_factor = st.slider("Sampling Factor", 1, 10, step=1)
    quantization_levels = st.slider("Number of Quantization Levels", 2, 256, step=1)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            sampled_image = image_sampling(input_image, sampling_factor)
            quantized_image = image_quantization(sampled_image, quantization_levels)
            st.image(display_side_by_side(sampled_image, quantized_image), caption="Sampled and Quantized Image", use_column_width=True)

def image_sampling(image, sampling_factor):
    """Reduce the spatial resolution of the image by downsampling."""
    return image[::sampling_factor, ::sampling_factor]

def image_quantization(image, levels):
    """Reduce the number of intensity levels."""
    image = np.array(image, dtype=float)
    quantized_image = np.floor(image / (256 / levels)) * (256 / levels)
    return np.uint8(quantized_image)

# In app.py for Sampling and Quantization
if menu_option == "Image Sampling and Quantization":
    st.title("Image Sampling and Quantization")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Sampling and Quantization")
    sampling_factor = st.slider("Sampling Factor", 1, 10, step=1)
    quantization_levels = st.slider("Number of Quantization Levels", 2, 256, step=1)
    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            sampled_image = image_sampling(input_image, sampling_factor)
            quantized_image = image_quantization(sampled_image, quantization_levels)
            st.image(display_side_by_side(sampled_image, quantized_image), caption="Sampled and Quantized Image", use_column_width=True)


def image_scaling(image, scale_factor):
    """Scale the image by the given scaling factor."""
    image_copy = np.copy(image)  # Create a copy of the original image
    height, width = image_copy.shape[:2]
    new_size = (int(width * scale_factor), int(height * scale_factor))
    scaled_image = cv2.resize(image_copy, new_size, interpolation=cv2.INTER_LINEAR)
    return scaled_image


# In app.py for Image Scaling
if menu_option == "Image Scaling":
    st.title("Image Scaling")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path for Scaling")
    scale_factor = st.slider("Scaling Factor", 0.1, 3.0, step=0.1)

    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            scaled_image = image_scaling(input_image, scale_factor)  # Apply scaling to a copy of the image
            st.image(display_side_by_side(input_image, scaled_image), caption="Original and Scaled Image",
                     use_column_width=True)

import os
import requests
from PIL import Image
import numpy as np
import streamlit as st
from io import BytesIO

def load_image(image_path):
    """Loads an image from either a URL or local file path."""
    if os.path.exists(image_path):
        # Load from local path
        try:
            img = Image.open(image_path)
            return np.array(img)
        except Exception as e:
            st.error(f"Failed to load image from the local file path. Error: {e}")
            return None
    else:
        # Load from URL
        try:
            response = requests.get(image_path)
            img = Image.open(BytesIO(response.content))
            return np.array(img)
        except Exception as e:
            st.error(f"Failed to load image from the URL. Error: {e}")
            return None


# In the main menu logic of app.py
if menu_option == "Image Acquisition":
    st.title("Image Acquisition")
    img_url_or_path = st.text_input("Enter the Image URL or Local Path")

    if img_url_or_path:
        input_image = load_image(img_url_or_path)
        if input_image is not None:
            # Display the loaded image
            st.image(input_image, caption="Acquired Image", use_column_width=True)



