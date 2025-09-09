from collections import defaultdict
import numpy as np


def arithmetic_encoding(image):
    """Performs Arithmetic Encoding on the image."""
    flat_image = image.flatten()
    low, high = 0.0, 1.0

    # Calculate frequency of each pixel
    frequency = defaultdict(int)
    for pixel in flat_image:
        frequency[pixel] += 1

    total_pixels = len(flat_image)
    prob_ranges = {}
    low_limit = 0.0

    # Build probability ranges for each pixel value
    for pixel, freq in frequency.items():
        high_limit = low_limit + freq / total_pixels
        prob_ranges[pixel] = (low_limit, high_limit)
        low_limit = high_limit

    # Perform encoding
    for pixel in flat_image:
        range_low, range_high = prob_ranges[pixel]
        range_width = high - low
        high = low + range_width * range_high
        low = low + range_width * range_low

    # Return the encoded value
    return (low + high) / 2, frequency


def arithmetic_decoding(encoded_value, frequency, image_shape):
    """Decodes the Arithmetic Encoded image."""
    total_pixels = np.prod(image_shape)  # Get total number of pixels (including color channels)
    prob_ranges = {}
    low_limit = 0.0

    # Build probability ranges for each pixel value
    for pixel, freq in frequency.items():
        high_limit = low_limit + freq / total_pixels
        prob_ranges[pixel] = (low_limit, high_limit)
        low_limit = high_limit

    # Decode the image
    decoded_image = []
    value = encoded_value

    for _ in range(total_pixels):
        for pixel, (low, high) in prob_ranges.items():
            if low <= value < high:
                decoded_image.append(pixel)
                value = (value - low) / (high - low)
                break

    return np.array(decoded_image).reshape(image_shape)
