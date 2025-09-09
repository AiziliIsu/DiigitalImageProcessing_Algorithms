import numpy as np


def run_length_encoding(image):
    """Performs Run Length Encoding (RLE) on the image."""
    flat_image = image.flatten()
    encoded = []
    prev_pixel = flat_image[0]
    count = 1

    for pixel in flat_image[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoded.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1

    encoded.append((prev_pixel, count))  # Add the last set
    return encoded


def rle_decode(encoded, shape):
    """Decodes RLE encoded image back to original."""
    decoded = []
    for pixel, count in encoded:
        decoded.extend([pixel] * count)
    return np.array(decoded).reshape(shape)
