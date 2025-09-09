import numpy as np


def exact_histogram_matching(input_image, reference_image, bins=256):
    input_hist, input_bins = np.histogram(input_image.flatten(), bins, [0, 256])
    ref_hist, ref_bins = np.histogram(reference_image.flatten(), bins, [0, 256])

    input_cdf = np.cumsum(input_hist)
    input_cdf_normalized = input_cdf / input_cdf[-1]  # Normalized CDF

    ref_cdf = np.cumsum(ref_hist)
    ref_cdf_normalized = ref_cdf / ref_cdf[-1]

    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        diff = np.abs(ref_cdf_normalized - input_cdf_normalized[i])
        mapping[i] = np.argmin(diff)

    matched_image = mapping[input_image]
    return matched_image, mapping
