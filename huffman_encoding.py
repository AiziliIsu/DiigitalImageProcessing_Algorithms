import heapq
from collections import defaultdict
import numpy as np


class HuffmanNode:
    def __init__(self, value, freq):
        self.value = value
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_tree(frequency):
    """Builds a Huffman tree based on pixel frequencies."""
    heap = [HuffmanNode(value, freq) for value, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        new_node = HuffmanNode(None, node1.freq + node2.freq)
        new_node.left = node1
        new_node.right = node2
        heapq.heappush(heap, new_node)

    return heap[0]


def generate_huffman_codes(node, current_code="", codes={}):
    """Generates Huffman codes for each pixel value."""
    if node is None:
        return
    if node.value is not None:
        codes[node.value] = current_code
    generate_huffman_codes(node.left, current_code + "0", codes)
    generate_huffman_codes(node.right, current_code + "1", codes)
    return codes


def huffman_encoding(image):
    """Encodes an image using Huffman Encoding."""
    flat_image = image.flatten()
    frequency = defaultdict(int)

    for pixel in flat_image:
        frequency[pixel] += 1

    huffman_tree = build_huffman_tree(frequency)
    huffman_codes = generate_huffman_codes(huffman_tree)

    encoded_image = ''.join([huffman_codes[pixel] for pixel in flat_image])
    return encoded_image, huffman_codes, huffman_tree


def huffman_decoding(encoded_image, huffman_codes, shape):
    """Decodes the Huffman encoded image."""
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded_image = []
    code = ""

    for bit in encoded_image:
        code += bit
        if code in reverse_codes:
            decoded_image.append(reverse_codes[code])
            code = ""

    return np.array(decoded_image).reshape(shape)
