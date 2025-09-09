import streamlit as st
import numpy as np
import cv2


# Function to create a sample matrix (for visualizing padding)
def create_sample_matrix(size):
    """Creates a simple matrix of given size."""
    return np.arange(1, size ** 2 + 1).reshape((size, size))


# Function to apply padding to a matrix
def apply_padding_to_matrix(matrix, padding_type, padding_size):
    """Applies padding to a matrix based on the padding type."""
    if padding_type == "Zero Padding":
        padded_matrix = cv2.copyMakeBorder(matrix, padding_size, padding_size, padding_size, padding_size,
                                           cv2.BORDER_CONSTANT, value=0)
    elif padding_type == "Replication Padding":
        padded_matrix = cv2.copyMakeBorder(matrix, padding_size, padding_size, padding_size, padding_size,
                                           cv2.BORDER_REPLICATE)
    else:
        padded_matrix = matrix  # No Padding
    return padded_matrix


# Function to handle the padding menu in the app
def padding_menu():
    """Displays padding options and matrix visualization."""
    st.title("Padding Options Visualization")

    # Sidebar for padding options
    padding_option = st.sidebar.selectbox(
        "Choose Padding Option:",
        ("Zero Padding", "Replication Padding", "No Padding")
    )

    # Slider for setting padding size
    padding_size = st.sidebar.slider("Padding Size", 1, 5, step=1)

    # Slider for matrix size
    matrix_size = st.sidebar.slider("Matrix Size", 3, 10, step=1)

    # Generate a sample matrix to apply padding to
    matrix = create_sample_matrix(matrix_size)

    # Display the original matrix
    st.write("Original Matrix:")
    display_matrix(matrix)

    # Apply padding to the matrix
    padded_matrix = apply_padding_to_matrix(matrix, padding_option, padding_size)

    # Display the padded matrix
    st.write(f"Padded Matrix ({padding_option}):")
    display_matrix(padded_matrix)


# Function to display a matrix without indexes
def display_matrix(matrix):
    """Displays a matrix without indexes."""
    matrix_str = '\n'.join([' '.join([f'{int(cell):2}' for cell in row]) for row in matrix])
    st.text(matrix_str)
