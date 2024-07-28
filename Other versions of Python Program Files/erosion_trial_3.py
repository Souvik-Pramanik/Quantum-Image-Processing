# Import necessary libraries
import numpy as np
import cv2
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import matplotlib.pyplot as plt
import mplcursors

# Function to perform classical morphological erosion
def classical_erosion(image, kernel):
    return cv2.erode(image, kernel, iterations=1)

# Function to create a quantum circuit for a hypothetical quantum erosion
def quantum_erosion_circuit(image, kernel):
    # Placeholder: Create a simple quantum circuit
    # This is a simplified example, not a real erosion operation
    n_qubits = int(np.ceil(np.log2(image.size)))
    qc = QuantumCircuit(n_qubits)

    # Example operation (not actual erosion)
    qc.h(range(n_qubits))
    qc.measure_all()

    return qc

# Function to execute the quantum circuit using transpile and assemble
def run_quantum_erosion(image, kernel):
    qc = quantum_erosion_circuit(image, kernel)
    backend = Aer.get_backend('qasm_simulator')
    
    # Transpile the quantum circuit for the backend
    tqc = transpile(qc, backend)
    
    # Assemble the transpiled circuit into a Qobj
    qobj = assemble(tqc)
    
    # Run the assembled job on the backend
    result = backend.run(qobj).result()
    counts = result.get_counts()
    
    # Placeholder: Process result (not a real erosion operation)
    return image  # Return the original image as a placeholder

# Function to convert a real image to a 4x4 binary image
def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to 4x4
    resized_image = cv2.resize(image, (4, 4), interpolation=cv2.INTER_AREA)
    # Convert the image to binary (0 or 255)
    _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

# Function to display images side by side with outlines and coordinates
def display_images_with_borders(original, processed, titles):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    im0 = axes[0].imshow(original, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].set_xticks(np.arange(-0.5, original.shape[1], 1))
    axes[0].set_yticks(np.arange(-0.5, original.shape[0], 1))
    axes[0].set_xticklabels(np.arange(0, original.shape[1] + 1, 1))
    axes[0].set_yticklabels(np.arange(0, original.shape[0] + 1, 1))
    axes[0].grid(color='white', linestyle='-', linewidth=1)
    
    # Display processed image
    im1 = axes[1].imshow(processed, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].set_xticks(np.arange(-0.5, processed.shape[1], 1))
    axes[1].set_yticks(np.arange(-0.5, processed.shape[0], 1))
    axes[1].set_xticklabels(np.arange(0, processed.shape[1] + 1, 1))
    axes[1].set_yticklabels(np.arange(0, processed.shape[0] + 1, 1))
    axes[1].grid(color='white', linestyle='-', linewidth=1)
    
    plt.tight_layout()

    # Add cursor to show coordinates
    cursor0 = mplcursors.cursor(im0, hover=True)
    cursor1 = mplcursors.cursor(im1, hover=True)
    
    cursor0.connect("add", lambda sel: sel.annotation.set_text(
        f"x={int(sel.target[0])}, y={int(sel.target[1])}"))
    cursor1.connect("add", lambda sel: sel.annotation.set_text(
        f"x={int(sel.target[0])}, y={int(sel.target[1])}"))

    plt.show()

# Example usage with a real image
image_path = r'C:\Users\Snaptokon\OneDrive\Documents\TINT\Research\Codes\random_4x4_image.png'  # Replace with the path to your image
binary_image = preprocess_image(image_path)
kernel = np.ones((2, 2), np.uint8)

# Perform classical erosion
eroded_image_classical = classical_erosion(binary_image, kernel)

# Display the original binary image and the classical erosion result side by side with borders
display_images_with_borders(binary_image, eroded_image_classical, ["Original Binary Image", "Classical Erosion Result"])

# Perform quantum erosion (placeholder)
eroded_image_quantum = run_quantum_erosion(binary_image, kernel)

# Display the original binary image and the quantum erosion placeholder result side by side with borders
display_images_with_borders(binary_image, eroded_image_quantum, ["Original Binary Image", "Quantum Erosion Placeholder Result"])
