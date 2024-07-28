import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer

def load_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((4, 4))  # Resize image to 4x4 pixels
    image = np.array(image)
    image = (image > 128).astype(int)  # Convert to binary image
    return image

def encode_image_to_quantum(image):
    n = image.size
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    
    # Encode the binary image into the quantum circuit
    for i, pixel in enumerate(image.flatten()):
        if pixel == 1:
            qc.x(qr[i])
    return qc

def apply_dilation_operator(qc, image_size):
    qr = qc.qregs[0]
    
    # Apply a controlled operation for each pixel and its neighbors
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            idx = i * image_size[1] + j
            if i > 0:  # Top neighbor
                qc.cx(qr[idx], qr[idx - image_size[1]])
            if i < image_size[0] - 1:  # Bottom neighbor
                qc.cx(qr[idx], qr[idx + image_size[1]])
            if j > 0:  # Left neighbor
                qc.cx(qr[idx], qr[idx - 1])
            if j < image_size[1] - 1:  # Right neighbor
                qc.cx(qr[idx], qr[idx + 1])
                
    return qc

def apply_erosion_operator(qc, image_size):
    qr = qc.qregs[0]
    
    # Apply a controlled operation for each pixel and its neighbors
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            idx = i * image_size[1] + j
            if i > 0:  # Top neighbor
                qc.ccx(qr[idx - image_size[1]], qr[idx], qr[idx])
            if i < image_size[0] - 1:  # Bottom neighbor
                qc.ccx(qr[idx + image_size[1]], qr[idx], qr[idx])
            if j > 0:  # Left neighbor
                qc.ccx(qr[idx - 1], qr[idx], qr[idx])
            if j < image_size[1] - 1:  # Right neighbor
                qc.ccx(qr[idx + 1], qr[idx], qr[idx])
                
    return qc

def decode_quantum_to_image(counts, image_size):
    image = np.zeros(image_size)
    max_count_key = max(counts, key=counts.get)  # Find the most probable outcome
    for i, bit in enumerate(max_count_key[::-1]):
        image[i // image_size[1], i % image_size[1]] = int(bit)
    return image

def measure_circuit(qc, size):
    cr = ClassicalRegister(size)
    qc.add_register(cr)
    qc.measure(range(size), range(size))
    return qc

def morphological_dilation(image_path):
    image = load_image(image_path)
    image_size = image.shape
    
    # Encode the image into a quantum circuit
    qc = encode_image_to_quantum(image)
    
    # Apply dilation operator
    qc = apply_dilation_operator(qc, image_size)
    
    # Measure the quantum circuit
    qc = measure_circuit(qc, image.size)
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc)
    result = job.result()
    counts = result.get_counts()
    
    # Decode the result to an image
    dilated_image = decode_quantum_to_image(counts, image_size)
    
    # Display the original and dilated images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(dilated_image, cmap='gray')
    axs[1].set_title('Dilated Image')
    plt.show()

def morphological_erosion(image_path):
    image = load_image(image_path)
    image_size = image.shape
    
    # Encode the image into a quantum circuit
    qc = encode_image_to_quantum(image)
    
    # Apply erosion operator
    qc = apply_erosion_operator(qc, image_size)
    
    # Measure the quantum circuit
    qc = measure_circuit(qc, image.size)
    
    # Execute the circuit
    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc)
    result = job.result()
    counts = result.get_counts()
    
    # Decode the result to an image
    eroded_image = decode_quantum_to_image(counts, image_size)
    
    # Display the original and eroded images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(eroded_image, cmap='gray')
    axs[1].set_title('Eroded Image')
    plt.show()

def morphological_edge_detection(image_path):
    image = load_image(image_path)
    image_size = image.shape
    
    # Encode the image into a quantum circuit
    qc = encode_image_to_quantum(image)
    
    # Apply dilation operator
    dilated_qc = apply_dilation_operator(qc.copy(), image_size)
    
    # Apply erosion operator
    eroded_qc = apply_erosion_operator(qc.copy(), image_size)
    
    # Measure the quantum circuits
    dilated_qc = measure_circuit(dilated_qc, image.size)
    eroded_qc = measure_circuit(eroded_qc, image.size)
    
    # Execute the circuits
    backend = Aer.get_backend('qasm_simulator')
    dilated_t_qc = transpile(dilated_qc, backend)
    dilated_job = backend.run(dilated_t_qc)
    dilated_result = dilated_job.result()
    dilated_counts = dilated_result.get_counts()
    
    eroded_t_qc = transpile(eroded_qc, backend)
    eroded_job = backend.run(eroded_t_qc)
    eroded_result = eroded_job.result()
    eroded_counts = eroded_result.get_counts()
    
    # Decode the results to images
    dilated_image = decode_quantum_to_image(dilated_counts, image_size)
    eroded_image = decode_quantum_to_image(eroded_counts, image_size)
    
    # Compute edge image (dilated - eroded)
    edge_image = dilated_image - eroded_image
    edge_image = np.clip(edge_image, 0, 1)
    
    # Display the original and edge-detected images
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(edge_image, cmap='gray')
    axs[1].set_title('Edge Detected Image')
    plt.show()

# Example usage
image_path = r'C:\Users\Snaptokon\OneDrive\Documents\TINT\Research\Codes\single_dot_image.png'  # Update this path to your image file
morphological_dilation(image_path)
morphological_erosion(image_path)
morphological_edge_detection(image_path)
