import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer

def load_image(image_path):
    """
    Load a binary image from the given file path and convert it to a numpy array.
    """
    from PIL import Image
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = image.resize((4, 4))  # Resize image to 4x4 pixels
    image = np.array(image)
    image = (image > 128).astype(int)  # Convert to binary image
    return image

def encode_image_to_quantum(image):
    """
    Encode a binary image into a quantum circuit.
    """
    n = image.size
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    
    # Encode the binary image into the quantum circuit
    for i, pixel in enumerate(image.flatten()):
        if pixel == 1:
            qc.x(qr[i])
    return qc

def apply_dilation_operator(qc, image_size):
    """
    Apply a simple dilation operator on the quantum circuit.
    """
    n = image_size[0] * image_size[1]
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

def decode_quantum_to_image(counts, image_size):
    """
    Decode the quantum measurement results to an image.
    """
    image = np.zeros(image_size)
    max_count_key = max(counts, key=counts.get)  # Find the most probable outcome
    for i, bit in enumerate(max_count_key[::-1]):
        image[i // image_size[1], i % image_size[1]] = int(bit)
    return image

def morphological_dilation(image_path):
    image = load_image(image_path)
    image_size = image.shape
    
    # Encode the image into a quantum circuit
    qc = encode_image_to_quantum(image)
    
    # Apply dilation operator
    qc = apply_dilation_operator(qc, image_size)
    
    # Measure the quantum circuit
    cr = ClassicalRegister(image.size)
    qc.add_register(cr)
    qc.measure(range(image.size), range(image.size))
    
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

# Example usage
image_path = r'C:/Users/Snaptokon/OneDrive/Documents/TINT/Research/Codes/single_dot_image.png'  # Update this path to your image file
morphological_dilation(image_path)
