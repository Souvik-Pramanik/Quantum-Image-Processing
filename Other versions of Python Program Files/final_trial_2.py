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
    
    for i, pixel in enumerate(image.flatten()):
        if pixel == 1:
            qc.x(qr[i])
    return qc

def apply_dilation_operator(qc, image_size):
    n = image_size[0] * image_size[1]
    qr = qc.qregs[0]
    
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
    image = np.zeros(image_size)
    max_count_key = max(counts, key=counts.get)  # Find the most probable outcome
    for i, bit in enumerate(max_count_key[::-1]):
        image[i // image_size[1], i % image_size[1]] = int(bit)
    return image

def morphological_dilation(image_path):
    image = load_image(image_path)
    image_size = image.shape
    
    qc = encode_image_to_quantum(image)
    
    qc = apply_dilation_operator(qc, image_size)
    
    cr = ClassicalRegister(image.size)
    qc.add_register(cr)
    qc.measure(range(image.size), range(image.size))
    
    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend)
    job = backend.run(t_qc)
    result = job.result()
    counts = result.get_counts()
    
    dilated_image = decode_quantum_to_image(counts, image_size)
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(np.zeros(image_size), cmap='gray')  # Placeholder for intermediate step
    axs[1].set_title('Intermediate Step')
    axs[2].imshow(dilated_image, cmap='gray')
    axs[2].set_title('Dilated Image')
    plt.show()

image_path = r'C:\Users\Snaptokon\OneDrive\Documents\TINT\Research\Codes\single_dot_image.png'  # Update this path to your image file
morphological_dilation(image_path)
