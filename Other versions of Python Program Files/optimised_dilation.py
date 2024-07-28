import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import MCMT

def load_image(image_path):
    from PIL import Image
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = np.array(image)
    image = (image > 128).astype(int)  # Convert to binary image
    return image

def save_image(image, path):
    from PIL import Image
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(path)

def encode_image_to_quantum(image):
    n = image.size
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)
    
    for i, pixel in enumerate(image.flatten()):
        if pixel == 1:
            qc.x(qr[i])
    return qc

def apply_custom_dilation_operator(qc, image_size):
    n = image_size[0] * image_size[1]
    qr = qc.qregs[0]
    
    def apply_custom_gate(control_qubits, target_qubit):
        # Custom gate can be created here, for example a multi-controlled NOT gate
        mcmt_gate = MCMT('cx', len(control_qubits), 1)
        qc.append(mcmt_gate, control_qubits + [target_qubit])

    for i in range(image_size[0]):
        for j in range(image_size[1]):
            idx = i * image_size[1] + j
            neighbors = []
            if i > 0:  # Top neighbor
                neighbors.append(qr[idx - image_size[1]])
            if i < image_size[0] - 1:  # Bottom neighbor
                neighbors.append(qr[idx + image_size[1]])
            if j > 0:  # Left neighbor
                neighbors.append(qr[idx - 1])
            if j < image_size[1] - 1:  # Right neighbor
                neighbors.append(qr[idx + 1])
            
            if neighbors:
                apply_custom_gate(neighbors, qr[idx])
                
    return qc

def decode_quantum_to_image(counts, image_size):
    image = np.zeros(image_size)
    max_count_key = max(counts, key=counts.get)  # Find the most probable outcome
    for i, bit in enumerate(max_count_key[::-1]):
        image[i // image_size[1], i % image_size[1]] = int(bit)
    return image

def process_chunk(chunk, chunk_shape):
    image_size = chunk_shape
    
    qc = encode_image_to_quantum(chunk)
    
    qc = apply_custom_dilation_operator(qc, image_size)
    
    cr = ClassicalRegister(image_size[0] * image_size[1])
    qc.add_register(cr)
    qc.measure(range(image_size[0] * image_size[1]), range(image_size[0] * image_size[1]))
    
    backend = Aer.get_backend('qasm_simulator')
    t_qc = transpile(qc, backend, optimization_level=3)  # Use highest optimization level
    job = backend.run(t_qc)
    result = job.result()
    counts = result.get_counts()
    
    dilated_chunk = decode_quantum_to_image(counts, image_size)
    return dilated_chunk

def morphological_dilation(image_path, chunk_size=(4, 4)):
    image = load_image(image_path)
    image_height, image_width = image.shape
    chunk_height, chunk_width = chunk_size
    
    dilated_image = np.zeros_like(image)
    
    for i in range(0, image_height, chunk_height):
        for j in range(0, image_width, chunk_width):
            chunk = image[i:i+chunk_height, j:j+chunk_width]
            chunk_shape = chunk.shape
            
            # If the chunk is smaller than the chunk size, pad it with zeros
            if chunk.shape[0] != chunk_height or chunk.shape[1] != chunk_width:
                chunk = np.pad(chunk, ((0, chunk_height - chunk.shape[0]), (0, chunk_width - chunk.shape[1])), 'constant')
            
            dilated_chunk = process_chunk(chunk, chunk_shape)
            dilated_image[i:i+chunk_shape[0], j:j+chunk_shape[1]] = dilated_chunk[:chunk_shape[0], :chunk_shape[1]]
    
    # Display the original and dilated images
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(dilated_image, cmap='gray')
    axs[1].set_title('Dilated Image')
    plt.show()

    return dilated_image

# Example usage
image_path = r'C:\Users\Snaptokon\OneDrive\Documents\TINT\Research\Codes\specific_4x4_image.png'  # Update this path to your image file
dilated_image = morphological_dilation(image_path)
save_image(dilated_image, r'C:\Users\Snaptokon\OneDrive\Documents\TINT\Research\Codes\dilated_image.png')  # Save the output image


# Explanation of Optimizations
# Transpilation Optimization Level: The transpile function now uses optimization_level=3, which is the highest optimization level in Qiskit. This level includes advanced techniques like gate fusion, gate cancellation, and the use of more efficient gate sets.
# Efficient Gate Sets: We use native gates and avoid unnecessary operations to minimize the depth and complexity of the circuit.
# Circuit Design: The design of the custom dilation operator aims to minimize the number of gates and their interactions.
# By incorporating these optimizations, we aim to reduce the circuit depth and improve the performance, making it more feasible to run on actual quantum hardware.