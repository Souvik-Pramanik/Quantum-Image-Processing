from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, Aer, transpile, assemble, execute
import numpy as np
import matplotlib.pyplot as plt

def load_image(image_path, size=(4, 4)):
    from PIL import Image
    image = Image.open(image_path).convert('L')
    image = image.resize(size)
    image = np.asarray(image)
    image = (image > 128).astype(int)  # Convert to binary image
    return image

def encode_image_to_quantum(image):
    num_qubits = image.size
    qc = QuantumCircuit(num_qubits)
    
    for idx, pixel in enumerate(image.flatten()):
        if pixel == 1:
            qc.x(idx)
    return qc

def apply_dilation_operator(qc, image_size):
    num_qubits = image_size[0] * image_size[1]
    
    for i in range(image_size[0]):
        for j in range(image_size[1]):
            idx = i * image_size[1] + j
            neighbors = []
            if i > 0:  # up
                neighbors.append(idx - image_size[1])
            if i < image_size[0] - 1:  # down
                neighbors.append(idx + image_size[1])
            if j > 0:  # left
                neighbors.append(idx - 1)
            if j < image_size[1] - 1:  # right
                neighbors.append(idx + 1)
            
            for n in neighbors:
                qc.cx(idx, n)
    return qc

def decode_quantum_to_image(counts, image_size):
    max_state = max(counts, key=counts.get)
    binary_string = format(int(max_state, 16), f'0{image_size[0] * image_size[1]}b')
    image = np.array(list(map(int, binary_string))).reshape(image_size)
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
    qobj = assemble(t_qc)
    result = execute(qc, backend).result()
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
morphological_dilation('small_binary_image.png')