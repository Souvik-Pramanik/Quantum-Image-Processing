import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import MCMT
from PIL import Image
import io
import ipywidgets as widgets

def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert image to grayscale
    image = np.array(image)
    image = (image > 128).astype(int)  # Convert to binary image
    return image

def save_image(image, path):
    image = Image.fromarray((image * 255).astype(np.uint8))
    image.save(path)

def morphological_dilation(image_path, chunk_size=(4, 4)):
    # Load the image
    image = load_image(image_path)
    
    # Define dilation operation
    def dilate_chunk(chunk, chunk_shape):
        # Your dilation logic here
        return chunk
    
    # Apply dilation operation to image chunks
    dilated_image = np.zeros_like(image)
    chunk_height, chunk_width = chunk_size
    for i in range(0, image.shape[0], chunk_height):
        for j in range(0, image.shape[1], chunk_width):
            chunk = image[i:i+chunk_height, j:j+chunk_width]
            chunk_shape = chunk.shape
            dilated_chunk = dilate_chunk(chunk, chunk_shape)
            dilated_image[i:i+chunk_shape[0], j:j+chunk_shape[1]] = dilated_chunk
            
    # Display the original and dilated images
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(dilated_image, cmap='gray')
    axs[1].set_title('Dilated Image')
    plt.show()

# Create file upload widget
file_upload = widgets.FileUpload(accept='.png, .jpg, .jpeg')

# Create chunk size widgets
chunk_height_slider = widgets.IntSlider(value=4, min=1, max=32, step=1, description='Chunk Height:')
chunk_width_slider = widgets.IntSlider(value=4, min=1, max=32, step=1, description='Chunk Width:')
chunk_size_box = widgets.HBox([chunk_height_slider, chunk_width_slider])

# Create button to trigger dilation
dilate_button = widgets.Button(description='Dilate Image')

def on_dilate_button_clicked(b):
    if file_upload.value:
        file_content = file_upload.value[list(file_upload.value.keys())[0]]['content']
        image = Image.open(io.BytesIO(file_content)).convert('L')
        image_np = np.array(image)
        chunk_size = (chunk_height_slider.value, chunk_width_slider.value)
        morphological_dilation(image_np, chunk_size)

dilate_button.on_click(on_dilate_button_clicked)

# Display widgets
display(file_upload)
display(chunk_size_box)
display(dilate_button)
