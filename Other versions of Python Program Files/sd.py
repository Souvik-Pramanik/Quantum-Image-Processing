

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Checkerboard pattern
single_dot_image = np.array([
    [0, 0, 0, 0],
    [0, 255, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

def save_image(image, filename):
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save(filename)

def display_image(image):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Save and display the checkerboard image
save_image(single_dot_image, 'single_dot_image.png')
display_image(single_dot_image)
