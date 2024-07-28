import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_random_image(size=(4, 4)):
    """
    Generate a random 4x4 black and white image with 8-bit pixels.
    """
    image = np.random.choice([0, 255], size=size)
    return image

def create_specific_image(pattern):
    """
    Create a specific 4x4 black and white image with 8-bit pixels based on a provided pattern.
    
    Args:
    pattern (list of lists): A 4x4 list containing 0s and 255s.
    
    Returns:
    numpy.ndarray: A 4x4 black and white image.
    """
    if len(pattern) != 4 or any(len(row) != 4 for row in pattern):
        raise ValueError("Pattern must be a 4x4 list of lists containing 0s and 255s.")
    image = np.array(pattern)
    return image

def save_image(image, filename):
    """
    Save the black and white image as a PNG file.
    
    Args:
    image (numpy.ndarray): The black and white image to save.
    filename (str): The file path where the image will be saved.
    """
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image.save(filename)

def display_image(image):
    """
    Display the black and white image using matplotlib.
    
    Args:
    image (numpy.ndarray): The black and white image to display.
    """
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a random 4x4 black and white image
    random_image = generate_random_image()
    print("Random Image:")
    print(random_image)
    display_image(random_image)
    save_image(random_image, 'random_4x4_image.png')
    
    # Create a specific 4x4 black and white image
    specific_pattern = [
        [255, 0, 255, 0],
        [0, 255, 0, 255],
        [255, 0, 255, 0],
        [0, 255, 0, 255]
    ]
    specific_image = create_specific_image(specific_pattern)
    print("Specific Image:")
    print(specific_image)
    display_image(specific_image)
    save_image(specific_image, 'specific_4x4_image.png')


# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt

# # Checkerboard pattern
# checkerboard_image = np.array([
#     [255, 0, 255, 0],
#     [0, 255, 0, 255],
#     [255, 0, 255, 0],
#     [0, 255, 0, 255]
# ])

# def save_image(image, filename):
#     pil_image = Image.fromarray(image.astype(np.uint8))
#     pil_image.save(filename)

# def display_image(image):
#     plt.imshow(image, cmap='gray', vmin=0, vmax=255)
#     plt.show()

# # Save and display the checkerboard image
# save_image(checkerboard_image, 'checkerboard_4x4.png')
# display_image(checkerboard_image)
