from PIL import Image
import matplotlib.pyplot as plt

# Path to the output image
output_path = "outputs/styled_image.jpg"

# Load and display the image
image = Image.open(output_path)
plt.imshow(image)
plt.axis('off')  # Turn off axes
plt.show()
