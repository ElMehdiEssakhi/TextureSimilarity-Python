import matplotlib.pyplot as plt
import numpy as np
# --- PARAMETERS FOR LOW CONTRAST ---
height, width = 100, 100
min_intensity = 200  # The darkest point in the image
max_intensity = 250  # The brightest point in the image
# The difference between these is 50, which is low contrast compared to 255.

# 1. Create a 2D NumPy array (100x100 low-contrast gradient)
# Create a linear space from 100 to 150
high_contrast_gradient_1d = np.linspace(min_intensity, max_intensity, width, dtype=np.float32)

# Tile it vertically to create the 2D array
high_contrast_array = np.tile(high_contrast_gradient_1d, (height, 1))

# Final cast to np.uint8 (0-255 integer type)
high_contrast_array = high_contrast_array.astype(np.uint8)

# 2. Display the array as an image
plt.figure(figsize=(6, 6))
# The vmin/vmax should be 0/255 to properly show the narrow range visually
plt.imshow(high_contrast_array, cmap='gray', vmin=0, vmax=255) 
plt.title("High Contrast Grayscale Image (Range 200-250)")
plt.colorbar(label='Intensity')
plt.axis('off')
plt.show()

# 3. Save the image
plt.imsave("high_contrast_image.png", high_contrast_array, cmap='gray')