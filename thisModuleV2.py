import matplotlib.pyplot as plt
import numpy as np

def rgb2gray(I):
    # Handle PNGs that load as floats 0-1
    if I.max() <= 1.0:
        I = (I * 255).astype(np.uint8)
    
    gray = 0.2989 * I[:,:,0] + 0.5870 * I[:,:,1] + 0.1140 * I[:,:,2]
    return gray.astype(np.uint8)

def normalizeImage(I):
    I = I.astype(float) # Ensure we don't do integer division
    Imin = I.min()
    Imax = I.max()
    
    # Avoid division by zero
    if Imax == Imin:
        return np.zeros_like(I)
        
    return (I - Imin) / (Imax - Imin)

def getImages(nbr_im, filename='image_database/'):
    # Load first image to get dimensions
    I = plt.imread(f'{filename}1.jpg')
    gray_template = rgb2gray(I)
    nl, nc = gray_template.shape
    
    # Pre-allocate matrix
    images = np.zeros((nl, nc, nbr_im))
    
    for i in range(nbr_im):
        try:
            curr_img = plt.imread(f'{filename}{i+1}.jpg')
            curr_gray = rgb2gray(curr_img)
            
            # CRITICAL: Ensure dimensions match
            if curr_gray.shape != (nl, nc):
                print(f"Warning: Image {i+1} size mismatch. Resizing not implemented.")
                # Ideally, you would use cv2.resize() here
                continue
                
            images[:,:,i] = curr_gray
        except FileNotFoundError:
            print(f"Error: Image {i+1} not found.")
            
    return images
