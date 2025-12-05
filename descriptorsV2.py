import numpy as np
import thisModule as tm
# --- GROUP 1: GLOBAL FEATURES (Input = Raw Image) ---

def histo(img):
    hitsreq,binsreq=np.histogram(img,256,[0, 256])
    return hitsreq

def average_color(img): # brightness
    return np.mean(img)

def variance(image):
    return np.var(image)

def entropie(I):
    counts= tm.histo(I)
    p = counts / counts.sum()
    p = p[p > 0] # remove zeros
    return -np.sum(p * np.log2(p))

def contrast_rms(I):
    # Renamed to avoid confusion with GLCM contrast
    return np.std(I)

def energie_global(I):
    # Energy of the IMAGE (sum of squared pixels)
    # Caution: This value depends heavily on image size!
    return np.sum(I.astype(float)**2)

# --- GROUP 2: GLCM GENERATION ---

def cooccurence(I):
    rows, cols = I.shape
    # FIX: Size must be 256x256 to hold values 0-255
    CC = np.zeros((256, 256)).astype(np.uint32)
    
    # Simple Horizontal (0 degree) GLCM
    for i in range(rows):
        for j in range(cols - 1):
            pixel_val = I[i, j]
            neighbor_val = I[i, j + 1]
            CC[pixel_val, neighbor_val] += 1
            
    # OPTIONAL: Normalize GLCM so features don't depend on image size
    CC = CC / CC.sum() 
    return CC

def cooccurence_fast(I):
    I1 = I[:, :-1].ravel()
    I2 = I[:, 1:].ravel()
    CC = np.zeros((256, 256), dtype=np.uint32)
    np.add.at(CC, (I1, I2), 1)
    return CC


# --- GROUP 3: TEXTURE FEATURES (Input = GLCM Matrix) ---

def contrast_glcm(GLCM):
    # Input must be the MATRIX from cooccurence(), not the image
    rows, cols = GLCM.shape
    s = 0
    for i in range(rows):
        for j in range(cols):
            # i and j are Intensity Values here
            s += GLCM[i, j] * ((i - j) ** 2)
    return s

def homogenite_glcm(GLCM):
    rows, cols = GLCM.shape
    mo = 0
    for i in range(rows):
        for j in range(cols):
            mo += GLCM[i, j] / (1 + abs(i - j))
    return mo

def energie_glcm(GLCM):
    # Energy of the TEXTURE (Uniformity)
    return np.sum(GLCM**2)

# --- MASTER FUNCTION ---

def get_final_vector(image):
    # --- PART 1: First Order (Global) ---
    # Good for lighting and overall complexity
    var_global = variance(image)
    ent_global = entropie(image) # The histogram-based one
    
    # --- PART 2: Second Order (GLCM) ---
    # Good for patterns (stripes, spots, wood grain)
    
    # 1. Prepare image for GLCM (must be integer 0-255)
    #img_uint8 = (tm.normalizeImage(image) * 255).astype(np.uint8)
    
    # 2. Generate Matrix
    matrix = cooccurence(image)
    
    # 3. Extract Features from Matrix
    contrast_tex = contrast_glcm(matrix)
    homogen_tex = homogenite_glcm(matrix)
    energy_tex = energie_glcm(matrix)
    
    # --- PART 3: Combine ---
    # This is your data point for Machine Learning
    feature_vector = [var_global, ent_global, contrast_tex, homogen_tex, energy_tex]
    norm = np.linalg.norm(feature_vector)
    feature_vector = feature_vector / norm

    return np.array(feature_vector)