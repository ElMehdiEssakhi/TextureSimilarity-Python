import numpy as np
import thisModule as tm
import descriptors as ds

def find_similar_images(query, images, feature_extractor, setSize=66, resultNumber=8):
    """
    Generic function to find similar images based on ANY feature.
    
    Args:
        query: The query image.
        images: The database of images (3D matrix).
        feature_extractor: A FUNCTION that takes an image and returns a feature vector.
    """
    
    # 1. Calculate the feature for the Query image ONCE
    # We apply the feature_extractor function to the query
    query_feature = feature_extractor(query)
    
    # Flatten if it's a matrix (like GLCM or Histogram) to ensure Euclidean works
    if isinstance(query_feature, np.ndarray):
        query_feature = query_feature.flatten()

    D = []
    
    # 2. Loop through database
    for i in range(setSize):
        # Extract the single image from the stack
        db_img = images[:, :, i]
        
        # Calculate feature for the database image using the SAME function
        db_feature = feature_extractor(db_img)
        
        if isinstance(db_feature, np.ndarray):
            db_feature = db_feature.flatten()
            
        # 3. Calculate Distance
        dist = tm.EuclideanDistance(db_feature, query_feature)
        
        couple = [i + 1, dist]
        D.append(couple)
        
    return sorted(D, key=lambda item: item[1])[:resultNumber]

# Define the "Recipe" for Energy
def get_energy(img):
    # 1. Grayscale, 2. Normalize, 3. Calculate Energy
    return ds.energie(tm.normalizeImage(tm.rgb2gray(img)))
# Define the "Recipe" for Co-occurrence
def get_cooccurence(img):
    # 1. Grayscale, 2. Normalize, 3. Scale to 255, 4. Cast to uint8
    norm = tm.normalizeImage(tm.rgb2gray(img))
    uint8_img = (norm * 255).astype(np.uint8)
    return ds.cooccurence(uint8_img)
# Define the "Recipe" for Histogram
def get_histogram(img):
    norm = tm.normalizeImage(tm.rgb2gray(img))
    hist = tm.ds.histo(norm)
    return hist / norm.size # Return normalized histogram