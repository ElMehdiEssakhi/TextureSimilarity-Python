# Image Similarity Search

A comprehensive Python project for finding visually similar images in a dataset using multiple distance metrics and texture analysis algorithms.

## 📋 Overview

This project implements and evaluates various image similarity search techniques, from simple pixel-based comparisons to advanced texture feature extraction using GLCM (Gray-Level Co-occurrence Matrix). The work includes:

- **8+ distance metrics** for image comparison
- **Texture analysis** using first-order and second-order statistics
- **Educational notebooks** explaining key concepts like normalization and contrast
- **Performance optimization** through feature precomputation

## 🎯 Key Features

### Distance Metrics Implemented

1. **Pixel-wise Euclidean Distance** - Raw pixel-by-pixel comparison
2. **Color Histogram Distance** - Compares pixel intensity distributions
3. **Average Color Distance** - Global brightness/color comparison
4. **Variance Distance** - Measures pixel intensity spread (Precision: 6/8)
5. **Energy Distance** - Sum of squared pixel values
6. **Entropy Distance** - Information-theoretic approach
7. **GLCM-based Features:**
   - Contrast (Precision: 3/8)
   - Homogeneity (Precision: 6/8)
   - Co-occurrence Matrix (Precision: 2/8)
8. **Texture-based Combined Features** - Optimized feature vector approach

### Dataset

- **66 color images** organized into **6 categories**
- **11 images per category**
- JPEG format for efficient processing


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TextureSimilarity-Python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

Key packages:
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `jupyter` - Interactive notebooks

See `requirements.txt` for complete list.

## 📖 Usage

### Quick Start: Similarity Search

```python
import thisModule as tm
import distances as dist
import matplotlib.pyplot as plt

# Load a query image
query_image = tm.ImageAsArrayNum(7)

# Load all images in database
all_images = tm.getImages(nbr_im=66)

# Find 8 most similar images using variance distance
results = dist.distance_variance(query_image, all_images)

# Visualize results
tm.plotResults(results)
```

### Using Precomputed Features (Optimized)

```python
import thisModule as tm
import distances as ds
import descriptorsV2 as desV2  

# Precompute feature vectors for all images (one-time cost)
all_images = tm.getImages(66)
features = desV2.preComputeAllImagesFeaturesVector(all_images)

# Query with precomputed features (fast)
results = ds.distance_texture_precomputed(46, features)
tm.plotResults(results)
```

### Available Distance Functions

```python
# Single image queries
dist.distance_pixel(query_image)
dist.distance_histogram(query_image, all_images)
dist.distance_average_color(query_image, all_images)
dist.distance_variance(query_image, all_images)
dist.distance_energie_global(query_image, all_images)
dist.disatance_entropie(query_image, all_images)

# GLCM-based metrics
dist.distance_contrast_glcm(query_image, all_images)
dist.distance_homogenite_glcm(query_image, all_images)
dist.distance_cooccurence(query_image, all_images)

# Optimized texture features
dist.distance_texture_precomputed(image_id, precomputed_features)
```

## 📊 Experiment Results

### Phase 1: Baseline Metrics

| Method | Precision |
|--------|-----------|
| Pixel-wise Distance | 3/8 |
| Color Histogram | 3/8 |
| Average Color | 4/8 |

### Phase 2: Texture & Statistical Features

| Method | Precision |
|--------|-----------|
| Energy Distance | 4/8 |
| Entropy Distance | 4/8 |
| **Variance Distance** | **6/8** ⭐ |
| Contrast (GLCM) | 3/8 |
| **Homogeneity (GLCM)** | **6/8** ⭐ |
| Co-occurrence Matrix | 2/8 |

**Best Performers:** Variance and GLCM Homogeneity metrics achieve 6/8 precision.

## 🧠 Educational Notebooks

### 1. Explaining Pixel Normalization (`explainingPixelNormalization.ipynb`)

**Key Insight:** How normalization corrects for brightness differences

### 2. Explaining Histogram Normalization (`explainingHistNormalization.ipynb`)

**Key Insight:** How normalization enables resolution-invariant comparison

## 🔧 Core Modules

### `thisModule.py`

Utility functions for image processing:

```python
ImageAsArrayNum(index)              # Load image by ID
ImageAsArray(name, foldername)      # Load image by filename
rgb2gray(image)                     # Convert RGB to grayscale
normalizeImage(image)               # Min-max normalization
imageToVector(image)                # Flatten 2D to 1D
histo(image)                        # Compute histogram
EuclideanDistance(x, y)             # Euclidean distance metric
getImages(nbr_im)                   # Load entire dataset
plotResults(distances)              # Visualize top 8 matches
```

### `descriptorsV2.py`

Feature extraction algorithms:

**First-Order Features (Global):**
- `variance(image)` - Pixel intensity spread
- `entropie(image)` - Information content
- `average_color(image)` - Mean brightness
- `energia_global(image)` - Sum of squared pixels

**Second-Order Features (GLCM):**
- `cooccurence(image)` - Generate Gray-Level Co-occurrence Matrix
- `contrast_glcm(matrix)` - Texture contrast
- `homogenite_glcm(matrix)` - Texture homogeneity
- `energie_glcm(matrix)` - Texture uniformity

**Combined Vector:**
- `get_final_vector(image)` - Normalized feature vector combining all features

### `distances.py`

Distance metric implementations for similarity search.

## 🎓 How It Works

### Image Comparison Pipeline

1. **Load Query Image** → RGB to grayscale conversion
2. **Normalize** → Min-max scaling to [0, 1]
3. **Extract Features** → Compute texture descriptors
4. **Distance Calculation** → Compare with all images
5. **Sort Results** → Rank by similarity (ascending distance)
6. **Visualize** → Display top 8 matches

### Feature Extraction Strategy

The project uses a hybrid approach combining:

- **First-order statistics:** Variance, entropy (sensitive to overall intensity)
- **Second-order statistics:** GLCM features (sensitive to texture patterns)
- **Normalization:** All features are L2-normalized for scale-invariance

This combination captures both global image properties and local texture characteristics.

## ⚙️ Configuration

### Image Database Format

Images should be stored in `image_database/` as JPEG files:
- Naming: `1.jpg`, `2.jpg`, ..., `66.jpg`
- Format: Standard JPEG (RGB color)
- Organization: 11 images per category (6 categories total)

### Optional: Custom Categories

Edit `thisModule.py` to support different dataset sizes:

```python
def plotFirstFromEachCategory(categoriesCount, CategorySize, lines=2, columns=3):
    # Adjust categoriesCount and CategorySize as needed
```

## 📈 Performance Notes

- **Precomputation:** One-time feature extraction takes ~5-10 seconds for 66 images
- **Query Time (without precomputation):** ~1-2 seconds per query
- **Query Time (with precomputation):** ~100-200ms per query
- **Memory Usage:** ~10MB for precomputed feature vectors

## 🔍 Limitations & Future Work

### Current Limitations

1. **Modest Precision:** Best metrics achieve 6/8 (75%) precision
2. **Small Dataset:** Only 66 images; may not generalize well
3. **Color Information:** Converted to grayscale; color histograms not fully exploited
4. **Single Orientation:** GLCM uses only horizontal direction

### Potential Improvements

- [ ] Deep learning-based embeddings (ResNet, VGG features)
- [ ] Multi-directional GLCM (0°, 45°, 90°, 135°)
- [ ] Color-space analysis (HSV, LAB instead of just grayscale)
- [ ] Larger, more diverse dataset
- [ ] Hybrid ensemble of best-performing metrics
- [ ] Speed optimization with spatial hashing (LSH)

## 📝 License

This project is provided as-is for learning purposes. Use and modify freely.

## 👤 Author

EL MEHDI ES-SAKHI

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

---

**Last Updated:** December 2025
