import cv2 
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.filters import sobel, prewitt, scharr, roberts
from skimage.morphology import disk, opening, closing
from skimage.segmentation import watershed, chan_vese, slic, quickshift, felzenszwalb
from skimage.color import rgb2gray, label2rgb
from skimage.feature import canny

# Function to open a file selection window
def select_image():
    root = tk.Tk()
    root.withdraw()  # Hide main window
    file_path = filedialog.askopenfilename(title="Select a Lung Scan Image",
                                           filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    return file_path

def apply_colormap(image, colormap=cv2.COLORMAP_JET):
    """Applies a color map to enhance visualization."""
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(image, colormap)

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    if img is None:
        raise FileNotFoundError(f"Error: Unable to load image '{image_path}'. Please check the file!")

    img = cv2.resize(img, (256, 256))  # Resize for consistency
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply various filters and processing techniques with colormap
    filters = [
        ("Original", img),
        ("Grayscale", apply_colormap(gray, cv2.COLORMAP_JET)),
        ("Gaussian Blur", apply_colormap(cv2.GaussianBlur(gray, (5, 5), 0), cv2.COLORMAP_HOT)),
        ("Median Blur", apply_colormap(cv2.medianBlur(gray, 5), cv2.COLORMAP_JET)),
        ("Bilateral Filter", apply_colormap(cv2.bilateralFilter(gray, 9, 75, 75), cv2.COLORMAP_PARULA)),
        ("Canny Edges", apply_colormap(canny(gray, sigma=1).astype(np.uint8) * 255, cv2.COLORMAP_JET)),
        ("Sobel Filter", apply_colormap(sobel(gray) * 255, cv2.COLORMAP_JET)),
        ("Prewitt Filter", apply_colormap(prewitt(gray) * 255, cv2.COLORMAP_HOT)),
        ("Scharr Filter", apply_colormap(scharr(gray) * 255, cv2.COLORMAP_PARULA)),
        ("Roberts Filter", apply_colormap(roberts(gray) * 255, cv2.COLORMAP_HOT)),
        ("Adaptive Threshold", apply_colormap(cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))),
        ("Otsu Threshold", apply_colormap(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])),
        ("Laplacian Edge Detection", apply_colormap(cv2.Laplacian(gray, cv2.CV_64F), cv2.COLORMAP_HOT)),
        ("Histogram Equalization", apply_colormap(cv2.equalizeHist(gray), cv2.COLORMAP_JET)),
        ("CLAHE (Contrast Adaptive Histogram)", apply_colormap(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), cv2.COLORMAP_HOT))
    ]

    # Apply segmentation techniques with color
    seg_gray = rgb2gray(img)
    segmentation = [
        ("Otsu Segmentation", apply_colormap(cv2.threshold((seg_gray * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])),
        ("Canny Edge Segmentation", apply_colormap(canny(seg_gray, sigma=1).astype(np.uint8) * 255, cv2.COLORMAP_HOT)),
        ("Watershed Segmentation", apply_colormap(watershed(seg_gray, markers=250, compactness=0.01), cv2.COLORMAP_JET)),
        ("Chan-Vese Segmentation", apply_colormap(chan_vese(seg_gray, mu=0.1, lambda1=1, lambda2=1, tol=1e-3).astype(np.uint8) * 255, cv2.COLORMAP_PARULA)),
        ("SLIC Segmentation", label2rgb(slic(img, n_segments=250, compactness=10, sigma=1), img, kind='avg')),
        ("Quickshift Segmentation", label2rgb(quickshift(img, kernel_size=3, max_dist=6), img, kind='avg')),
        ("Felzenszwalb Segmentation", label2rgb(felzenszwalb(img, scale=100, sigma=0.5, min_size=50), img, kind='avg'))
    ]

    return img, filters, segmentation

def plot_results(title, images):
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    axes = axes.ravel()

    for i, (filter_title, image) in enumerate(images[:24]):  # Show first 24 filters
        if len(image.shape) == 2:  # Grayscale images
            axes[i].imshow(image, cmap='gray')
        else:  # Color images
            axes[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[i].set_title(filter_title, fontsize=8)
        axes[i].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def detect_cancer():
    image_path = select_image()
    
    if not image_path:
        print("No image selected. Please try again.")
        return
    
    original, processed_images, segmentations = preprocess_image(image_path)

    # Display filter results
    plot_results("Image Processing Filters", processed_images)
    
    # Display segmentation results
    plot_results("Image Segmentation Techniques", segmentations)

    # Simulate cancer detection (Replace with ML model later)
    diagnosis = np.random.choice(["Normal", "Cancer"])
    cancer_stage = np.random.choice(["Stage 1", "Stage 2", "Stage 3", "Stage 4"]) if diagnosis == "Cancer" else None
    
    # Show result
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f"Diagnosis: {diagnosis}" + (f" ({cancer_stage})" if cancer_stage else ""))
    plt.axis("off")
    plt.show()

# Run the analysis
detect_cancer()
