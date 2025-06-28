import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
import torch
from torchvision import transforms

# ------------------------ Image Upload & Processing Pipeline ------------------------
def process_uploaded_image(IMG_SIZE):
    # Properly initialize Tkinter for file selection
    root = tk.Tk()
    root.withdraw()
    img_path = askopenfilename(title="Select Lung Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])

    if not img_path:
        print("❌ No image selected.")
        return

    print(f"✅ Selected Image: {img_path}")

    # Load Image
    original_img = cv2.imread(img_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # ----------------- Preprocessing -----------------
    # Convert to Grayscale
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Resize Image
    resized_img = cv2.resize(gray_img, IMG_SIZE)

    # Normalize Image
    normalized_img = resized_img.astype(np.float32) / 255.0

    # ----------------- Noise Analysis -----------------
    laplacian = cv2.Laplacian(resized_img, cv2.CV_64F)
    noise_level = laplacian.var()
    print(f"🔎 Detected Noise Level: {noise_level:.2f}")

    # ----------------- Noise Removal -----------------
    median_filtered = cv2.medianBlur(resized_img, 5)
    bilateral_filtered = cv2.bilateralFilter(resized_img, 9, 75, 75)

    # ----------------- Edge Detection -----------------
    edges = cv2.Canny(resized_img, 50, 150)

    # ----------------- Segmentation -----------------
    _, thresh_img = cv2.threshold(median_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    segmented_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)

    # ----------------- Visualization -----------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(resized_img, cmap='gray')
    axs[0, 1].set_title('Grayscale & Resized')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(median_filtered, cmap='gray')
    axs[0, 2].set_title('Median Filtered')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(bilateral_filtered, cmap='gray')
    axs[1, 0].set_title('Bilateral Filtered')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(edges, cmap='gray')
    axs[1, 1].set_title('Edge Detection')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(segmented_img)
    axs[1, 2].set_title('Segmented Image with Contours')
    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.show()

    return normalized_img, segmented_img, edges

# Example Call
IMG_SIZE = (224, 224)  # Adjust as per model requirements
processed_data = process_uploaded_image(IMG_SIZE)
