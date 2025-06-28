import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
import torch
from torchvision import transforms

# ------------------------ Image Upload & Processing Pipeline ------------------------
def process_uploaded_image(IMG_SIZE):
    # ✅ 1. Image Acquisition
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

    # ✅ 2. Image Enhancement
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)

    # Contrast Enhancement
    contrast_stretch = cv2.normalize(gray_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    hist_equalized = cv2.equalizeHist(gray_img)

    # ✅ 3. Image Restoration
    gaussian_denoise = cv2.GaussianBlur(gray_img, (5, 5), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(gray_img, -1, sharpen_kernel)

    # ✅ 4. Color Image Processing
    hsv_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2HSV)
    lab_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2LAB)

    # ✅ 5. Morphological Processing
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(gray_img, kernel, iterations=1)
    dilated_img = cv2.dilate(gray_img, kernel, iterations=1)
    opened_img = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
    closed_img = cv2.morphologyEx(gray_img, cv2.MORPH_CLOSE, kernel)

    # ✅ 6. Segmentation & Representation
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmented_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(segmented_img, contours, -1, (0, 255, 0), 2)

    # ✅ 7. Image Recognition (Placeholder for ML Model)
    print("🧠 Image Recognition Step - Can be implemented with ML Model")

    # ----------------- Visualization -----------------
    fig, axs = plt.subplots(3, 4, figsize=(18, 12))
    
    axs[0, 0].imshow(original_img)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(gray_img, cmap='gray')
    axs[0, 1].set_title('Grayscale')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(contrast_stretch, cmap='gray')
    axs[0, 2].set_title('Contrast Stretched')
    axs[0, 2].axis('off')

    axs[0, 3].imshow(hist_equalized, cmap='gray')
    axs[0, 3].set_title('Histogram Equalization')
    axs[0, 3].axis('off')

    axs[1, 0].imshow(gaussian_denoise, cmap='gray')
    axs[1, 0].set_title('Gaussian Denoised')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(sharpened_img, cmap='gray')
    axs[1, 1].set_title('Sharpened Image')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(hsv_img)
    axs[1, 2].set_title('HSV Image')
    axs[1, 2].axis('off')

    axs[1, 3].imshow(lab_img)
    axs[1, 3].set_title('LAB Image')
    axs[1, 3].axis('off')

    axs[2, 0].imshow(opened_img, cmap='gray')
    axs[2, 0].set_title('Morphological Opening')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(closed_img, cmap='gray')
    axs[2, 1].set_title('Morphological Closing')
    axs[2, 1].axis('off')

    axs[2, 2].imshow(thresh_img, cmap='gray')
    axs[2, 2].set_title('Threshold Segmentation')
    axs[2, 2].axis('off')

    axs[2, 3].imshow(segmented_img)
    axs[2, 3].set_title('Contour Detection')
    axs[2, 3].axis('off')

    plt.tight_layout()
    plt.show()

    return segmented_img  # Return segmented image for further use

# Example Call
IMG_SIZE = (224, 224)  # Adjust based on model input requirements
processed_data = process_uploaded_image(IMG_SIZE)
