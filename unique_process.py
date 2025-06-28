import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from torchvision import transforms
from skimage import filters, restoration

# Constants
IMG_SIZE = (128, 128)  # Image resize size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Disease classes
CLASSES = {
    0: "Normal",
    1: "Pneumonia",
    2: "Tuberculosis",
    3: "Lung Cancer",
    4: "Fibrosis",
    5: "Edema",
    6: "Pleural Effusion",
    7: "Emphysema",
    8: "Bronchitis",
    9: "Sarcoidosis",
    10: "Asthma",
    11: "Interstitial Lung Disease"
}

def apply_filters(image):
    """Applies multiple filtering techniques to the given image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Histogram Equalization (Color) - Corrected
    hsv_img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    v_equalized = cv2.equalizeHist(v)  # Apply equalization only to the V channel
    hsv_equalized = cv2.merge([h, s, v_equalized])
    rgb_equalized = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2RGB)

    # Pseudo-coloring using OpenCV colormaps
    pseudo_colored_jet = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    pseudo_colored_hot = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    pseudo_colored_cool = cv2.applyColorMap(gray, cv2.COLORMAP_COOL)
    pseudo_colored_spring = cv2.applyColorMap(gray, cv2.COLORMAP_SPRING)
    pseudo_colored_autumn = cv2.applyColorMap(gray, cv2.COLORMAP_AUTUMN)
    pseudo_colored_winter = cv2.applyColorMap(gray, cv2.COLORMAP_WINTER)

    # Deep Learning Filter (Example: Random noise for demonstration)
    # Replace this with an actual deep learning model (e.g., denoising autoencoder)
    deep_learning_filter = np.clip(gray + np.random.normal(0, 25, gray.shape), 0, 255).astype(np.uint8)

    filters_list = {
        "Original": gray,
        "Gaussian Blur": cv2.GaussianBlur(gray, (5, 5), 0),
        "Median Blur": cv2.medianBlur(gray, 5),
        "Bilateral Filter": cv2.bilateralFilter(gray, 9, 75, 75),
        "Sobel X": cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5),
        "Sobel Y": cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5),
        "Laplacian": cv2.Laplacian(gray, cv2.CV_64F),
        "Canny Edge": cv2.Canny(gray, 100, 200),
        "Unsharp Masking": filters.unsharp_mask(gray, radius=1, amount=1),
        "Wavelet Transform": restoration.denoise_wavelet(gray),
        "Guided Filter": cv2.ximgproc.guidedFilter(gray, gray, 8, 500),
        "Adaptive Threshold": cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2),
        "Equalized Histogram": cv2.equalizeHist(gray),
        "Non-Local Means": cv2.fastNlMeansDenoising(gray, h=10),
        "Retinex (Low-Light)": np.clip(np.log1p(gray) * 20, 0, 255).astype(np.uint8),
        "CLAHE": cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray),
        "Scharr X": cv2.Scharr(gray, cv2.CV_64F, 1, 0),
        "Scharr Y": cv2.Scharr(gray, cv2.CV_64F, 0, 1),
        "Prewitt X": filters.prewitt_h(gray),
        "Prewitt Y": filters.prewitt_v(gray),
        "Roberts": filters.roberts(gray),
        "Gabor Filter": cv2.getGaborKernel((21, 21), 5, np.pi/4, 10, 0.5, 0, ktype=cv2.CV_32F),
        "Log Transform": np.uint8(255 * (np.log1p(gray) / np.log(1 + 255))),
        "Histogram Equalization (Color)": rgb_equalized,
        "Pseudo-Coloring (Jet)": cv2.cvtColor(pseudo_colored_jet, cv2.COLOR_BGR2RGB),
        "Pseudo-Coloring (Hot)": cv2.cvtColor(pseudo_colored_hot, cv2.COLOR_BGR2RGB),
        "Pseudo-Coloring (Cool)": cv2.cvtColor(pseudo_colored_cool, cv2.COLOR_BGR2RGB),
        "Pseudo-Coloring (Spring)": cv2.cvtColor(pseudo_colored_spring, cv2.COLOR_BGR2RGB),
        "Pseudo-Coloring (Autumn)": cv2.cvtColor(pseudo_colored_autumn, cv2.COLOR_BGR2RGB),
        "Pseudo-Coloring (Winter)": cv2.cvtColor(pseudo_colored_winter, cv2.COLOR_BGR2RGB),
        "Deep Learning Filter": deep_learning_filter
    }

    return filters_list


def predict_disease(image):
    """Uses CNN model to classify lung diseases."""
    normalized_img = image.astype(np.float32) / 255.0

    # CNN Prediction
    cnn_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor()
    ])(image).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_output = torch.rand(1, len(CLASSES))  # Replace with cnn_model(cnn_input)
        cnn_pred = torch.argmax(F.softmax(cnn_output, dim=1), 1).item()

    return cnn_pred


def process_uploaded_image():
    """Handles image upload, filtering, and classification."""
    # Image upload
    Tk().withdraw()
    img_path = askopenfilename(title="Select Lung Image")

    if not img_path:
        print("No image selected")
        return

    # Load image
    original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    # Apply filters
    filtered_images = apply_filters(original_img)

    # ------------------- Frame 1: Display 24 Filtered Images -------------------
    fig, axs = plt.subplots(4, 6, figsize=(18, 12))  # 4 rows, 6 columns
    for ax, (title, img) in zip(axs.flatten(), filtered_images.items()):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(title, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # ------------------- Frame 2: Final Diagnosis -------------------
    cnn_pred = predict_disease(original_img)
    disease_name = CLASSES[cnn_pred]

    result_img = cv2.resize(original_img, IMG_SIZE)
    cv2.putText(result_img, f"Diagnosis: {disease_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(result_img)
    plt.title("Final Diagnosis Result")
    plt.axis('off')
    plt.show()


# Run the function
process_uploaded_image()