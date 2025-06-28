import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# ========== CONFIGURATION ==========
class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MODEL_PATH = os.path.join('models', 'lung_cnn.pth')  # Default path (can be overridden)
    USE_MOCK_MODEL = True  # Set to False if you have the real model file

app = Flask(__name__)
app.config.from_object(Config)

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# ========== MODEL DEFINITION ==========
class LungCNN(nn.Module):
    def __init__(self):
        super(LungCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

# ========== MODEL LOADING (WITH MOCK SUPPORT) ==========
def load_model():
    model = LungCNN()
    
    if app.config['USE_MOCK_MODEL']:
        print("⚠️ Using MOCK model (no real predictions)")
        return model
    
    model_path = app.config['MODEL_PATH']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("✅ Model loaded successfully!")
    return model

model = load_model()

# ========== IMAGE PROCESSING ==========
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0), img

# ========== PREDICTION ==========
def predict_lung_image(image_tensor):
    if app.config['USE_MOCK_MODEL']:
        return "Mock Result (Normal)", "green", 0.3  # Fake prediction
    
    with torch.no_grad():
        output = model(image_tensor)
        prob = output.item()
        if prob > 0.5:
            return "Abnormal (Possible Lung Cancer)", "red", prob
        else:
            return "Normal", "green", prob

# ========== FLASK ROUTES ==========
@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify(error="No file uploaded"), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No file selected"), 400
    
    if not allowed_file(file.filename):
        return jsonify(error="Invalid file type"), 400
    
    # Save and process image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    try:
        img_tensor, original_img = preprocess_image(filepath)
        result, color, prob = predict_lung_image(img_tensor)
        
        # Generate visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(original_img)
        plt.title(f"Result: {result} (Confidence: {prob:.2f})", color=color)
        plt.axis('off')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return jsonify({
            'result': result,
            'probability': prob,
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

if __name__ == '__main__':
    app.run(debug=True)