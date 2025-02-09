import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Image upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16MB

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Load the trained model
model = tf.keras.models.load_model('mobilnetV3model.keras')

# Disease class labels
class_labels = [
    'Acne and Rosacea Photos',
    'Atopic Dermatitis Photos',
    'Eczema Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Psoriasis pictures Lichen Planus and related diseases'
]

# Disease-related deficiencies and advice
disease_data = {
    "Acne and Rosacea Photos": {
        "deficiency": ["Zinc", "Vitamin A", "Vitamin E", "Omega-3"],
        "advice": "Increase intake of Zinc, Vitamin A, and Omega-3 by consuming nuts, carrots, and fatty fish."
    },
    "Eczema Photos": {
        "deficiency": ["Vitamin D", "Omega-3", "Zinc", "Vitamin B6"],
        "advice": "Include Vitamin D, Omega-3, and Zinc-rich foods such as fish, eggs, and seeds in your diet."
    },
    "Atopic Dermatitis Photos": {
        "deficiency": ["Vitamin D", "Zinc", "Omega-3"],
        "advice": "Increase Omega-3 intake from sources like salmon, flaxseeds, and walnuts."
    },
    "Light Diseases and Disorders of Pigmentation": {
        "deficiency": ["Vitamin B12", "Copper", "Zinc"],
        "advice": "Consume foods high in Vitamin B12 and Copper, such as dairy, shellfish, and leafy greens."
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "deficiency": ["Vitamin D", "Omega-3", "Selenium", "Vitamin E"],
        "advice": "Eat foods rich in Vitamin D and Omega-3, including salmon, sunflower seeds, and nuts."
    }
}

def allowed_file(filename):
    """Check if the uploaded file has a valid extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    """Preprocess the image to match the model's input format."""
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Apply MobileNetV3 preprocessing
    return img_array

def predict_disease(img_path):
    """Predict the skin disease from the image."""
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class

@app.route('/')
def home():
    """Render the homepage."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle image upload, process it, and return disease prediction with advice."""
    if 'file' not in request.files:
        return jsonify({"error": "❌ No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "❌ No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict the disease
        disease = predict_disease(file_path)

        # Remove the image after processing
        os.remove(file_path)

        # Retrieve disease-related information
        disease_info = disease_data.get(disease, {"deficiency": [], "advice": "No specific advice available."})

        return jsonify({
            "disease": disease,
            "deficiency": disease_info["deficiency"],
            "advice": disease_info["advice"]
        })
    
    return jsonify({"error": "❌ Invalid file type"}), 400


@app.route('/get-advice', methods=['GET'])
def get_advice():
    disease = request.args.get("disease")
    if not disease or disease not in disease_data:
        return jsonify({"error": "❌ No advice available for this condition"}), 400
    
    return jsonify({"advice": disease_data[disease]["advice"]})


if __name__ == '__main__':
    app.run(debug=True)
