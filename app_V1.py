import logging
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from utils import convert_to_image  # Assuming this is a custom utility function
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG

app = Flask(__name__)
CORS(app) 
app.config['UPLOAD_FOLDER'] = 'uploads/'
# Category names based on index
categories = ["ID_BACK", "ID_FRONT", "KRA", "PASSPORT"]
# Load the pre-trained model
def load_document_classifier_model():
    model_path = 'document_classifier_model1.h5'
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    return load_model(model_path)

model = load_document_classifier_model()

@app.route('/', methods=['GET'])
def index():
    return "Welcome to the Flask backend!"

@app.route('/classify', methods=['POST'])
def classify_documents():
    # Ensure that files are included in the request
    if 'file' not in request.files:
        error_message = 'No file provided'
        logging.error(error_message)
        return jsonify({'error': error_message}), 400

    files = request.files.getlist('file')  # Get list of files

    results = []

    for file_obj in files:
        # Save the file to a temporary location
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_obj.filename))
        file_obj.save(temp_file_path)

        # Convert the file to image(s)
        images = convert_to_image(temp_file_path)
        
        # Resize images to the expected size (224x224)
        resized_images = [img.resize((224, 224)) for img in images]

        # Perform prediction for each resized image
        for j, img in enumerate(resized_images):
            img_array = np.array(img) / 255.  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Perform prediction
            prediction = model.predict(img_array)
            predicted_category_index = np.argmax(prediction)
            confidence_score = float(np.max(prediction))  # Convert to a serializable type
            predicted_category = categories[predicted_category_index]
            result = {
                'filename': file_obj.filename,
                #'page': j + 1,
                #'predicted_category_index': int(predicted_category_index),  # Convert to int for clarity
                'confidence_score': confidence_score,
                'predicted_category': predicted_category
            }
            results.append(result)

        # Remove the temporary uploaded file after classification
        os.remove(temp_file_path)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
