from flask import Flask, request, jsonify
import logging
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from google.cloud import vision
from utils import convert_to_image  # Assuming this is a custom utility function
from flask_cors import CORS
import io
import cv2
import re

# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set logging level to DEBUG
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cic-project001-87c84597f0f6.json'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://localhost:3000"]}})
app.config['UPLOAD_FOLDER'] = 'uploads/'
vision_client = vision.ImageAnnotatorClient()

latest_extracted_data = None  # Store the latest extracted data
# Load the pre-trained model
def load_document_classifier_model():
    model_path = 'document_classifier_model1.h5'
    return load_model(model_path)

model = load_document_classifier_model()

@app.route('/classify', methods=['POST'])
def classify_documents():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    files = request.files.getlist('file')
    results = []

    for file_obj in files:
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_obj.filename))
        try:
            file_obj.save(temp_file_path)
            images = convert_to_image(temp_file_path)
            for img in images:
                img_array = np.array(img.resize((224, 224))) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction = model.predict(img_array)
                predicted_category_index = np.argmax(prediction)
                confidence_score = float(np.max(prediction))

                extracted_data = classify_and_extract(temp_file_path, predicted_category_index)
                logging.info(f'Extracted data for {file_obj.filename}: {extracted_data}')

                results.append({
                    'filename': file_obj.filename,
                    'category_index': int(predicted_category_index),
                    'confidence': float(confidence_score),
                    'data': extracted_data
                })
        finally:
            os.remove(temp_file_path)
            
    global latest_extracted_data
    latest_extracted_data = extracted_data 

    return jsonify(results)

@app.route('/api/extracted-data', methods=['GET'])
def get_extracted_data():
    global latest_extracted_data
    if latest_extracted_data:
        return jsonify(latest_extracted_data)
    else:
        return jsonify({'error': 'No extracted data available'}), 404
    
    
    
def classify_and_extract(image_path, category_index):
    patterns = {
        1: {  # ID_FRONT
            'serial_number': r"SERIAL NUMBER:\s*(\d+)",
            'full_names': r"FULL NAMES\n([^\n]+)",
            'id_number': r"ID NUMBER:\s*(\d+)",
            'date_of_birth': r"DATE OF BIRTH\s+(\d{2}\.\d{2}\.\d{4})",
            'sex': r"SEX\n(\w+)"
        },
        2: {  # KRA
            #'tax_pin': r'[A-Z]\d{9}[A-Z]'
            'tax_pin': r'\b([A-Z]\d{9}[A-Z])\b'

        }
    }.get(category_index, {})

    if not patterns:
        return {'info': 'Text extraction not performed for this category.'}

    return extract_text_data(image_path, patterns)

def extract_text_data(image_path, patterns):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = vision_client.text_detection(image=image)
    texts = response.text_annotations

    data = {}
    if texts:
        description = texts[0].description
        logging.debug(f"OCR Output: {description}")
        for key, pattern in patterns.items():
            match = re.search(pattern, description, re.MULTILINE)
            if match:
                try:
                    extracted_value = match.group(1).strip()
                    if key == 'full_names':
                        name_parts = extracted_value.split()
                        data['first_name'] = name_parts[0] if len(name_parts) > 0 else 'Not found'
                        data['other_names'] = ' '.join(name_parts[1:-1]) if len(name_parts) > 2 else 'Not found'
                        data['surname_name'] = name_parts[-1] if len(name_parts) > 2 else 'Not found'
                    else:
                        data[key] = extracted_value
                except IndexError:
                    logging.error(f"Failed to extract group for {key} with pattern: {pattern}")
                    data[key] = 'Not found'
            else:
                logging.warning(f"No match found for {key} with pattern: {pattern}")
                data[key] = 'Not found'
    return data

if __name__ == "__main__":
    app.run(debug=True)
