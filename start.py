from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"/analyze": {
    "origins": ["https://social.memz.live", "http://localhost:3000"],
    "methods": ["POST"],
    "allow_headers": ["Content-Type"],
}})

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()  # Convert numpy scalar to Python type
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    else:
        return obj

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if not file or file.filename.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
        return jsonify({'error': 'Invalid file type'}), 400

    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    try:
        analysis_results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

    # Convert numpy types to native Python types
    converted_results = convert_numpy_types(analysis_results)

    # Draw rectangles and emotion text
    for face in analysis_results:
        x = face["region"]["x"]
        y = face["region"]["y"]
        w = face["region"]["w"]
        h = face["region"]["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (10, 180, 10), 3)
        cv2.putText(img, face["dominant_emotion"], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 180, 10), 3)

    # Convert image to base64
    _, buffer = cv2.imencode('.png', img)
    img_as_text = "data:image/png;base64," + base64.b64encode(buffer).decode()

    return jsonify({'result': converted_results, 'image': img_as_text}), 200

@app.route('/', methods=['GET'])
def main():
    return jsonify({'OK': 'All good'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)