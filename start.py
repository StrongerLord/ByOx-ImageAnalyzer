from flask import Flask, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import cv2
import numpy as np
import io
import base64

app = Flask(__name__)
CORS(app, resources={r"analyze": {
	"origins": ["https://social.memz.live"],
	"methods": ["POST"],
	"allow_headers": ["Content-Type"],
}})

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
	color_image_flag = 1
	img = cv2.imdecode(data, color_image_flag)

	try:
		result = DeepFace.analyze(img, actions=['emotion'])
	except ValueError:
		return jsonify({'error': 'No face detected in the image'}), 400

	# Draw rectangles around detected faces and put emotion text
	for face in result:
		cv2.rectangle(img, (face["region"]["x"], face["region"]["y"]), (face["region"]["x"] + face["region"]["w"], face["region"]["y"] + face["region"]["h"]), (10, 180, 10), 3)
		cv2.putText(img, face["dominant_emotion"], (face["region"]["x"], face["region"]["y"]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (10, 180, 10), 3)

	# Convert the image to base64 and return
	retval, buffer = cv2.imencode('.png', img)
	img_as_text = "data:image/png;base64," + base64.b64encode(buffer).decode()

	return jsonify({'result': result, 'image': img_as_text}), 200

@app.route('/', methods=['GET'])
def main():
	return jsonify({'OK': 'All good'}), 200


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)
