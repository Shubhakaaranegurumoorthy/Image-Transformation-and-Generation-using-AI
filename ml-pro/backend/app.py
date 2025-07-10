# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import requests
# import cv2
# import numpy as np
# import pytesseract

# app = Flask(__name__)
# CORS(app) # Allow frontend requests

# # 🔹 API Keys (Replace with your actual API keys)
# DEEPAI_API_KEY = "53731fc7-2a63-4ce8-aa81-688d647cb9a0"
# FAL_API_KEY = "ca4e1063-c637-4db8-9734-80c2540cffa2:dbf88ab51874221e262c906a544988ec"

# # 1️⃣ Image Colorization using DeepAI
# @app.route('/colorize', methods=['POST'])
# def colorize_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     file = request.files['image']
#     response = requests.post(
#     "https://api.deepai.org/api/colorizer",
#     files={'image': file},
#     headers={'api-key': DEEPAI_API_KEY}
#     )

#     if response.status_code == 200:
#         return jsonify(response.json())
#     else:
#         return jsonify({"error": "Failed to process image"}), 500

# # 2️⃣ Image Decolorization (Convert to Grayscale)
# @app.route('/decolorize', methods=['POST'])
# def decolorize_image():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     file = request.files['image']
#     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, buffer = cv2.imencode('.png', gray_image)

#     return buffer.tobytes(), 200, {'Content-Type': 'image/png'}

# # 3️⃣ Text-to-Image using FAL AI
# @app.route('/text-to-image', methods=['POST'])
# def text_to_image():
#     data = request.get_json()
#     text = data.get("text")

#     if not text:
#         return jsonify({"error": "Text description is required"}), 400

#     fal_api_url = "https://api.fal.ai/v1/text-to-image"
#     headers = {
#         "Authorization": f"Bearer {FAL_API_KEY}",
#         "Content-Type": "application/json"
#      }

#     payload = {
#         "prompt": text,
#         "model": "stable-diffusion-xl"
# }

#     response = requests.post(fal_api_url, json=payload, headers=headers)

#     if response.status_code == 200:
#         image_url = response.json().get("image_url")
#         return jsonify({"image_url": image_url})
#     else:
#         return jsonify({"error": "Failed to generate image"}), 500

# # 4️⃣ Image-to-Text (OCR)
# @app.route('/image-to-text', methods=['POST'])
# def image_to_text():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image file provided"}), 400

#     file = request.files['image']
#     image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
#     text = pytesseract.image_to_string(image)

#     return jsonify({"extracted_text": text.strip()})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify,send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import os
import time
import base64
import requests


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed


@app.route('/decolorize', methods=['POST'])
def decolorize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_stream = io.BytesIO(file.read())
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image file'}), 400

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Encode to base64
    _, buffer = cv2.imencode('.jpg', gray_image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'image': image_base64})




# Paths to load the model
DIR = r"C:\Users\Sharmada\Desktop\final ml-pro\ml-pro\backend"
PROTOTXT = os.path.join(DIR, r"models\colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"models\pts_in_hull.npy")
MODEL = os.path.join(DIR, r"models\colorization_release_v2.caffemodel")

# Load the Model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Reshape cluster centers
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

@app.route("/colorize", methods=["POST"])
def colorize():
    # Check if file is uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400
    
    file = request.files["file"]
    
    # Check if file has a valid name
    if file.filename == "":
        return jsonify({"error": "No selected file!"}), 400

    # Read the image from the uploaded file
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image file!"}), 400

    # Preprocess the image
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict 'ab' channels using the neural network
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Combine with original L channel
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Encode the colorized image to Base64
    _, buffer = cv2.imencode('.jpg', colorized)  # Encode the image as JPEG
    base64_image = base64.b64encode(buffer).decode('utf-8')  # Convert to Base64 string

    # Return the Base64 image in the JSON response
    return jsonify({
        "message": "Image colorized successfully!",
        "image": base64_image
    }), 200


# Hugging Face API Configuration
API_KEY = "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"
HF_API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

@app.route('/generate-image', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get("prompt")
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    response = requests.post(HF_API_URL, json={"inputs": prompt}, headers=HEADERS)
    
    if response.status_code == 200:
        image_bytes = response.content
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        return jsonify({"generated_image_base64": f"data:image/png;base64,{encoded_image}"})
    else:
        return jsonify({"error": response.text}), response.status_code



# Hugging Face API Key (Replace with your own key)
api_key = "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"



# Hugging Face API Key (Replace with your own key)
api_key = "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"

# Define API URL
url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
headers = {"Authorization": f"Bearer {api_key}"}

@app.route('/image-to-text', methods=['POST'])
def image_to_text():
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({"error": "No image uploaded"}), 400

    # Convert image to bytes
    image = Image.open(image_file).convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()

    # Send request to Hugging Face API
    response = requests.post(url, headers=headers, data=img_bytes)

    if response.status_code == 200:
        caption = response.json()[0]["generated_text"]
        return jsonify({"caption": caption})
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == '__main__':
    app.run(debug=True)





 





