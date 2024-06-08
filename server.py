from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import os
import uuid
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('covid19_xray_model.h5')

# Define class names
class_names = ['Covid', 'Normal']

def predict(img):
    img = img.convert('RGB')  # Convert image to RGB format
    img = img.resize((331, 331))
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    predictions = model.predict(img)
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the uploaded file
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join('static/uploads', filename)
        file.save(filepath)

        img = Image.open(file)
        predictions = predict(img)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100
        return render_template('result.html', prediction=class_names[class_idx], confidence=confidence, image_url=url_for('static', filename='uploads/' + filename))
    else:
        return redirect(request.url)

if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)
