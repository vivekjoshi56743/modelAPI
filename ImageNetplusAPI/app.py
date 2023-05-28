from flask import Flask, jsonify, request
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import PIL.Image as Image
from flask_cors import CORS
import io

model = tf.keras.models.load_model(
       ('model4.h5'),
       custom_objects={'KerasLayer':hub.KerasLayer}
)

app = Flask(__name__)
cors = CORS(app)

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return img


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    image_data = request.files['image'].read()
    # Preprocess the image data
    processed_image = preprocess_image(image_data)
    # Make a prediction
    prediction = model.predict(processed_image[np.newaxis, ...])
    # Convert the prediction to a JSON object
    classes = []
    with open('final_classes.txt') as f:
        classes = f.read().splitlines()

    res = classes[np.argmax(prediction)] 
    response = jsonify(res)
    return response


if __name__ == '__main__':
    app.run()
