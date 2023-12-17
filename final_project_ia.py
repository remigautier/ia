from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
import numpy as np

# Load the pre-trained InceptionV3 model on ImageNet
inception_model = InceptionV3(weights='imagenet')

# Function to preprocess an image for use with InceptionV3
def preprocess_image_for_inception(img_path):
    img = keras_image.load_img(img_path, target_size=(299, 299))  # InceptionV3 supports 299x299 images
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Path of the image to use
image_path = 'helico.jpg'

# Preprocess the image
processed_image = preprocess_image_for_inception(image_path)

# Make predictions using the InceptionV3 model
predictions = inception_model.predict(processed_image)

# Decode the predictions and print the top three predictions
decoded_predictions = decode_predictions(predictions, top=3)[0]
print("Predictions:")
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")

