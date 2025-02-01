import argparse
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json
from PIL import Image
import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

batch_size = 32
image_size = 224

def load_model(saved_keras_model):
    """Load the saved Keras model."""
    loaded_keras_model = tf.keras.models.load_model(saved_keras_model)
    return loaded_keras_model

def load_class_names(flower_classes):
    """Load the class names from a JSON file."""
    with open(flower_classes, 'r') as f:
        class_names = json.load(f)
    return class_names

def process_image(image):
    """Process the image for prediction."""
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255.0  
    return image.numpy()

def predict(image_path, model, top_k):
    """Predict the top K classes for an image."""
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    
    prediction = model.predict(np.expand_dims(processed_test_image, axis=0))
    
    top_k_prediction_values, top_k_prediction_indices = tf.math.top_k(prediction, top_k)
    
    top_k_prediction_values = top_k_prediction_values.numpy()
    top_k_prediction_indices = top_k_prediction_indices.numpy()
    
    return top_k_prediction_values[0], top_k_prediction_indices[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a flower class')
    
    parser.add_argument('-i', '--input_image', default='./test_images/hard-leaved_pocket_orchid.jpg', action='store', help='Image path', type=str)
    parser.add_argument('-m', '--model', default='./flower_classifier_model.h5', action='store', help='Path to saved Keras model', type=str)
    parser.add_argument('-k', '--top_k', default=5, action='store', help='Top K flower classes', type=int)
    parser.add_argument('-n', '--category_names', default='./label_map.json', action='store', help='Path to class names JSON file', type=str)
    
    args = parser.parse_args()
    image_path = args.input_image
    saved_keras_model = args.model
    top_k = args.top_k
    flower_classes = args.category_names

    model = load_model(saved_keras_model)
    class_names = load_class_names(flower_classes)
    
    top_k_prediction_values, top_k_prediction_indices = predict(image_path, model, top_k)
    
    top_flower_classes = [class_names[str(index + 1)] for index in top_k_prediction_indices]
    
    print('Top Probabilities:', top_k_prediction_values)
    print('Top Classes:', top_flower_classes)
