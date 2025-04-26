import os
import warnings

# Making sure only error logs are shown
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import logging
import numpy as np
from PIL import Image

# Show only error logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

import tensorflow as tf

tf.get_logger().setLevel('ERROR')
# Turn off XLA JIT
tf.config.optimizer.set_jit(False)

try:
    import absl.logging as _absl_logging
    _absl_logging.set_verbosity(_absl_logging.ERROR)
except ImportError:
    pass

# Load the trained Keras model (ensure rice_model.keras is in the same directory)
model = tf.keras.models.load_model('rice_model.keras')

class_names = ['Ipsala', 'Jasmine', 'Arborio', 'Karacadag', 'Basmati']

def preprocess_image(image_path, target_size=(50, 50)):
    """
    Load an image file, convert to grayscale, resize, normalize, and expand dims
    """
    img = Image.open(image_path).convert('L')  # Converting to greyscale
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = arr.reshape((target_size[0], target_size[1], 1))
    return np.expand_dims(arr, axis=0)


def predict(image_path):
    """
    Predict the class of a rice grain image.
    Returns the class name as a string.
    """
    input_arr = preprocess_image(image_path)
    preds = model.predict(input_arr)
    idx = int(np.argmax(preds, axis=1)[0])
    return class_names[idx]