import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

def classify_image(image_path):
    # Load the pre-trained model
    model = load_model("modele/model.keras") 
    classes = ['Bombarder','Divers', 'Fighter', 'Recognition', 'Transport']

    
    img = image.load_img(image_path, color_mode="rgb", target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

   
    predictions = model.predict(img_array)
    
   
    c_index = np.argmax(predictions)
    c = classes[c_index]
    
    return c