import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import resnet50







def predict_image(image_path):
    kal = []
    model = load_model('modele/my_model.keras')
    class_names = ['Bombarder','Divers', 'Fighter', 'Recognition', 'Transport']
    img = image.load_img(image_path, color_mode="grayscale", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    
    predicted_class_index = np.argmax(predictions)
    predicted_probability = predictions[0][predicted_class_index]

    predicted_class = class_names[predicted_class_index]
    kal.append(predicted_class)
    kal.append(predicted_probability)
    
    return predicted_class

def predict_res(image_path):
    class_names= ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C17', 'C2', 'C5', 'E2', 'E7', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'J10', 'J20', 'JAS39', 'KC135', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57', 'Tornado', 'Tu160', 'Tu95', 'U2', 'US2', 'V22', 'Vulcan', 'XB70', 'YF23']
    loaded_model = tf.saved_model.load('modele/my_model_ResNet50')

    img = image.load_img(image_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = resnet50.preprocess_input(img_array)
    
    predictions = loaded_model.signatures['serving_default'](tf.constant(img_array))['output_0']
    
    predictions = predictions.numpy()
    
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class



# import cv2
# import torch
# import numpy as np

# def convert_pytorch_model_to_onnx(model, input_shape, output_path):
#     # Export the PyTorch model to ONNX
#     dummy_input = torch.randn(1, *input_shape)
#     torch.onnx.export(model, dummy_input, output_path, verbose=True)

# def predict_yolov8_classification(image_path, onnx_model_path, class_names):
#     class_names= ['A10', 'A400M', 'AG600', 'AV8B', 'B1', 'B2', 'B52', 'Be200', 'C130', 'C17', 'C2', 'C5', 'E2', 'E7', 'EF2000', 'F117', 'F14', 'F15', 'F16', 'F18', 'F22', 'F35', 'F4', 'J10', 'J20', 'JAS39', 'KC135', 'MQ9', 'Mig31', 'Mirage2000', 'P3', 'RQ4', 'Rafale', 'SR71', 'Su24', 'Su25', 'Su34', 'Su57', 'Tornado', 'Tu160', 'Tu95', 'U2', 'US2', 'V22', 'Vulcan', 'XB70', 'YF23']
#     # Load the image
#     img = cv2.imread(image_path)

#     # Check if image loading was successful
#     if img is None:
#         print("Error: Could not read image!")
#         return None

#     # Load the ONNX model
#     net = cv2.dnn.readNetFromONNX(onnx_model_path)

#     # Prepare the image for prediction
#     blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
#     net.setInput(blob)

#     # Make predictions
#     outputs = net.forward()

#     # Get predictions
#     class_ids = np.argmax(outputs, axis=1)
#     confidences = np.max(outputs, axis=1)

#     # Process predictions
#     results = []
#     for class_id, confidence in zip(class_ids, confidences):
#         class_name = class_names[class_id]
#         results.append((class_name, confidence))

#     return results



# # Convertir le modèle en format ONNX
# input_shape = (3, 416, 416)  # Changer la taille d'entrée en fonction de votre modèle