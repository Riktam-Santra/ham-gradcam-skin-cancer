import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
IMG_SIZE = 224  # Match your model's input size
LABELS = ["akiec", "bcc", "bkl", "df", "nv", "vasc", "mel"]  # HAM10000 labels
LABEL_MAPPING = {
    "akiec": "Actinic keratoses and intraepithelial carcinomae",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "vasc": "Pyogenic granulomas and hemorrhage",
    "mel": "Melanoma"
}

def load_image(image_path):
    """
    Loads and preprocesses an image for prediction.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

def predict_with_h5_model(model_path, image_path):
    """
    Predict using a .h5 model.
    """
    model = tf.keras.models.load_model(model_path)
    image = load_image(image_path)
    predictions = model.predict(image)
    predicted_index = int(np.argmax(predictions))
    label = LABELS[predicted_index]
    confidence = predictions[0][predicted_index]
    return label, confidence

def predict_with_tflite_model(model_path, image_path):
    """
    Predict using a .tflite model.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = load_image(image_path).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = int(np.argmax(output_data))
    label = LABELS[predicted_index]
    confidence = output_data[0][predicted_index]
    return label, confidence

if __name__ == "__main__":
    model_path_h5 = "Skin Cancer.h5"
    model_path_tflite = "Skin.tflite"
    image_path = "sample_false.jpg"

    # Predict using .h5 model
    try:
        label, confidence = predict_with_h5_model(model_path_h5, image_path)
        print(f"Prediction (.h5): {label} ({LABEL_MAPPING[label]})")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error using .h5 model: {e}")

    # Predict using .tflite model
    try:
        label, confidence = predict_with_tflite_model(model_path_tflite, image_path)
        print(f"Prediction (.tflite): {label} ({LABEL_MAPPING[label]})")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"Error using .tflite model: {e}")
