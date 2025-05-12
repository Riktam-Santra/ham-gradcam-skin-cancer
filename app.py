from flask import Flask, render_template, request
import os
import tensorflow as tf
from gradcam import make_gradcam_heatmap, save_and_overlay_gradcam
from main import load_image, predict_with_h5_model, LABEL_MAPPING

UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'skin_cancer_model.h5'
LAST_CONV_LAYER_NAME = 'conv2d_6'  # Replace with your actual last conv layer name

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = tf.keras.models.load_model(MODEL_PATH)

# "Call" the model once
# dummy_input = tf.zeros((1, 28, 28, 3))
# model(dummy_input)  # Important!

model.summary()

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
            image_file.save(image_path)

            image = load_image(image_path)
            label, confidence = predict_with_h5_model(MODEL_PATH, image_path)

            # Generate Grad-CAM
            heatmap = make_gradcam_heatmap(image, model)
            cam_path = save_and_overlay_gradcam(image_path, heatmap)

            return render_template("result.html", prediction=LABEL_MAPPING[label], confidence=confidence, image_path=os.path.basename(image_path), cam_image=os.path.basename(cam_path))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)