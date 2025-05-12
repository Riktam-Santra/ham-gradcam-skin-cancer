import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="last_conv", pred_index=None):
    # Get the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # Create a model that maps input to activations of last conv layer and output
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        # Forward pass
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradient of the class output w.r.t. feature map
    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise RuntimeError("Failed to compute gradients. Check if the model is correctly built and connected.")

    # Pool the gradients over the width and height
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by the 'importance' of the channel
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize to 0–1 for display
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_overlay_gradcam(img_path, heatmap, cam_path=None, alpha=0.4):
    if cam_path is None:
        cam_path = img_path.replace(".jpg", "_cam.jpg")

    img = cv2.imread(img_path)
    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    cv2.imwrite(cam_path, superimposed_img)

    return cam_path