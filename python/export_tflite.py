# Convert gesture_model.h5 to a fully int8-quantized TensorFlow Lite model.
# Uses the training data as a representative dataset for calibration.
import numpy as np
import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("gesture_model.h5")

# Load training data to use as the representative dataset for quantization
X_train = np.load("X_train.npy").astype(np.float32)


def representative_dataset():
    for i in range(len(X_train)):
        sample = X_train[i:i + 1]
        yield [sample]


# Set up the converter for full integer (int8) quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Convert the model
tflite_model = converter.convert()

# Save the quantized model to disk
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved model.tflite")