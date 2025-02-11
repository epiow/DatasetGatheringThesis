import tf2onnx
import tensorflow as tf
import keras

# Load your Keras model
model = keras.models.load_model("weights.h5")

# Convert to ONNX
input_signature = [
    tf.TensorSpec(
        shape=(None, 75, 46, 140, 1),  # Adjust to your model's input shape
        dtype=tf.float32,
        name="input"
    )
]
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13
)

# Save the ONNX model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())