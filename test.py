import tensorflow as tf

# Check if GPU is available
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Create a simple TensorFlow operation
def test_gpu():
    # Create two random matrices
    a = tf.random.normal([1000, 1000])
    b = tf.random.normal([1000, 1000])

    # Perform matrix multiplication
    c = tf.matmul(a, b)
    return c

# Run the operation and print the result
print("Testing GPU...")
print(test_gpu())