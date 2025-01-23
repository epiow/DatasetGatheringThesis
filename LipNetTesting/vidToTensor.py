import cv2
import tensorflow as tf
import numpy as np

def video_to_tensor(video_path, use_gpu=True):
    """
    Convert an MP4 video file to a TensorFlow tensor using GPU acceleration.
    
    Args:
        video_path (str): Path to the MP4 file
        use_gpu (bool): Whether to use GPU acceleration
        
    Returns:
        tf.Tensor: Tensor of shape (frames, height, width, channels)
    """
    # Check GPU availability
    if use_gpu:
        if not tf.config.list_physical_devices('GPU'):
            print("Warning: No GPU found. Falling back to CPU.")
        else:
            print("GPU detected and enabled.")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    frames = []
    with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert BGR to RGB and to float32
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = tf.cast(frame, tf.float32) / 255.0
            
            frames.append(frame)
    
    cap.release()
    
    # Stack frames and ensure they're on GPU
    with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
        video_tensor = tf.stack(frames)
        
        # Optional: You can force the tensor to be processed on GPU
        if use_gpu:
            video_tensor = tf.identity(video_tensor)
    
    return video_tensor

def save_tensor_video(tensor, output_path, fps=30, use_gpu=True):
    """
    Convert a tensor back to an MP4 video file using GPU acceleration.
    
    Args:
        tensor (tf.Tensor): Input tensor of shape (frames, height, width, channels)
        output_path (str): Path to save the MP4 file
        fps (int): Frames per second for the output video
        use_gpu (bool): Whether to use GPU acceleration
    """
    with tf.device('/GPU:0' if use_gpu else '/CPU:0'):
        # Denormalize and convert to uint8
        video_array = tf.cast(tensor * 255, tf.uint8)
        video_array = video_array.numpy()
    
    # Get video dimensions
    height, width = video_array.shape[1:3]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in video_array:
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()

# Optional: Configure GPU memory growth to avoid memory issues
def configure_gpu():
    """
    Configure GPU settings for optimal performance.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

# First, configure GPU settings
configure_gpu()

# Convert video to tensor using GPU
video_path = "preprocessed_lip_video2.mp4"
tensor = video_to_tensor(video_path, use_gpu=True)

print(f"Tensor shape: {tensor.shape}")
print(f"Tensor device: {tensor.device}")

# Save back to video using GPU
output_path = "output_video2.mp4"
save_tensor_video(tensor, output_path, use_gpu=True)