import cv2
import numpy as np
import dlib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Activation, TimeDistributed
from keras.layers import Flatten, Bidirectional, LSTM, Dropout, Dense
from keras.layers import Layer, Dense, Multiply, Concatenate, GlobalAveragePooling3D, Reshape
from ctc import CTC
import time
import sys

# Keep the vocabulary and character conversion setup
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)
lexicon = [
        "Maayong buntag",
        "Maayong hapon",
        "Maayong Gabii",
        "Amping",
        "Maayo Man Ko",
        "Palihug",
        "Mag-amping ka",
        "Walay Sapayan",
        "Unsa imong buhaton?",
        "Daghang Salamat",
        "Naimbag a bigat",
        "Naimbag a malem",
        "Naimbag a rabii",
        "Diyos iti agyaman",
        "Mayat Met, agyamanak",
        "Paki",
        "Ag im-imbag ka",
        "Awan ti ania",
        "Anat ub-ubraem",
        "Agyamanak un-unay"
    ]

class SelectiveFeatureFusionModule(Layer):
    def __init__(self):
        super(SelectiveFeatureFusionModule, self).__init__()

    def build(self, input_shape):
        self.w1 = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True)
        self.w2 = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True)
        self.projection = Dense(input_shape[-1], kernel_initializer='he_normal')

    def call(self, inputs):
        weighted_input1 = inputs * self.w1
        weighted_input2 = inputs * self.w2
        fused_features = Concatenate()([weighted_input1, weighted_input2])
        fused_features = self.projection(fused_features)
        return fused_features

def create_model():
    model = Sequential()
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(SelectiveFeatureFusionModule())
    model.add(Reshape((75, 5 * 17 * 75)))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
    
    return model

class LipReadingSystem:
    def __init__(self):
        self.target_height = 46
        self.target_width = 140
        self.sequence_length = 75
        self.frame_rate = 25
        self.recording_resolution = (640, 480)
        
        print("Initializing lip reading system...")
        
        # Set memory growth for GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled")
            except RuntimeError as e:
                print(f"Error setting GPU memory growth: {e}")
        
        # Load model with reduced memory footprint
        try:
            tf.keras.backend.clear_session()
            self.model = create_model()
            self.model.load_weights('checkpointlatest.weights.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Initialize face detection
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor("./LipNetTesting/shape_predictor_68_face_landmarks.dat")
            print("Face predictor loaded successfully")
        except Exception as e:
            print(f"Error loading face predictor: {e}")
            sys.exit(1)

    def collect_frames(self, source):
        """Collect frames from video source"""
        print("Collecting frames...")
        frame_buffer = []
        
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.recording_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.recording_resolution[1])
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return None
            
        start_time = time.time()
        
        while len(frame_buffer) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_buffer.append(frame)
            
            # Maintain precise frame rate
            current_count = len(frame_buffer)
            target_elapsed = current_count / self.frame_rate
            actual_elapsed = time.time() - start_time
            sleep_time = target_elapsed - actual_elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cap.release()
        
        if len(frame_buffer) < self.sequence_length:
            print(f"Warning: Only collected {len(frame_buffer)} frames")
            return None
            
        return frame_buffer
    def preprocess_frame(self, frame):
        """Preprocess a single frame for the model with improved ROI extraction and normalization"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = self.predictor(gray, face)
            mouth_points = [(landmarks.part(n).x, landmarks.part(n).y) 
                        for n in range(48, 68)]
            
            # Calculate bounding box with padding
            x_coords = [p[0] for p in mouth_points]
            y_coords = [p[1] for p in mouth_points]
            padding = 5
            
            x_min = max(0, min(x_coords) - padding)
            y_min = max(0, min(y_coords) - padding)
            x_max = min(gray.shape[1], max(x_coords) + padding)
            y_max = min(gray.shape[0], max(y_coords) + padding)
            
            # Extract and resize mouth ROI
            mouth_roi = gray[y_min:y_max, x_min:x_max]
            lip_frame = cv2.resize(mouth_roi, (self.target_width, self.target_height))
            
            # Convert to tensor for consistent processing
            lip_frame = tf.convert_to_tensor(lip_frame)
            lip_frame = tf.expand_dims(lip_frame, axis=-1)
            
            # Apply same normalization as training
            mean = tf.reduce_mean(lip_frame)
            std = tf.math.reduce_std(tf.cast(lip_frame, tf.float32))
            lip_frame = tf.cast((lip_frame - mean), tf.float32) / std
            
            return lip_frame.numpy()
        
        # Fallback to static crop if no face detected
        static_crop = gray[190:236, 80:220]
        static_crop = cv2.resize(static_crop, (self.target_width, self.target_height))
        static_crop = tf.convert_to_tensor(static_crop)
        static_crop = tf.expand_dims(static_crop, axis=-1)
        
        mean = tf.reduce_mean(static_crop)
        std = tf.math.reduce_std(tf.cast(static_crop, tf.float32))
        return tf.cast((static_crop - mean), tf.float32) / std.numpy()


    def process_frames(self, frame):
        """Process a list of frames and return prediction"""
        if len(frame) < self.sequence_length:
            return f"Error: Not enough frames. Need {self.sequence_length}, got {len(frame)}"
            
        frame_tensors = []
        
        # Process only the required number of frames
        for frame in frame[:self.sequence_length]:
            processed = self.preprocess_frame(frame)
            if processed is not None:
                frame_tensors.append(processed)
        
        if len(frame_tensors) < self.sequence_length:
            return "Error: Failed to process some frames"
        
        try:
            # Stack frames into a single tensor
            X = np.stack(frame_tensors)
            X = np.expand_dims(X, axis=0)
            
            yhat = self.model.predict(X)
            decoded = tf.keras.backend.ctc_decode(yhat, 
                                                input_length=[75], 
                                                greedy=True)[0][0].numpy()
            
            prediction_chars = []
            for num in decoded[0]:
                if num >= 0:
                    char = num_to_char(num).numpy().decode('utf-8')
                    prediction_chars.append(char)
            print("Actual Output: " + (''.join(prediction_chars)))
            
            return CTC.correct_to_lexicon((''.join(prediction_chars)), lexicon)
            
        except Exception as e:
            return f"Error during inference: {str(e)}"

def main():
    # Initialize the system
    print("Lip Reading System - Console Version")
    system = LipReadingSystem()
    
    while True:
        print("\nOptions:")
        print("1. Process from webcam")
        print("2. Process from video file")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == "1":
            print("Recording from webcam for 3 seconds...")
            prediction = system.process_video(0)  # 0 for default webcam
            print(f"Prediction: {prediction}")
            
        elif choice == "2":
            video_path = input("Enter the path to the video file: ")
            print(f"Processing video: {video_path}")
            prediction = system.process_video(video_path)
            print(f"Prediction: {prediction}")
            
        elif choice == "3":
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()