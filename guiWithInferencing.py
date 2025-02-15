# main.py
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import dlib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D, MaxPool3D, Activation, TimeDistributed
from keras.layers import Flatten, Bidirectional, LSTM, Dropout, Dense
from keras.layers import Layer, Dense, Multiply, Concatenate, GlobalAveragePooling3D, Reshape
from keras.initializers import Orthogonal, HeNormal
import json
from ctc import CTC

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
        # Learnable weights for feature fusion
        self.w1 = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True)
        self.w2 = self.add_weight(shape=(input_shape[-1],), initializer='he_normal', trainable=True)
        # Learnable projection to reduce channels after fusion
        self.projection = Dense(input_shape[-1], kernel_initializer='he_normal')

    def call(self, inputs):
        # Apply attention weights to the input features
        weighted_input1 = inputs * self.w1
        weighted_input2 = inputs * self.w2
        # Concatenate the weighted features
        fused_features = Concatenate()([weighted_input1, weighted_input2])
        # Project the fused features back to the original number of channels
        fused_features = self.projection(fused_features)
        return fused_features

def create_model():
    model = Sequential()
    # First Conv3D layer
    # Input shape: (75, 46, 140, 1)
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))
    # Shape after pooling: (75, 5, 17, 75)

    # Add Selective Feature Fusion Module (SFFM)
    model.add(SelectiveFeatureFusionModule())

    # Reshape before LSTM - multiply the last three dimensions: 5 * 17 * 75 = 6375
    model.add(Reshape((75, 5 * 17 * 75)))  # Shape will be (75, 6375)

    # LSTM layers
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    # Output layer
    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
        
    return model

class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson Lip Reading System")
        self.root.geometry("1024x600")
        
        # Load the model
        try:
            # Create model with architecture
            self.model = create_model()
            # Load weights
            self.model.load_weights('./checkpointlatest.weights.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Initialize preprocessing parameters
        self.target_height = 46  # Match model input
        self.target_width = 140  # Match model input
        self.sequence_length = 75
        self.frame_rate = 25  # 25 FPS for 3 seconds = 75 frames
        self.recording_resolution = (640, 480)  # Match dataset recording resolution
        
        # Camera and processing variables
        self.canvas_width = 480
        self.canvas_height = 360
        self.cap = None
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor("./LipNetTesting/shape_predictor_68_face_landmarks.dat")
            print("Face predictor loaded successfully")
        except Exception as e:
            print(f"Error loading face predictor: {e}")
            self.predictor = None
            
        self.inference_running = False
        self.current_prediction = ""
        self.updating_camera = True
        self.frame_buffer = []
        
        self.setup_gui()
        self.start_camera()

    def setup_gui(self):
        # Video display frame
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.pack(padx=10, pady=10, fill=tk.X)
        
        # Camera canvas
        self.canvas = tk.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(anchor="center", pady=10)
        
        # Loading overlay
        self.loading_label = ttk.Label(
            self.video_frame,
            text="Processing...",
            font=("Helvetica", 24),
            background="white",
            relief="solid"
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.loading_label.place_forget()
        
        # Control frame
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(pady=5)
        
        # Start button
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start Inference", 
            command=self.toggle_inference
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Status indicators
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(pady=5)
        
        self.model_status = ttk.Label(
            self.status_frame,
            text="Model Status: " + ("Loaded" if self.model else "Not Loaded"),
            font=("Helvetica", 10)
        )
        self.model_status.pack(pady=2)
        
        # Prediction display
        self.pred_label = ttk.Label(
            self.root, 
            text="Prediction: ", 
            font=("Helvetica", 14)
        )
        self.pred_label.pack(pady=5)

    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print("Camera initialized successfully")
            self.show_camera_feed()
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.pred_label.config(text="Error: Camera not available")

    def show_camera_feed(self):
        if not self.updating_camera:
            return
            
        ret, frame = self.cap.read()
        if ret:
            frame = self.detect_mouth(frame)
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        self.root.after(10, self.show_camera_feed)

    def detect_mouth(self, frame):
        if self.predictor is None:
            return frame
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            mouth_points = landmarks.parts()[48:68]
            
            x = min([p.x for p in mouth_points])
            y = min([p.y for p in mouth_points])
            w = max([p.x for p in mouth_points]) - x
            h = max([p.y for p in mouth_points]) - y
            
            margin_x = int(w * 0.5)
            margin_y = int(h * 0.5)
            
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(frame.shape[1] - x, w + 2 * margin_x)
            h = min(frame.shape[0] - y, h + 2 * margin_y)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame

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


    def process_frames(self):
        if len(self.frame_buffer) < self.sequence_length:
            print(f"Not enough frames: {len(self.frame_buffer)}")
            return
            
        frame_tensors = []
        processed_frames = []
        
        # Process only the required number of frames
        for frame in self.frame_buffer[:self.sequence_length]:
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
            print("Predicted Output: " + (CTC.correct_to_lexicon((''.join(prediction_chars)), lexicon)))
            self.current_prediction = CTC.correct_to_lexicon((''.join(prediction_chars)), lexicon)
            
        except Exception as e:
            return f"Error during inference: {str(e)}"


    def toggle_inference(self):
        if not self.inference_running and self.model is not None:
            self.inference_running = True
            self.start_button.config(text="Collecting Frames...", state=tk.DISABLED)
            self.frame_buffer = []
            threading.Thread(target=self.collect_frames).start()
        else:
            self.pred_label.config(text="Error: Model not loaded")

    def collect_frames(self):
        """Collect frames with precise timing to match dataset recording"""
        start_time = time.time()
        self.frame_buffer = []
        
        while len(self.frame_buffer) < self.sequence_length and self.inference_running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
                
                # Maintain precise frame rate timing
                current_count = len(self.frame_buffer)
                target_elapsed = current_count / self.frame_rate
                actual_elapsed = time.time() - start_time
                sleep_time = target_elapsed - actual_elapsed
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        if len(self.frame_buffer) >= self.sequence_length:
            self.root.after(0, self.start_processing)
        else:
            self.root.after(0, self.cancel_processing)

    def start_processing(self):
        """Final checks before processing"""
        if len(self.frame_buffer) == self.sequence_length:
            self.updating_camera = False
            self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            self.start_button.config(text="Processing...")
            threading.Thread(target=self.process_frames).start()
        else:
            self.cancel_processing()

    def cancel_processing(self):
        """Handle incomplete frame collection"""
        self.loading_label.place_forget()
        self.updating_camera = True
        self.show_camera_feed()
        self.inference_running = False
        self.start_button.config(text="Start Inference", state=tk.NORMAL)
        self.pred_label.config(text="Prediction: Collection interrupted")

    def toggle_inference(self):
        if not self.inference_running and self.model is not None:
            self.inference_running = True
            self.start_button.config(text="Preparing...", state=tk.DISABLED)
            # Reset frame buffer and start timed collection
            self.frame_buffer = []
            threading.Thread(target=self.collect_frames).start()
        else:
            self.pred_label.config(text="Error: Model not loaded")

    def finish_processing(self):
        self.loading_label.place_forget()
        self.updating_camera = True
        self.show_camera_feed()
        
        self.pred_label.config(text=f"Prediction: {self.current_prediction}")
        self.inference_running = False
        self.start_button.config(text="Start Inference", state=tk.NORMAL)

    def on_closing(self):
        self.inference_running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LipReadingApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()