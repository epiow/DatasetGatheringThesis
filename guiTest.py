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
from keras.layers import Layer, Multiply, Concatenate, GlobalAveragePooling3D, Reshape
from keras.initializers import Orthogonal, HeNormal
import difflib

# --- Define the vocabulary and string lookup layers ---
# (Keep in mind that your training must match these tokens!)
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)

# --- Define the fixed set of phrases ---
phrases = [
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

def get_closest_match(prediction):
    """
    Given a raw prediction string, convert it to lowercase,
    and match it against our predefined phrases (ignoring case).
    Returns the best matching phrase or "Unknown" if none is close enough.
    """
    # Convert the predicted text to lowercase for matching.
    prediction_lower = prediction.lower()
    # Create a mapping from lowercase phrases to the original phrases.
    phrase_map = {p.lower(): p for p in phrases}
    # Use difflib to find the closest match.
    matches = difflib.get_close_matches(prediction_lower, list(phrase_map.keys()), n=1, cutoff=0.4)
    if matches:
        return phrase_map[matches[0]]
    else:
        return "Unknown"

# --- Define a custom layer (Selective Feature Fusion Module) ---
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

# --- Build the lip reading model ---
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
    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(128, kernel_initializer=Orthogonal(), return_sequences=True)))
    model.add(Dropout(0.5))

    # Output layer: note that the output dimension is based on the vocabulary size.
    model.add(Dense(char_to_num.vocabulary_size() + 1, kernel_initializer='he_normal', activation='softmax'))
        
    return model

# --- Define the GUI application for lip reading ---
class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson Lip Reading System")
        self.root.geometry("1024x600")
        
        # Load the model
        try:
            # Create model with architecture
            self.model = create_model()
            # Load weights (make sure the weights file exists and matches the model)
            self.model.load_weights('./checkpoint.weights.h5')
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            
        # Initialize preprocessing parameters
        self.target_height = 46  # Match model input height
        self.target_width = 140  # Match model input width
        self.sequence_length = 75
        
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
        """Preprocess a single frame for the model."""
        if self.predictor is None:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if len(faces) > 0:
            face = faces[0]
            landmarks = self.predictor(gray, face)
            lip_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) 
                                   for n in range(48, 68)])
            
            x, y, w, h = cv2.boundingRect(lip_points)
            lip_frame = frame[y:y + h, x:x + w]
            lip_frame = cv2.resize(lip_frame, (self.target_width, self.target_height))
            
            # Convert to grayscale
            lip_frame = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY)
            
            # Normalize to range [-1, 1]
            lip_frame = (lip_frame - 127.5) / 127.5
            
            # Add channel dimension
            lip_frame = np.expand_dims(lip_frame, axis=-1)
            
            return lip_frame
        return None

    def process_frames(self):
        if len(self.frame_buffer) < self.sequence_length:
            print(f"Not enough frames: {len(self.frame_buffer)}")
            return
        
        # Preprocess the frames
        processed_frames = []
        for frame in self.frame_buffer[:self.sequence_length]:
            processed = self.preprocess_frame(frame)
            if processed is not None:
                processed_frames.append(processed)
        
        if len(processed_frames) < self.sequence_length:
            print("Some frames could not be processed")
            return
        
        try:
            # Prepare input for model (batch size 1)
            X = np.array(processed_frames)
            X = np.expand_dims(X, axis=0)
            
            # Model inference
            yhat = self.model.predict(X)
            
            # Use CTC decoder to convert the prediction into a sequence of character indices
            decoded = tf.keras.backend.ctc_decode(
                yhat, input_length=[self.sequence_length], greedy=True
            )[0][0].numpy()
            
            # Convert indices to characters using num_to_char lookup
            prediction_chars = []
            for num in decoded[0]:
                if num >= 0:  # Skip padding tokens (if any)
                    char = num_to_char(num).numpy().decode('utf-8')
                    prediction_chars.append(char)
            
            # Form the raw predicted string
            raw_prediction = ''.join(prediction_chars)
            print(f"Raw prediction: {raw_prediction}")
            
            # Match the raw prediction to the closest phrase in our list
            self.current_prediction = get_closest_match(raw_prediction)
            print(f"Matched phrase: {self.current_prediction}")
            
        except Exception as e:
            print(f"Error during inference: {e}")
            self.current_prediction = f"Error: {str(e)}"
        
        self.root.after(0, self.finish_processing)

    def toggle_inference(self):
        if not self.inference_running and self.model is not None:
            self.inference_running = True
            self.start_button.config(text="Collecting Frames...", state=tk.DISABLED)
            self.frame_buffer = []
            threading.Thread(target=self.collect_frames).start()
        else:
            self.pred_label.config(text="Error: Model not loaded")

    def collect_frames(self):
        for _ in range(self.sequence_length):
            if not self.inference_running:
                break
            ret, frame = self.cap.read()
            if ret:
                self.frame_buffer.append(frame)
            time.sleep(0.04)  # ~25 fps
        
        if self.frame_buffer:
            self.root.after(0, self.start_processing)

    def start_processing(self):
        self.updating_camera = False
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.start_button.config(text="Processing...")
        
        threading.Thread(target=self.process_frames).start()

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
