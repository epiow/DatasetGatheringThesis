import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import dlib  # For facial landmark detection

class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson Lip Reading System")
        
        # Camera and model variables
        self.cap = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./LipNetTesting/shape_predictor_68_face_landmarks.dat")
        self.inference_running = False
        self.current_prediction = ""
        
        # GUI Setup
        self.setup_gui()
        
        # Start camera thread
        self.start_camera()

    def setup_gui(self):
        # Video display
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.pack(padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.video_frame, width=640, height=480)
        self.canvas.pack()
        
        # Control buttons
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(pady=10)
        
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start Inference", 
            command=self.toggle_inference
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Prediction display
        self.pred_label = ttk.Label(
            self.root, 
            text="Prediction: ", 
            font=("Helvetica", 14)
        )
        self.pred_label.pack(pady=10)

    def start_camera(self):
        # Initialize CSI or USB camera (use CSI for Jetson Nano)
        self.cap = cv2.VideoCapture(0)  # For USB camera
        # For CSI camera: 
        # self.cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM) ! ... ")
        
        self.show_camera_feed()

    def show_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Detect mouth and draw bounding box
            frame = self.detect_mouth(frame)
            
            # Convert to Tkinter-compatible format
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        self.root.after(10, self.show_camera_feed)  # Update every 10ms

    def detect_mouth(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            mouth_points = landmarks.parts()[48:68]  # Mouth landmarks (indices 48-67)
            
            # Get bounding box coordinates for the mouth
            x = min([p.x for p in mouth_points])
            y = min([p.y for p in mouth_points])
            w = max([p.x for p in mouth_points]) - x
            h = max([p.y for p in mouth_points]) - y
            
            # Add margin to extend the bounding box
            margin_x = int(w * 0.5)  # Extend horizontally by 50% of mouth width
            margin_y = int(h * 0.5)  # Extend vertically by 50% of mouth height
            
            x = max(0, x - margin_x)  # Ensure the bounding box stays within the frame
            y = max(0, y - margin_y)
            w = min(frame.shape[1] - x, w + 2 * margin_x)  # Adjust width and height
            h = min(frame.shape[0] - y, h + 2 * margin_y)
            
            # Draw the extended bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return frame

    def toggle_inference(self):
        if not self.inference_running:
            self.inference_running = True
            self.start_button.config(text="Stop Inference")
            threading.Thread(target=self.run_inference).start()
        else:
            self.inference_running = False
            self.start_button.config(text="Start Inference")

    def run_inference(self):
        # Replace this with your actual model inference code
        while self.inference_running:
            ret, frame = self.cap.read()
            if ret:
                # Preprocess frame (crop mouth region, resize, normalize)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                if len(faces) > 0:
                    face = faces[0]
                    landmarks = self.predictor(gray, face)
                    mouth_points = landmarks.parts()[48:68]
                    
                    # Extract mouth ROI
                    x = min([p.x for p in mouth_points])
                    y = min([p.y for p in mouth_points])
                    w = max([p.x for p in mouth_points]) - x
                    h = max([p.y for p in mouth_points]) - y
                    mouth_roi = frame[y:y+h, x:x+w]
                    
                    # Resize to model input shape (e.g., 224x224)
                    mouth_roi = cv2.resize(mouth_roi, (224, 224))
                    
                    # Run model inference (placeholder)
                    self.current_prediction = "Maayong buntag (Cebuano)"  # Replace with model output
                    
                    # Update GUI
                    self.pred_label.config(text=f"Prediction: {self.current_prediction}")
            
            time.sleep(0.1)  # Adjust inference speed

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