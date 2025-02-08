import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import dlib

class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jetson Lip Reading System")
        # Set the main window size to 1024x600 to fit the display
        self.root.geometry("1024x600")
        
        # Define canvas dimensions that maintain a 4:3 aspect ratio.
        # (For example, 667x500 fits well within 1024x600, leaving room for controls.)
        self.canvas_width = 480
        self.canvas_height = 360
        
        # Camera and model variables
        self.cap = None
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("./LipNetTesting/shape_predictor_68_face_landmarks.dat")
        self.inference_running = False
        self.current_prediction = ""
        self.updating_camera = True
        self.frame_buffer = []  # To store collected frames
        
        # Setup the GUI elements
        self.setup_gui()
        
        # Start the camera feed
        self.start_camera()

    def setup_gui(self):
        # Create a labeled frame for the video display
        self.video_frame = ttk.LabelFrame(self.root, text="Camera Feed")
        self.video_frame.pack(padx=10, pady=10, fill=tk.X)
        
        # Create a canvas for the camera feed using our defined dimensions.
        # Using pack(anchor="center") centers the canvas horizontally.
        self.canvas = tk.Canvas(self.video_frame, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(anchor="center", pady=10)
        
        # Loading overlay shown during processing
        self.loading_label = ttk.Label(
            self.video_frame,
            text="Processing...",
            font=("Helvetica", 24),
            background="white",
            relief="solid"
        )
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.loading_label.place_forget()
        
        # Control buttons frame
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(pady=5)
        
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start Inference", 
            command=self.toggle_inference
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Prediction label to show the model output
        self.pred_label = ttk.Label(
            self.root, 
            text="Prediction: ", 
            font=("Helvetica", 14)
        )
        self.pred_label.pack(pady=5)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        # Optionally, set the camera resolution if supported:
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.show_camera_feed()

    def show_camera_feed(self):
        if not self.updating_camera:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # Process the frame (e.g. to detect and mark the mouth region)
            frame = self.detect_mouth(frame)
            # Resize the frame to our canvas size before displaying it.
            frame = cv2.resize(frame, (self.canvas_width, self.canvas_height))
            # Convert BGR (OpenCV format) to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the canvas with the new frame
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        
        self.root.after(10, self.show_camera_feed)

    def detect_mouth(self, frame):
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

    def toggle_inference(self):
        if not self.inference_running:
            self.inference_running = True
            self.start_button.config(text="Collecting Frames...", state=tk.DISABLED)
            self.frame_buffer = []  # Clear any previous frames
            threading.Thread(target=self.collect_frames).start()

    def collect_frames(self):
        # Collect approximately 75 frames (roughly 3 seconds at ~25 fps)
        for _ in range(75):
            if not self.inference_running:
                break
            ret, frame = self.cap.read()
            if ret:
                # Process the frame to detect the mouth region and store it
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray)
                if len(faces) > 0:
                    face = faces[0]
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
                    
                    mouth_roi = frame[y:y+h, x:x+w]
                    mouth_roi = cv2.resize(mouth_roi, (224, 224))
                    self.frame_buffer.append(mouth_roi)
            
            # Maintain an approximate capture rate of 25 fps
            time.sleep(0.04)
        
        # After collecting frames, start the processing routine.
        if self.frame_buffer:
            self.root.after(0, self.start_processing)

    def start_processing(self):
        # Pause the camera feed and display the loading overlay.
        self.updating_camera = False
        self.loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.start_button.config(text="Processing...")
        
        # Simulate processing (replace with actual inference code)
        threading.Thread(target=self.process_frames).start()

    def process_frames(self):
        # Replace this simulation with your actual model inference.
        # The frame_buffer list holds the 75 collected frames.
        time.sleep(2)  # Simulate a processing delay.
        self.current_prediction = "Maayong buntag (Cebuano)"  # Replace with real output.
        
        # Update the UI after processing is complete.
        self.root.after(0, self.finish_processing)

    def finish_processing(self):
        # Hide the loading overlay and resume the camera feed.
        self.loading_label.place_forget()
        self.updating_camera = True
        self.show_camera_feed()
        
        # Update the prediction display.
        self.pred_label.config(text=f"Prediction: {self.current_prediction}")
        
        # Reset the inference state and re-enable the button.
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
