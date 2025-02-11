import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import librosa
import sounddevice as sd
import soundfile as sf
import queue
import wave
import os

"""
ca1 = Maayong buntag
cb1 = Maayong hapon
cc1 = Maayong Gabii
cd1 = Amping
ce1 = Maayo Man Ko
cf1 = Palihug
cg1 = Mag-amping ka
ch1 = Walay Sapayan
ci1 = Unsa imong buhaton?
cj1 = Daghang Salamat

ia1 = Naimbag a bigat
ib1 = Naimbag a malem
ic1 = Naimbag a rabii
id1 = Diyos iti agyaman
ie1 = Mayat Met, agyamanak
if1 = Paki
ig1 = Ag im-imbag ka
ih1 = Awan ti ania
ii1 = Anat ub-ubraem
ij1 = Agyamanak un-unay
"""
class VideoAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("LipNet Video Annotator (3-Second Limit)")
        
        # Create necessary folders
        self.mp4_folder = "mp4"
        self.alignment_folder = "alignment"
        self.audio_folder = "audio"  # For WAV files
        
        os.makedirs(self.mp4_folder, exist_ok=True)
        os.makedirs(self.alignment_folder, exist_ok=True)
        os.makedirs(self.audio_folder, exist_ok=True)
        
        # Video recording variables
        self.recording = False
        self.video_filename = None
        self.cap = None
        self.audio_queue = queue.Queue()
        self.audio_data = []
        
        # Recording settings
        self.frame_rate = 25.0
        self.max_frames = 75  # 3 seconds at 25fps
        self.frame_count = 0
        self.resolution = (640, 480)
        self.audio_sample_rate = 44100
        
        # Annotation variables
        self.current_frame = 0
        self.total_frames = 0
        self.annotations = []
        self.target_max_frame = 74500
        
        self.setup_gui()
        
    def get_video_path(self, filename):
        return os.path.join(self.mp4_folder, filename)
        
    def get_audio_path(self, filename):
        return os.path.join(self.audio_folder, filename)
        
    def get_alignment_path(self, filename):
        return os.path.join(self.alignment_folder, filename)
        
    def setup_gui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Video preview
        self.preview_label = ttk.Label(self.main_frame)
        self.preview_label.grid(row=0, column=0, columnspan=2)
        
        # Recording controls
        self.filename_entry = ttk.Entry(self.main_frame, width=30)
        self.filename_entry.grid(row=1, column=0, pady=5)
        self.filename_entry.insert(0, "recording_1")
        
        self.record_button = ttk.Button(self.main_frame, text="Start Recording", 
                                      command=self.toggle_recording)
        self.record_button.grid(row=1, column=1, pady=5)
        
        # Recording timer
        self.timer_label = ttk.Label(self.main_frame, text="Time: 0.0s / 3.0s")
        self.timer_label.grid(row=1, column=2, pady=5)
        
        # Waveform display
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, pady=5)
        
        # Timeline slider
        self.timeline_var = tk.DoubleVar()
        self.timeline_slider = ttk.Scale(self.main_frame, from_=0, to=self.max_frames,
                                       orient=tk.HORIZONTAL, variable=self.timeline_var,
                                       command=self.update_frame)
        self.timeline_slider.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Frame counter
        self.frame_label = ttk.Label(self.main_frame, text=f"Frame: 0/{self.max_frames}")
        self.frame_label.grid(row=3, column=2, padx=5)
        
        # Annotation controls
        annotation_frame = ttk.LabelFrame(self.main_frame, text="Annotation", padding="5")
        annotation_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E))
        
        self.word_entry = ttk.Entry(annotation_frame, width=20)
        self.word_entry.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(annotation_frame, text="Set Start", 
                                     command=self.set_start_time)
        self.start_button.grid(row=0, column=1, padx=5)
        
        self.end_button = ttk.Button(annotation_frame, text="Set End", 
                                   command=self.set_end_time)
        self.end_button.grid(row=0, column=2, padx=5)
        
        # Annotation list
        self.tree = ttk.Treeview(self.main_frame, columns=('Start', 'End', 'Word'),
                                show='headings', height=5)
        self.tree.heading('Start', text='Start Frame')
        self.tree.heading('End', text='End Frame')
        self.tree.heading('Word', text='Word')
        self.tree.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Save button
        self.save_button = ttk.Button(self.main_frame, text="Save Annotations",
                                    command=self.save_annotations)
        self.save_button.grid(row=6, column=0, columnspan=2, pady=5)
        
    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        if self.recording:
            self.audio_queue.put(indata.copy())
            self.audio_data.extend(indata.flatten())
        
    def toggle_recording(self):
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
            
    def clear_annotations(self):
        """Clear all annotations and reset the treeview"""
        self.annotations = []
        for item in self.tree.get_children():
            self.tree.delete(item)
            
    def start_recording(self):
        # Clear existing annotations before starting new recording
        self.clear_annotations()
        
        base_filename = self.filename_entry.get()
        self.video_filename = self.get_video_path(f"{base_filename}.mp4")
        self.audio_filename = self.get_audio_path(f"{base_filename}.wav")
        self.recording = True
        self.frame_count = 0
        self.record_button.config(text="Stop Recording")
        self.audio_data = []
        
        # Start recording threads
        self.record_thread = threading.Thread(target=self.record_video)
        self.record_thread.start()
        
        # Start audio recording
        self.audio_stream = sd.InputStream(callback=self.audio_callback,
                                         channels=1,
                                         samplerate=self.audio_sample_rate)
        self.audio_stream.start()
        
        # Start timer update
        self.update_timer()
    def update_timer(self):
        if self.recording:
            time_elapsed = self.frame_count / self.frame_rate
            self.timer_label.config(text=f"Time: {time_elapsed:.1f}s / 3.0s")
            self.root.after(100, self.update_timer)
        
    def record_video(self):
        try:
            cap = cv2.VideoCapture(2)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(self.video_filename, fourcc, self.frame_rate,
                                self.resolution)
            
            start_time = time.time()
            while self.recording and self.frame_count < self.max_frames:
                ret, frame = cap.read()
                if ret:
                    out.write(frame)
                    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)
                    imgtk = ImageTk.PhotoImage(image=img)
                    self.preview_label.imgtk = imgtk
                    self.preview_label.configure(image=imgtk)
                    self.frame_count += 1
                    
                    # Ensure we maintain proper frame rate
                    elapsed = time.time() - start_time
                    target_elapsed = self.frame_count / self.frame_rate
                    if elapsed < target_elapsed:
                        time.sleep(target_elapsed - elapsed)
                
                if self.frame_count >= self.max_frames:
                    self.stop_recording()
                    
            cap.release()
            out.release()
            
            # Save audio
            if self.audio_data:
                # Trim audio to match video duration
                audio_samples = int(self.audio_sample_rate * 3)  # 3 seconds
                audio_data_trimmed = np.array(self.audio_data[:audio_samples])
                sf.write(self.audio_filename, audio_data_trimmed, self.audio_sample_rate)
            
            self.load_video_for_annotation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error recording video: {str(e)}")
            self.recording = False
            self.record_button.config(text="Start Recording")
        
    def stop_recording(self):
        self.recording = False
        self.record_button.config(text="Start Recording")
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
    def load_video_for_annotation(self):
        try:
            self.cap = cv2.VideoCapture(self.video_filename)
            self.total_frames = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), self.max_frames)
            self.timeline_slider.config(to=self.total_frames)
            
            # Load audio waveform using librosa
            if os.path.exists(self.audio_filename):
                y, sr = librosa.load(self.audio_filename)
                times = np.arange(len(y)) / sr
                
                self.ax.clear()
                self.ax.plot(times, y)
                self.ax.set_xlabel('Time (s)')
                self.ax.set_ylabel('Amplitude')
                self.ax.set_xlim(0, 3)  # Set x-axis limit to 3 seconds
                self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error loading video: {str(e)}")
        
    def update_frame(self, value):
        if self.cap is not None:
            frame_pos = int(float(value))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self.cap.read()
            if ret:
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.preview_label.imgtk = imgtk
                self.preview_label.configure(image=imgtk)
                self.frame_label.config(text=f"Frame: {frame_pos}/{self.total_frames}")
                
    def set_start_time(self):
        self.current_start = int(self.timeline_var.get())
        messagebox.showinfo("Start Frame", f"Start frame set to: {self.current_start}")
        
    def set_end_time(self):
        current_end = int(self.timeline_var.get())
        word = self.word_entry.get()
        if hasattr(self, 'current_start'):
            if word.strip():
                self.annotations.append((self.current_start, current_end, word))
                self.tree.insert('', 'end', values=(self.current_start, current_end, word))
                self.word_entry.delete(0, tk.END)
            else:
                messagebox.showwarning("Warning", "Please enter a word before setting end time")
                
    def save_annotations(self):
        if not self.annotations:
            messagebox.showwarning("Warning", "No annotations to save")
            return
            
        align_filename = self.get_alignment_path(f"{self.filename_entry.get()}.align")
        try:
            with open(align_filename, 'w') as f:
                # Add initial silence if needed
                if self.annotations and self.annotations[0][0] > 0:
                    start_norm = 0
                    end_norm = self.normalize_frame_number(self.annotations[0][0])
                    f.write(f"{start_norm} {end_norm} sil\n")
                    
                # Write normalized annotations
                for start, end, word in sorted(self.annotations):
                    start_norm = self.normalize_frame_number(start)
                    end_norm = self.normalize_frame_number(end)
                    f.write(f"{start_norm} {end_norm} {word}\n")
                    
                # Add final silence if needed
                if self.annotations and self.annotations[-1][1] < self.total_frames:
                    start_norm = self.normalize_frame_number(self.annotations[-1][1])
                    end_norm = self.target_max_frame
                    f.write(f"{start_norm} {end_norm} sil\n")
                    
            messagebox.showinfo("Success", f"Saved annotations to {align_filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving annotations: {str(e)}")

    def normalize_frame_number(self, frame_num):
        """Normalize frame numbers to range 0-74500"""
        if self.total_frames == 0:
            return 0
        return int((frame_num / self.total_frames) * self.target_max_frame)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = VideoAnnotator(root)
    root.mainloop()