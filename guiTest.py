def preprocess_frame(self, frame):
    """Memory efficient frame preprocessing"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = self.detector(gray)
    
    if len(faces) > 0:
        face = faces[0]
        landmarks = self.predictor(gray, face)
        mouth_points = [(landmarks.part(n).x, landmarks.part(n).y) 
                       for n in range(48, 68)]
        
        x_coords = [p[0] for p in mouth_points]
        y_coords = [p[1] for p in mouth_points]
        padding = 5
        
        x_min = max(0, min(x_coords) - padding)
        y_min = max(0, min(y_coords) - padding)
        x_max = min(gray.shape[1], max(x_coords) + padding)
        y_max = min(gray.shape[0], max(y_coords) + padding)
        
        mouth_roi = gray[y_min:y_max, x_min:x_max]
        lip_frame = cv2.resize(mouth_roi, (self.target_width, self.target_height))
        
        # Use numpy for preprocessing instead of TensorFlow to reduce memory usage
        lip_frame = lip_frame.astype(np.float32)
        mean = np.mean(lip_frame)
        std = np.std(lip_frame)
        lip_frame = (lip_frame - mean) / (std + 1e-6)  # Add small epsilon to avoid division by zero
        return np.expand_dims(lip_frame, axis=-1)
    
    # Fallback to static crop
    static_crop = gray[190:236, 80:220]
    static_crop = cv2.resize(static_crop, (self.target_width, self.target_height))
    static_crop = static_crop.astype(np.float32)
    mean = np.mean(static_crop)
    std = np.std(static_crop)
    static_crop = (static_crop - mean) / (std + 1e-6)
    return np.expand_dims(static_crop, axis=-1)

def process_video(self, source):
    """Process video with batch processing to manage memory"""
    frames = self.collect_frames(source)
    if frames is None:
        return "Error: Failed to collect frames"
    
    # Process frames in smaller batches
    BATCH_SIZE = 25  # Process 25 frames at a time
    processed_frames = []
    
    try:
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i:i + BATCH_SIZE]
            batch_processed = []
            
            for frame in batch_frames:
                processed = self.preprocess_frame(frame)
                if processed is not None:
                    batch_processed.append(processed)
            
            if batch_processed:
                processed_frames.extend(batch_processed)
                
            # Clear memory
            tf.keras.backend.clear_session()
            
        if len(processed_frames) < self.sequence_length:
            return "Error: Failed to process enough frames"
        
        # Ensure we only use the required number of frames
        processed_frames = processed_frames[:self.sequence_length]
        
        # Stack frames and make prediction
        X = np.stack(processed_frames)
        X = np.expand_dims(X, axis=0)
        
        # Run prediction with memory optimization
        with tf.device('/CPU:0'):  # Force CPU usage if GPU memory is limited
            yhat = self.model.predict(X, batch_size=1)
            
        decoded = tf.keras.backend.ctc_decode(yhat, 
                                            input_length=[75,75], 
                                            greedy=True)[0][0].numpy()
        
        prediction_chars = []
        for num in decoded[0]:
            if num >= 0:
                char = num_to_char(num).numpy().decode('utf-8')
                prediction_chars.append(char)
        
        return ''.join(prediction_chars)
        
    except Exception as e:
        return f"Error during inference: {str(e)}"

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
            self.model.load_weights('test.weights.h5')
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