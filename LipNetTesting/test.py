import cv2
import numpy as np
import dlib

def preprocess_video(video_path, output_path, target_width=None, target_height=None, augment=False):
    """
    Preprocesses video by isolating lip regions, enhancing contrast, downsampling, and normalizing.

    Args:
        video_path: Path to the input video file.
        output_path: Path to save the preprocessed video.
        target_width: Desired width of the output frames (optional).
        target_height: Desired height of the output frames (optional).
        augment: Boolean flag to apply data augmentation (e.g., flipping).
    """
    # Load dlib face detector and facial landmarks predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Output format
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (target_width, target_height))  # Adjust FPS as needed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for landmark detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)
        if len(faces) > 0:
            # Use the first detected face
            face = faces[0]

            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Extract lip region (landmarks 48-67)
            lip_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
            x, y, w, h = cv2.boundingRect(lip_points)

            # Crop the lip region and resize it
            lip_frame = frame[y:y + h, x:x + w]
            lip_frame = cv2.resize(lip_frame, (target_width, target_height))

            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_frame = clahe.apply(cv2.cvtColor(lip_frame, cv2.COLOR_BGR2GRAY))
            enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_GRAY2BGR)

            # Data Augmentation (optional)
            if augment and np.random.rand() < 0.5:
                enhanced_frame = cv2.flip(enhanced_frame, 1)

            # Z-Normalization
            normalized_frame = (enhanced_frame - 127.5) / 127.5
            normalized_frame = ((normalized_frame + 1) * 127.5).astype('uint8')

            out.write(normalized_frame)
        else:
            print("No face detected in this frame. Skipping.")

    cap.release()
    out.release()

# Example usage
video_path = 'test_input.mp4'
output_path = 'preprocessed_lip_video2.mp4'
preprocess_video(video_path, output_path, target_width=224, target_height=224, augment=False)
