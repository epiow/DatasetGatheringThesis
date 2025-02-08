import cv2

# List available devices (this will check first 10 device indices)
def list_video_devices():
    available_devices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_devices.append(i)
            cap.release()
    return available_devices

# Example usage
devices = list_video_devices()
print(f"Available devices: {devices}")