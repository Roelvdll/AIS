import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = 'gesture_recognition_model.keras'
IMG_HEIGHT = 128
IMG_WIDTH = 128
# Make sure these match the order in your dataset/ folder
CLASS_NAMES = ['middle', 'peace', 'woensel'] 

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=0,
):
    """
    GStreamer pipeline for CSI Camera on Jetson Nano.
    """
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def main():
    # 1. Load the model
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Initialize Camera (CSI via GStreamer)
    print("Starting CSI Camera...")
    pipeline = gstreamer_pipeline(flip_method=0)
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Fallback to standard VideoCapture if GStreamer fails (e.g. if using a USB cam)
    if not cap.isOpened():
        print("CSI Camera not found via GStreamer. Testing default index...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open any camera.")
        return

    print("Running inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Preprocess the frame
        # Resize to match training
        input_frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        # BGR (OpenCV) to RGB (Model)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        # Add batch dimension
        input_batch = np.expand_dims(input_frame, axis=0)

        # 4. Predict
        predictions = model.predict(input_batch, verbose=0)
        score = predictions[0] # Using softmax outputs directly
        class_idx = np.argmax(score)
        label = CLASS_NAMES[class_idx]
        confidence = score[class_idx] * 100

        # 5. Display results
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        text = f"{label}: {confidence:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Jetson Nano Gesture Guard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()