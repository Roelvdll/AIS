import cv2
import numpy as np
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("Please install tflite-runtime or tensorflow")

# --- CONFIGURATION ---
MODEL_PATH = 'gesture_recognition_model.tflite'
IMG_HEIGHT = 128
IMG_WIDTH = 128
CLASS_NAMES = ['middle', 'peace', 'woensel']

def main():
    print("Loading TFLite model...")
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Starting Camera...")
    
    def get_pipeline(sensor_id):
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=640, height=360, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! appsink drop=1"
        )

    # 1. Try CSI Camera on Sensor 0
    print("Trying CSI Camera (Sensor 0)...")
    cap = cv2.VideoCapture(get_pipeline(0), cv2.CAP_GSTREAMER)

    # 2. Try CSI Camera on Sensor 1 (if Sensor 0 fails)
    if not cap.isOpened():
        print("Sensor 0 failed. Trying CSI Camera (Sensor 1)...")
        cap = cv2.VideoCapture(get_pipeline(1), cv2.CAP_GSTREAMER)

    # 3. Fallback to standard V4L2 (USB)
    if not cap.isOpened():
        print("CSI pipeline failed. Trying generic VideoCapture(0)...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open any camera.")
        return

    print("Running inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Preprocess
        img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        input_data = np.expand_dims(img, axis=0)

        # 2. Run Inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # 3. Parse Results
        predictions = output_data[0]
        class_idx = np.argmax(predictions)
        label = CLASS_NAMES[class_idx]
        confidence = predictions[class_idx] * 100

        # 4. Display
        color = (0, 255, 0) if confidence > 70 else (0, 0, 255)
        text = f"{label}: {confidence:.1f}%"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Jetson TFLite", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()