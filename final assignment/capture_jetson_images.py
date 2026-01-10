import cv2
import os
from datetime import datetime
import threading
from flask import Flask, Response
import time

# --- CONFIGURATION ---
OUTPUT_DIR = "jetson_dataset"  # Where to save images
CLASSES = ['middle', 'peace', 'woensel']  # Your 3 gestures
STREAM_PORT = 5000  # Port for web streaming

# Global variables for streaming
current_frame = None
frame_lock = threading.Lock()

# Flask app for streaming
app = Flask(__name__)

def get_pipeline():
    """GStreamer pipeline for Jetson CSI camera"""
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=21/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
    )

def generate_frames():
    """Generator function for MJPEG streaming"""
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                continue
            frame = current_frame.copy()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Homepage with video stream"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson Camera Capture</title>
        <style>
            body {{ 
                font-family: Arial; 
                text-align: center; 
                background: #222;
                color: white;
                padding: 20px;
            }}
            img {{ 
                max-width: 90%; 
                border: 3px solid #0f0;
                margin: 20px auto;
            }}
            h1 {{ color: #0f0; }}
            .instructions {{
                background: #333;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto;
                max-width: 600px;
            }}
        </style>
    </head>
    <body>
        <h1>📷 Jetson Camera - Image Capture Tool</h1>
        <div class="instructions">
            <h2>Instructions</h2>
            <p>Go back to your SSH terminal and press:</p>
            <p><strong>1</strong> = Capture MIDDLE gesture</p>
            <p><strong>2</strong> = Capture PEACE gesture</p>
            <p><strong>3</strong> = Capture WOENSEL gesture</p>
            <p><strong>q</strong> = Quit</p>
        </div>
        <img src="/video_feed" alt="Live Camera Feed">
    </body>
    </html>
    """
    return html

def main():
    # Create output directories
    for class_name in CLASSES:
        os.makedirs(f"{OUTPUT_DIR}/{class_name}", exist_ok=True)
    
    # Initialize camera
    print("Starting Camera...")
    cap = cv2.VideoCapture(get_pipeline(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("\n=== JETSON IMAGE CAPTURE TOOL ===")
    print("Instructions:")
    print("  Press '1' to capture MIDDLE finger gesture")
    print("  Press '2' to capture PEACE gesture")
    print("  Press '3' to capture WOENSEL gesture")
    print("  Press 'q' to quit")
    print("\nReady! Position your hand and press the number key.\n")
    
    # Counters for each class
    counters = {cls: len(os.listdir(f"{OUTPUT_DIR}/{cls}")) for cls in CLASSES}
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        
        # Display live feed with instructions
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press 1=MIDDLE | 2=PEACE | 3=WOENSEL | Q=QUIT", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show current counts
        y_offset = 60
        for i, cls in enumerate(CLASSES):
            text = f"{cls}: {counters[cls]} images"
            cv2.putText(display_frame, text, (10, y_offset + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame (this won't work over SSH, but will work if you have a display)
        try:
            cv2.imshow("Capture Tool", display_frame)
        except:
            pass  # Ignore if no display available
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('1'):
            selected_class = CLASSES[0]  # middle
        elif key == ord('2'):
            selected_class = CLASSES[1]  # peace
        elif key == ord('3'):
            selected_class = CLASSES[2]  # woensel
        else:
            continue  # No valid key pressed
        
        # Save the image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{OUTPUT_DIR}/{selected_class}/img_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        counters[selected_class] += 1
        print(f"✓ Saved {selected_class} #{counters[selected_class]}: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n=== CAPTURE SUMMARY ===")
    for cls in CLASSES:
        print(f"{cls}: {counters[cls]} images")
    print(f"\nImages saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
