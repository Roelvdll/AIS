import cv2
import os
from datetime import datetime
import threading
from flask import Flask, Response
import time
import sys
import select
import termios
import tty

# --- CONFIGURATION ---
OUTPUT_DIR = "jetson_dataset"
CLASSES = ['middle', 'peace', 'woensel']
STREAM_PORT = 5000

# Global variables
current_frame = None
frame_lock = threading.Lock()
app = Flask(__name__)

def get_pipeline():
    return (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1920, height=1080, format=NV12, framerate=21/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1"
    )

def generate_frames():
    global current_frame
    while True:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            frame = current_frame.copy()
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Jetson Camera Capture</title>
        <style>
            body { 
                font-family: Arial; 
                text-align: center; 
                background: #222;
                color: white;
                padding: 20px;
            }
            img { 
                max-width: 90%; 
                border: 3px solid #0f0;
                margin: 20px auto;
                display: block;
            }
            h1 { color: #0f0; }
            .instructions {
                background: #333;
                padding: 20px;
                border-radius: 10px;
                margin: 20px auto;
                max-width: 600px;
            }
            .key {
                background: #0f0;
                color: #000;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <h1>📷 Jetson Camera - Image Capture Tool</h1>
        <div class="instructions">
            <h2>Instructions</h2>
            <p>Go back to your SSH terminal and press:</p>
            <p><span class="key">1</span> = Capture MIDDLE gesture</p>
            <p><span class="key">2</span> = Capture PEACE gesture</p>
            <p><span class="key">3</span> = Capture WOENSEL gesture</p>
            <p><span class="key">q</span> = Quit</p>
        </div>
        <img src="/video_feed" alt="Live Camera Feed">
    </body>
    </html>
    """

def main():
    global current_frame
    
    # Create output directories
    for class_name in CLASSES:
        os.makedirs(f"{OUTPUT_DIR}/{class_name}", exist_ok=True)
    
    # Initialize camera
    print("Starting Camera...")
    cap = cv2.VideoCapture(get_pipeline(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Count existing images
    counters = {cls: len(os.listdir(f"{OUTPUT_DIR}/{cls}")) for cls in CLASSES}
    
    # Get IP address
    import socket
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    
    print(f"\n{'='*60}")
    print(f"🌐 OPEN THIS URL IN YOUR BROWSER:")
    print(f"   http://{ip_address}:{STREAM_PORT}")
    print(f"{'='*60}\n")
    
    # Start Flask in background
    flask_thread = threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=STREAM_PORT, 
                              debug=False, use_reloader=False, threaded=True)
    )
    flask_thread.daemon = True
    flask_thread.start()
    
    time.sleep(2)  # Let Flask start
    
    print("\n=== JETSON IMAGE CAPTURE TOOL ===")
    print("Instructions:")
    print("  Press '1' to capture MIDDLE finger gesture")
    print("  Press '2' to capture PEACE gesture")
    print("  Press '3' to capture WOENSEL gesture")
    print("  Press 'q' to quit")
    print("\nReady! Position your hand and press the number key.\n")
    
    # Terminal setup for instant key detection
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            # Add overlay text
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press 1=MIDDLE | 2=PEACE | 3=WOENSEL | Q=QUIT", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show counts
            y = 60
            for i, cls in enumerate(CLASSES):
                text = f"{cls}: {counters[cls]} images"
                cv2.putText(display_frame, text, (10, y + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Update stream
            with frame_lock:
                current_frame = display_frame
            
            # Check for key press (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                
                if key.lower() == 'q':
                    print("\n\nQuitting...")
                    break
                elif key in ['1', '2', '3']:
                    idx = int(key) - 1
                    selected_class = CLASSES[idx]
                    
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{OUTPUT_DIR}/{selected_class}/img_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    counters[selected_class] += 1
                    print(f"✓ Saved {selected_class} #{counters[selected_class]}: {filename}")
            
            time.sleep(0.03)  # ~30 fps
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
    
    cap.release()
    
    print("\n=== CAPTURE SUMMARY ===")
    for cls in CLASSES:
        print(f"{cls}: {counters[cls]} images")
    print(f"\nImages saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
