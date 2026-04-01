import os
import cv2
import mediapipe as mp
import math
import time
import json
import serial
import serial.tools.list_ports
from datetime import datetime, timedelta

os.environ["GLOG_minloglevel"] = "2"

CAMERA_INDEX_OPTIONS = [1, 0]
FRAME_WIDTH = 640
FRAME_HEIGHT = 320
EAR_THRESHOLD = 0.18
SLEEP_TRIGGER_TIME = 1.25
MAX_RECORDING_DURATION = 300
NO_FACE_IDLE_TIMEOUT = 5
NO_LANDMARK_THRESHOLD = 10
BAUD_RATE = 9600

SAVE_PATH = "saved"
VIDEO_FOLDER = os.path.join(SAVE_PATH, "video")
JSON_PATH = os.path.join(SAVE_PATH, "main.json")

os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

def load_json_safe(path):
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return {"accident": {}, "connected-ino": False}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("[ERROR] JSON is invalid or empty. Resetting.")
        return {"accident": {}, "connected-ino": False}

def find_arduino_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        print(f"[DEBUG] Port: {port.device}, Desc: {port.description}")
        if any(x in port.description for x in ["Arduino", "CH340", "USB-SERIAL", "Serial", "CP210"]):
            print(f"[INFO] Arduino found at {port.device}")
            return port.device
    print("[ERROR] No Arduino found.")
    return None

def get_available_camera():
    for index in CAMERA_INDEX_OPTIONS:
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is not None and cap.read()[0]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            print(f"[INFO] Using camera : {index}")
            return cap
        cap.release()
    raise Exception("[WARNING] Camera Status : No camera found!")

def update_arduino_status():
    port = find_arduino_port()
    if port is None:
        return None, False
    try:
        arduino = serial.Serial(port, BAUD_RATE)
        time.sleep(2)
        print("[INFO] Arduino status : connected.")
        arduino_connected = True
    except:
        arduino = None
        arduino_connected = False
        print("[WARNING] Arduino is not connected.")

    try:
        data = load_json_safe(JSON_PATH)
        data["connected-ino"] = arduino_connected
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except:
        print("[WARNING] JSON file couldn't be updated.")
    return arduino, arduino_connected

cap = get_available_camera()
arduino, arduino_connected = update_arduino_status()

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(landmarks, eye_indices):
    try:
        A = math.dist(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
        B = math.dist(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
        C = math.dist(landmarks[eye_indices[0]], landmarks[eye_indices[3]])
        return (A + B) / (2.0 * C)
    except:
        return 0.0

def create_accident_entry(counter, sleep_time, wake_time=None, video_path=""):
    duration = str(timedelta(seconds=int(wake_time - sleep_time))) if wake_time else "00:00:00"
    dt_str = datetime.fromtimestamp(sleep_time).strftime("%d/%m/%Y %H:%M:%S")
    key = f"accident-{counter}"

    try:
        data = load_json_safe(JSON_PATH)
        data["accident"][key] = {
            "time": dt_str,
            "opened": bool(wake_time),
            "how-long": duration,
            "video-path": video_path.replace("\\", "/")
        }
        with open(JSON_PATH, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print("[WARNING] Failed to write accident data:", e)

def delete_old_videos(folder, days_old=30):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and (now - os.path.getctime(path)) > (days_old * 86400):
            os.remove(path)

paused = False
closed_start_time = None
video_writer = None
video_filename = ""
video_start_time = None
accident_counter = 1
recording = False
face_last_seen = time.time()
no_landmark_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from camera.")
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        paused = False
        face_last_seen = time.time()

        face = results.multi_face_landmarks[0]
        landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face.landmark]

        mp_drawing.draw_landmarks(
            frame, face,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
        )

        visible_ears = []
        for eye in (LEFT_EYE, RIGHT_EYE):
            if all(0 <= landmarks[i][0] < w and 0 <= landmarks[i][1] < h for i in eye):
                visible_ears.append(eye_aspect_ratio(landmarks, eye))

        if visible_ears:
            avg_ear = sum(visible_ears) / len(visible_ears)
            eyes_closed = all(ear < EAR_THRESHOLD for ear in visible_ears)
            no_landmark_counter = 0
        else:
            no_landmark_counter += 1
            avg_ear = 0.0
            eyes_closed = no_landmark_counter > NO_LANDMARK_THRESHOLD

        if eyes_closed:
            if closed_start_time is None:
                closed_start_time = time.time()
            elif time.time() - closed_start_time >= SLEEP_TRIGGER_TIME:
                cv2.putText(frame, "ALERT: Don't Close Your Eyes!", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if not recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    video_filename = os.path.join(VIDEO_FOLDER, f"{timestamp}.avi")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    video_writer = cv2.VideoWriter(video_filename, fourcc, 20.0, (w, h))
                    video_start_time = time.time()
                    accident_start_time = video_start_time
                    recording = True
                    if arduino:
                        arduino.write(b'1')
        else:
            if recording:
                wake_time = time.time()
                create_accident_entry(accident_counter, accident_start_time, wake_time, video_filename)
                accident_counter += 1
                video_writer.release()
                video_writer = None
                recording = False
                if arduino:
                    arduino.write(b'0')
            closed_start_time = None
            no_landmark_counter = 0

        status = "[INFO] Eyes Status : Closed" if eyes_closed else "[INFO] Eyes Status : Open"
        color = (0, 0, 255) if eyes_closed else (0, 255, 0)
        cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    else:
        if not paused and time.time() - face_last_seen > NO_FACE_IDLE_TIMEOUT:
            paused = True
            if arduino:
                arduino.write(b'2')

        if paused:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dimmed = cv2.convertScaleAbs(gray, alpha=0.5)
            dimmed_bgr = cv2.cvtColor(dimmed, cv2.COLOR_GRAY2BGR)
            cv2.putText(dimmed_bgr, "Idle Mode: No Face detected!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 255), 2)
            cv2.imshow("Eye Detection", dimmed_bgr)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

    if recording and video_writer:
        timestamp_str = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        text_size, _ = cv2.getTextSize(timestamp_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_x = w - text_size[0] - 10
        cv2.putText(frame, timestamp_str, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        video_writer.write(frame)

        if time.time() - video_start_time >= MAX_RECORDING_DURATION:
            create_accident_entry(accident_counter, accident_start_time, None, video_filename)
            accident_counter += 1
            video_writer.release()
            video_writer = None
            recording = False
            if arduino:
                arduino.write(b'0')

    cv2.imshow("Eye Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
if video_writer:
    video_writer.release()
cv2.destroyAllWindows()
delete_old_videos(VIDEO_FOLDER, days_old=30)
