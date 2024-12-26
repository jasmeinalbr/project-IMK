import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

# Inisialisasi Pygame untuk alarm
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm2.mp3")

# Fungsi untuk menghitung jarak Euclidean
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Fungsi untuk menghitung EAR (Eye Aspect Ratio)
def calculate_ear(landmarks, eye_points):
    P1, P2, P3, P4, P5, P6 = [landmarks[point] for point in eye_points]
    vertical1 = euclidean_distance(P2, P6)
    vertical2 = euclidean_distance(P3, P5)
    horizontal = euclidean_distance(P1, P4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear

# Fungsi untuk menghitung MAR (Mouth Aspect Ratio)
def calculate_mar(landmarks, mouth_points):
    P1, P2, P3, P4, P5, P6, P7, P8 = [landmarks[point] for point in mouth_points]
    vertical1 = euclidean_distance(P2, P8)
    vertical2 = euclidean_distance(P3, P7)
    vertical3 = euclidean_distance(P4, P6)
    horizontal = euclidean_distance(P1, P5)
    mar = (vertical1 + vertical2 + vertical3) / (2.0 * horizontal)
    return mar

# Konfigurasi Mediapipe untuk Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Landmark untuk mata dan mulut
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 81, 181, 78, 308, 178, 402]

# Variabel state untuk perhitungan
state = {
    "eye": {"start_time": None, "count": 0, "last_reset": time.time()},
    "mouth": {"start_time": None, "count": 0, "last_reset": time.time()},
}

# Fungsi untuk memutar alarm
def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("alarm2.mp3")
        pygame.mixer.music.play()

# Fungsi untuk menghentikan alarm
def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# Fungsi untuk mendeteksi kantuk
def detect_drowsiness(frame, ear_threshold, mar_threshold):
    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi landmark wajah
    results = face_mesh.process(rgb_frame)
    status = "Normal"  # default status
    frame_color = (0, 255, 0)  # default green

    ear = 0
    mar = 0

    current_time = time.time()

    # Reset count jika melebihi waktu window
    for key in ["eye", "mouth"]:
        if current_time - state[key]["last_reset"] > 5 * 60:  # 5 menit
            state[key]["count"] = 0
            state[key]["last_reset"] = current_time

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(pt.x * width), int(pt.y * height)) for pt in face_landmarks.landmark]
            
            # Hitung EAR untuk mata kiri dan kanan
            left_ear = calculate_ear(landmarks, LEFT_EYE_POINTS)
            right_ear = calculate_ear(landmarks, RIGHT_EYE_POINTS)
            ear = (left_ear + right_ear) / 2.0

            # Hitung MAR untuk mulut
            mar = calculate_mar(landmarks, MOUTH_POINTS)

            # Deteksi mata tertutup
            if ear < ear_threshold:
                if state["eye"]["start_time"] is None:
                    state["eye"]["start_time"] = current_time
                elif current_time - state["eye"]["start_time"] > 2:
                    state["eye"]["count"] += 1
                    state["eye"]["start_time"] = None
                    if state["eye"]["count"] >= 3:
                        play_alarm()
                        state["eye"]["count"] = 0  # Reset setelah mencapai 3
                status = "Mata Tertutup"
                frame_color = (0, 0, 255)  # merah
            else:
                state["eye"]["start_time"] = None

            # Deteksi mulut terbuka
            if mar > mar_threshold:
                if state["mouth"]["start_time"] is None:
                    state["mouth"]["start_time"] = current_time
                elif current_time - state["mouth"]["start_time"] > 2:
                    state["mouth"]["count"] += 1
                    state["mouth"]["start_time"] = None
                    if state["mouth"]["count"] >= 3:
                        play_alarm()
                        state["mouth"]["count"] = 0  # Reset setelah mencapai 3
                status = "Mulut Terbuka"
                frame_color = (0, 0, 255)  # merah
            else:
                state["mouth"]["start_time"] = None

            # Gambar kotak di sekitar wajah
            x_min = min([int(pt.x * width) for pt in face_landmarks.landmark])
            y_min = min([int(pt.y * height) for pt in face_landmarks.landmark])
            x_max = max([int(pt.x * width) for pt in face_landmarks.landmark])
            y_max = max([int(pt.y * height) for pt in face_landmarks.landmark])

            # Gambar kotak dan status
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), frame_color, 2)
            cv2.putText(frame, status, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, frame_color, 2)

    # Tampilkan jumlah yawn dan mata tertutup di kiri atas
    cv2.putText(frame, f"Menguap: {state['mouth']['count']} | Mata Tertutup: {state['eye']['count']}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Tampilkan nilai EAR dan MAR di ujung kanan atas
    cv2.putText(frame, f"EAR: {ear:.2f}", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {mar:.2f}", (width - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

# Fungsi utama untuk menjalankan program
def main(ear_threshold, mar_threshold):
    # Terapkan penyesuaian pada threshold
    ear_threshold_adjusted = ear_threshold + 0.06  # Tambahkan 0.06 ke EAR
    mar_threshold_adjusted = mar_threshold - 4.5  # Kurangi 3.5 dari MAR
 


    print(f"Mulai deteksi kantuk dengan EAR: {ear_threshold_adjusted:.2f} dan MAR: {mar_threshold_adjusted:.2f}")

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Kamera tidak dapat diakses!")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Mirror frame
        frame = cv2.flip(frame, 1)  # Membalik frame secara horizontal

        # Gunakan threshold yang telah disesuaikan
        frame = detect_drowsiness(frame, ear_threshold_adjusted, mar_threshold_adjusted)
        cv2.imshow("Deteksi Kantuk", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()