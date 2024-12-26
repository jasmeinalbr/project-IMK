import cv2
import mediapipe as mp
import numpy as np
import csv
import os
import time
import pandas as pd
import pygame

# Konfigurasi Mediapipe untuk Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Landmark untuk mata dan mulut
LEFT_EYE_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 81, 181, 78, 308, 178, 402]

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

def calibrate_ear(camera):   
    print("Memulai kalibrasi EAR...")
    ear_values = []
    count = 0
    consecutive_valid_frames = 0
    last_detection_time = time.time() - 3  # Mulai dengan waktu yang cukup untuk jeda
    pygame.mixer.init()

    while count < 3:  # Mengumpulkan 3 sampel EAR yang valid
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Kotak untuk panduan posisi wajah
        center_x, center_y = width // 2, height // 2
        box_size = 200  # Ukuran kotak diperbesar
        frame_color = (0, 0, 255)  # Default merah
        status_text = "Muka Anda di luar frame"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dapatkan EAR
                landmarks = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in face_landmarks.landmark]
                left_ear = calculate_ear(landmarks, LEFT_EYE_POINTS)
                right_ear = calculate_ear(landmarks, RIGHT_EYE_POINTS)
                ear = (left_ear + right_ear) / 2.0 

                # Periksa apakah wajah dalam kotak
                face_x = int(face_landmarks.landmark[1].x * width)
                face_y = int(face_landmarks.landmark[1].y * height)
                if (center_x - box_size < face_x < center_x + box_size and
                        center_y - box_size < face_y < center_y + box_size):
                    frame_color = (0, 255, 0)  # Hijau
                    status_text = "Muka Anda dalam frame"

                    if ear < 0.2:  # EAR rendah menunjukkan mata tertutup
                        consecutive_valid_frames += 1
                        frame_color = (255, 0, 0)  # Biru
                        status_text = "Tahan sebentar"
                        if consecutive_valid_frames >= 90:  # Stabil selama 3 detik pada 30 FPS
                            if time.time() - last_detection_time > 3:
                                ear_values.append(ear)
                                count += 1
                                last_detection_time = time.time()  # Reset waktu terakhir
                                pygame.mixer.Sound("ping.mp3").play()  # Suara ping
                                consecutive_valid_frames = 0
                    else:
                        consecutive_valid_frames = 0

        # Gambar kotak dan status
        cv2.rectangle(frame, (center_x - box_size, center_y - box_size),
                      (center_x + box_size, center_y + box_size), frame_color, 2)
        cv2.putText(frame, status_text, (center_x - 150, center_y - box_size - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, frame_color, 2)

        # Tampilkan perintah dan percobaan keberapa
        cv2.putText(frame, "Tutup mata Anda.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Percobaan {count}/3", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Kalibrasi EAR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0
    avg_ear_adjusted = avg_ear * 0.8  # Ambil 80% dari rata-rata untuk threshold
    cv2.destroyAllWindows()
    print(f"Kalibrasi EAR selesai. Threshold EAR: {avg_ear_adjusted:.2f}")
    return avg_ear_adjusted

def calibrate_mar(camera):
    print("Memulai kalibrasi MAR...")
    mar_values = []
    count = 0
    consecutive_valid_frames = 0
    last_detection_time = time.time() - 3  # Mulai dengan waktu yang cukup untuk jeda
    pygame.mixer.init()

    while count < 3:  # Mengumpulkan 3 sampel MAR yang valid
        ret, frame = camera.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Kotak untuk panduan posisi wajah
        center_x, center_y = width // 2, height // 2
        box_size = 200  # Ukuran kotak diperbesar
        frame_color = (0, 0, 255)  # Default merah
        status_text = "Muka Anda di luar frame"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Dapatkan MAR
                landmarks = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in face_landmarks.landmark]
                mar = calculate_mar(landmarks, MOUTH_POINTS) 

                # Periksa apakah wajah dalam kotak
                face_x = int(face_landmarks.landmark[1].x * width)
                face_y = int(face_landmarks.landmark[1].y * height)
                if (center_x - box_size < face_x < center_x + box_size and
                        center_y - box_size < face_y < center_y + box_size):
                    frame_color = (0, 255, 0)  # Hijau
                    status_text = "Muka Anda dalam frame"

                    if mar > 9.0:  # MAR tinggi menunjukkan mulut terbuka
                        consecutive_valid_frames += 1
                        frame_color = (255, 0, 0)  # Biru
                        status_text = "Tahan sebentar"
                        if consecutive_valid_frames >= 90:  # Stabil selama 3 detik pada 30 FPS
                            if time.time() - last_detection_time > 3:
                                mar_values.append(mar)
                                count += 1
                                last_detection_time = time.time()  # Reset waktu terakhir
                                pygame.mixer.Sound("ping.mp3").play()  # Suara ping
                                consecutive_valid_frames = 0
                    else:
                        consecutive_valid_frames = 0

        # Gambar kotak dan status
        cv2.rectangle(frame, (center_x - box_size, center_y - box_size),
                      (center_x + box_size, center_y + box_size), frame_color, 2)
        cv2.putText(frame, status_text, (center_x - 150, center_y - box_size - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, frame_color, 2)

        # Tampilkan perintah dan percobaan keberapa
        cv2.putText(frame, "Buka mulut Anda.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Percobaan {count}/3", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Kalibrasi MAR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    avg_mar = sum(mar_values) / len(mar_values) if mar_values else 0
    avg_mar_adjusted = avg_mar * 1.2  # Ambil 120% dari rata-rata untuk threshold
    cv2.destroyAllWindows()
    print(f"Kalibrasi MAR selesai. Threshold MAR: {avg_mar_adjusted:.2f}")
    return avg_mar_adjusted


# Fungsi untuk menyimpan data kalibrasi ke CSV
def save_calibration(name, ear_threshold, mar_threshold):
    file_exists = os.path.isfile("calibration_data.csv")
    with open("calibration_data.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Name", "EAR_Threshold", "MAR_Threshold"])
        writer.writerow([name, ear_threshold, mar_threshold])
    print(f"Data kalibrasi untuk {name} berhasil disimpan.")

# Fungsi untuk memuat data kalibrasi
def load_calibration(name):
    try:
        data = pd.read_csv("calibration_data.csv")
        user_data = data[data["Name"] == name]
        if not user_data.empty:
            ear_threshold = user_data.iloc[0]["EAR_Threshold"]
            mar_threshold = user_data.iloc[0]["MAR_Threshold"]
            return ear_threshold, mar_threshold
        else:
            print(f"Data untuk {name} tidak ditemukan!")
            return None, None
    except FileNotFoundError:
        print("File calibration_data.csv tidak ditemukan!")
        return None, None

# Fungsi utama untuk kalibrasi dan deteksi
def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Kamera tidak dapat diakses!")
        return

    print("Halo, saya pendeteksi kantuk. Izinkan saya untuk mengkalibrasi muka Anda.")
    time.sleep(2)

    name = input("Masukkan nama Anda: ")

    # Periksa apakah data kalibrasi sudah ada
    ear_threshold, mar_threshold = load_calibration(name)
    if ear_threshold is None or mar_threshold is None:
        print("Data kalibrasi tidak ditemukan, memulai kalibrasi baru...")
        ear_threshold = calibrate_ear(camera)
        mar_threshold = calibrate_mar(camera)
        if ear_threshold and mar_threshold:
            save_calibration(name, ear_threshold, mar_threshold)
        else:
            print("Kalibrasi gagal, keluar dari program.")
            return

    print(f"Deteksi kantuk akan menggunakan EAR: {ear_threshold:.2f} dan MAR: {mar_threshold:.2f}")
    print("Mulai deteksi...")

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # Flip frame (mirror effect)
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame for face landmarks
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in face_landmarks.landmark]
                ear = calculate_ear(landmarks, LEFT_EYE_POINTS + RIGHT_EYE_POINTS)
                mar = calculate_mar(landmarks, MOUTH_POINTS)

                status = "Normal"
                if ear < ear_threshold:
                    status = "Mata Tertutup"
                if mar > mar_threshold:
                    status = "Mulut Terbuka"

                # Display status
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame with mirror effect
        cv2.imshow("Deteksi Kantuk", frame)

        # Check for key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 127:  # Tombol delete untuk reset kalibrasi
            print("Reset data kalibrasi...")
            ear_threshold, mar_threshold = calibrate_ear(camera), calibrate_mar(camera)
            save_calibration(name, ear_threshold, mar_threshold)

    camera.release()
    cv2.destroyAllWindows()
