from deteksi_kantuk import load_calibration, save_calibration, calibrate_ear, calibrate_mar
from main import main
import cv2

def run():
    name = input("Masukkan nama Anda: ")

    # Periksa data kalibrasi
    ear_threshold, mar_threshold = load_calibration(name)
    if ear_threshold is None or mar_threshold is None:
        print("Data kalibrasi tidak ditemukan. Memulai kalibrasi...")
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Kamera tidak dapat diakses!")
            return

        ear_threshold = calibrate_ear(camera)
        mar_threshold = calibrate_mar(camera)

        if ear_threshold and mar_threshold:
            save_calibration(name, ear_threshold, mar_threshold)
        else:
            print("Kalibrasi gagal, keluar dari program.")
            return

    print(f"Mulai deteksi kantuk dengan EAR: {ear_threshold:.2f} dan MAR: {mar_threshold:.2f}")
    main(ear_threshold, mar_threshold)

if __name__ == "__main__":
    run()
