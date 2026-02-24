import cv2
import numpy as np

try:
    import mediapipe as mp
    print(f"Berhasil memuat MediaPipe dari: {mp.__file__}") 
except AttributeError:
    print("ERROR: Terjadi konflik nama! Pastikan tidak ada file bernama 'mediapipe.py' di folder ini.")
    exit()

# Inisialisasi
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7)

# Load Gambar (Gunakan path yang benar)
img_default = cv2.imread("monkey-default.jpg") #
img_scream = cv2.imread("monkey-scream.jpg")   #
img_thinking = cv2.imread("monkey-thinking.jpg") #
img_aha = cv2.imread("monkey-aha.jpg")        #

# Pastikan gambar berhasil di-load
if img_default is None:
    print("Error: File gambar monyet tidak ditemukan!")
    exit()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Deteksi
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    current_monkey = img_default
    
    # Logika deteksi tetap sama seperti sebelumnya...
    # (Gunakan jarak antar landmark bibir dan posisi jari telunjuk)
    
    # Tampilkan
    display = cv2.resize(current_monkey, (w, h))
    cv2.imshow("Monkey Mirror AI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()