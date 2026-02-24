import cv2
import numpy as np
import mediapipe as mp

try:
    from mediapipe.python.solutions import hands as mp_hands
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    print("Berhasil memuat sub-modul melalui jalur internal!")
except ImportError:
    print("Error: Jalur internal tidak ditemukan. Instalasi MediaPipe tidak sempurna.")
    exit()

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, refine_landmarks=True)

img_default = cv2.imread("monkey-default.jpg") 
img_scream = cv2.imread("monkey-scream.jpg")   
img_thinking = cv2.imread("monkey-thinking.jpg") 
img_aha = cv2.imread("monkey-aha.jpg")        

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
    
    hand_results = hands.process(rgb_frame)
    face_results = face_mesh.process(rgb_frame)
    
    current_monkey = img_default
    
    display = cv2.resize(current_monkey, (w, h))
    cv2.imshow("Monkey Mirror AI", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()