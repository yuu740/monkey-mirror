import cv2
import mediapipe as mp
import numpy as np

# Init mp 
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(refine_landmarks=True, min_detection_confidence=0.7)

# Read assets
img_default = cv2.imread("./assets/monkey-default.jpg")
img_scream = cv2.imread("./assets/monkey-scream.jpg")
img_thinking = cv2.imread("./assets/monkey-thinking.jpg")
img_aha = cv2.imread("./assets/monkey-aha.jpg")

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    face_res = face_mesh.process(rgb_frame)
    hand_res = hands.process(rgb_frame)
    
    scores = {"AHA": 0, "THINKING": 0, "SCREAM": 0, "DEFAULT": 0}
    current_monkey = img_default
    status = "DEFAULT"

    # 1. Scream (Lip distancing)
    if face_res.multi_face_landmarks:
        face = face_res.multi_face_landmarks[0].landmark
        u_lip = (face[13].x * w, face[13].y * h) # Upper lip
        l_lip = (face[14].x * w, face[14].y * h) # Lower lip
        
        eye_dist = get_dist((face[33].x*w, face[33].y*h), (face[263].x*w, face[263].y*h))
        mouth_open = get_dist(u_lip, l_lip)
        scores["SCREAM"] = min(100, int((mouth_open / eye_dist) * 350))

    # 2. Thinking & Aha (Hand logic)
    if hand_res.multi_hand_landmarks:
        hand = hand_res.multi_hand_landmarks[0].landmark
        index_tip = (hand[8].x * w, hand[8].y * h) # Index finger
        
        if face_res.multi_face_landmarks:
            face = face_res.multi_face_landmarks[0].landmark
            nose_tip = (face[164].x * w, face[164].y * h) # Nose
            ear_ref = (face[454].x * w, face[454].y * h) # Right ear
            
            # AHA
            if index_tip[1] < face[10].y * h:
                dist_to_ear = get_dist(index_tip, ear_ref)
                scores["AHA"] = min(100, int(max(0, 100 - (dist_to_ear / eye_dist) * 100)))

            # THINKING
            dist_to_mouth = get_dist(index_tip, nose_tip)
            scores["THINKING"] = min(100, int(max(0, 100 - (dist_to_mouth / eye_dist) * 120)))

    # Parameter threshold 35%
    top_state = max(scores, key=scores.get)
    if scores[top_state] > 35:
        status = top_state
        if status == "AHA": current_monkey = img_aha
        elif status == "THINKING": current_monkey = img_thinking
        elif status == "SCREAM": current_monkey = img_scream
    else:
        scores["DEFAULT"] = 100

    input_view = frame.copy()
    if hand_res.multi_hand_landmarks:
        for hl in hand_res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(input_view, hl, mp_hands.HAND_CONNECTIONS)

    output_view = cv2.resize(current_monkey, (w, h))
    combined = np.hstack((input_view, output_view))

    # Display Confidence Table
    y_off = 80
    for state, score in scores.items():
        color = (0, 255, 0) if state == status else (200, 200, 200)
        cv2.putText(combined, f"{state}: {score}%", (w + 20, y_off), 2, 0.7, color, 2)
        y_off += 40

    cv2.imshow("Monkey Mirror Ultra Precision", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()