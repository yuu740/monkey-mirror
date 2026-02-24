import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

img_default = cv2.imread("./assets/monkey-default.jpg")
img_scream = cv2.imread("./assets/monkey-scream.jpg")
img_thinking = cv2.imread("./assets/monkey-thinking.jpg")
img_aha = cv2.imread("./assets/monkey-aha.jpg")

if img_default is None:
    print("Error: Asset gambar tidak ditemukan!")
    exit()

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def clamp(n, minn, maxn):
    return max(min(n, maxn), minn)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    results = model(frame, verbose=False, conf=0.5)
    
    scores = {"AHA": 0, "THINKING": 0, "SCREAM": 0, "DEFAULT": 0}
    current_monkey = img_default
    status = "DEFAULT"

    if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
        kpts = results[0].keypoints.xy[0].cpu().numpy()
        
        if len(kpts) > 10:
            nose = kpts[0]
            eye_l, eye_r = kpts[1], kpts[2]
            wrist_l, wrist_r = kpts[9], kpts[10]
            
            eye_dist = get_dist(eye_l, eye_r) + 1e-6 

            aha_val = max(0, (eye_l[1] - wrist_l[1]), (eye_r[1] - wrist_r[1]))
            scores["AHA"] = min(100, int((aha_val / eye_dist) * 100))

            dist_to_nose = min(get_dist(wrist_l, nose), get_dist(wrist_r, nose))
            scores["THINKING"] = min(100, int(max(0, 100 - (dist_to_nose / eye_dist) * 50)))

            mouth_y, mouth_x = int(nose[1] + eye_dist*0.5), int(nose[0])
            roi_h, roi_w = int(eye_dist), int(eye_dist)
            if 0 < mouth_y < h-roi_h and 0 < mouth_x < w-roi_w:
                roi = frame[mouth_y:mouth_y+roi_h, mouth_x-int(roi_w/2):mouth_x+int(roi_w/2)]
                if roi.size > 0:
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
                    dark_pixels = cv2.countNonZero(thresh)
                    scores["SCREAM"] = min(100, int((dark_pixels / (roi.size/3)) * 500))

            top_state = max(scores, key=scores.get)
            if scores[top_state] > 40: 
                status = top_state
                if status == "AHA": current_monkey = img_aha
                elif status == "THINKING": current_monkey = img_thinking
                elif status == "SCREAM": current_monkey = img_scream
            else:
                scores["DEFAULT"] = 100
                status = "DEFAULT"

    input_view = frame.copy()
    if results[0].keypoints is not None:
        for kp in kpts:
            if kp[0] > 0: cv2.circle(input_view, (int(kp[0]), int(kp[1])), 5, (0, 255, 0), -1)

    output_view = cv2.resize(current_monkey, (w, h))

    combined_frame = np.hstack((input_view, output_view))

    y_offset = 100
    for state, score in scores.items():
        color = (0, 255, 0) if state == status else (200, 200, 200)
        cv2.putText(combined_frame, f"{state}: {score}%", (w + 20, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_offset += 40

    cv2.imshow("Monkey Mirror AI - Dual View Mode", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()