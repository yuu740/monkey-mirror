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
            ear_l, ear_r = kpts[3], kpts[4] 
            wrist_l, wrist_r = kpts[9], kpts[10] 
            
            eye_dist = get_dist(eye_l, eye_r) + 1e-6 

            dist_to_ear = min(get_dist(wrist_l, ear_l), get_dist(wrist_r, ear_r))
            is_raised = min(wrist_l[1], wrist_r[1]) < eye_l[1] 
            
            if is_raised:
                scores["AHA"] = min(100, int(max(0, 100 - (dist_to_ear / eye_dist) * 60)))

            dist_to_nose = min(get_dist(wrist_l, nose), get_dist(wrist_r, nose))
            scores["THINKING"] = min(100, int(max(0, 100 - (dist_to_nose / eye_dist) * 80)))

            if scores["AHA"] < 40 and scores["THINKING"] < 40:
                mouth_y, mouth_x = int(nose[1] + eye_dist*0.4), int(nose[0])
                roi_size = int(eye_dist * 0.8)
                if 0 < mouth_y < h-roi_size and 0 < mouth_x < w-roi_size:
                    roi = frame[mouth_y:mouth_y+roi_size, mouth_x-int(roi_size/2):mouth_x+int(roi_size/2)]
                    if roi.size > 0:
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        _, thresh = cv2.threshold(gray_roi, 40, 255, cv2.THRESH_BINARY_INV)
                        dark_pixels = cv2.countNonZero(thresh)
                        scores["SCREAM"] = min(100, int((dark_pixels / (roi.size/3)) * 600))
            else:
                scores["SCREAM"] = 0 

            top_state = max(scores, key=scores.get)
            if scores[top_state] > 45: 
                status = top_state
                if status == "AHA": current_monkey = img_aha
                elif status == "THINKING": current_monkey = img_thinking
                elif status == "SCREAM": current_monkey = img_scream
            else:
                scores["DEFAULT"] = 100
                status = "DEFAULT"

    input_view = frame.copy()
    if results[0].keypoints is not None:
        cv2.circle(input_view, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), -1) 
        cv2.circle(input_view, (int(ear_l[0]), int(ear_l[1])), 5, (255, 0, 0), -1) 
        for wrist in [wrist_l, wrist_r]:
            if wrist[0] > 0: cv2.circle(input_view, (int(wrist[0]), int(wrist[1])), 7, (0, 255, 0), -1)

    output_view = cv2.resize(current_monkey, (w, h))
    combined_frame = np.hstack((input_view, output_view))

    y_off = 100
    for state, score in scores.items():
        color = (0, 255, 0) if state == status else (200, 200, 200)
        cv2.putText(combined_frame, f"{state}: {score}%", (w + 20, y_off), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_off += 40

    cv2.imshow("Monkey Mirror", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()