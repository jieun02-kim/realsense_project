# delivery_detection.py

import cv2
import numpy as np
import pyrealsense2 as rs
import signal, time
from ultralytics import YOLO
import math
import datetime
import json


import torch
from torchvision import models, transforms
import cv2, numpy as np

# 추가
#from groundingdino.util.inference import load_model, predict
import torchvision.transforms as T
import patient_info as info

# FPS 측정용
from collections import deque
import time
fps_history = deque(maxlen=10)


# 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# 모델 불러오기
model = models.mobilenet_v2(pretrained=False) 
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
model.load_state_dict(torch.load("gown_classifier.pth", map_location="cpu"))
model.eval()




# ====== 설정 ======
WIN = "Patient-Gown + Marker (ESC/q)"
YOLO_WEIGHTS = "yolov8n.pt"     # n/s/m 로 교체 가능 yolov8s.pt(Small), yolov8m.pt(Medium)
YOLO_DOOR_WEIGHTS = "door_yolov8n.pt"       # 문 탐지 (네가 학습한 pt 파일)
PERSON_CONF = 0.5       # 최소 신뢰도(confidence) 임계값.
DOOR_CONF = 0.5
COLOR_RES = (640, 480)  # (1280, 720) (640, 480)
FPS = 60
USE_DEPTH = True        # 거리 추정 원하면 True
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
# ==================

stop = False
def _sigint(sig, frame):
    global stop; stop = True
signal.signal(signal.SIGINT, _sigint)

# ---------- 환자복 판별 ----------MobileNetV2
def is_patient_gown(crop_bgr: np.ndarray) -> bool:
    if crop_bgr.size == 0:
        return False
    img = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224,224))
    tensor = transform(img).unsqueeze(0)  # (1,3,224,224)
    img_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    x = transform(img_rgb).unsqueeze(0)  # transform 안에서 ToPILImage()가 PIL 변환+Resize 수행

    
    with torch.no_grad():
        out = model(tensor)
        pred = torch.argmax(out, 1).item()
    return pred == 0  # 0=gown, 1=normal (ImageFolder 순서 기준) 


# ---------- QR/바코드 + QR(OpenCV) ----------
try:
    from pyzbar import pyzbar
    HAVE_PYZBAR = True
except Exception:
    HAVE_PYZBAR = False
qrd = cv2.QRCodeDetector()

def decode_markers(bgr):
    """
    ROI에서 QR/바코드/ArUco 탐지.
    반환: dict { 'qr':[(txt, pts)], 'aruco':[(id, corners)] }
    pts/corners 는 ROI 좌표계 기준 np.int32
    """
    out = {'qr': [], 'aruco': []}


    # 3) ArUco
    try:
        detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)
        corners, ids, _ = detector.detectMarkers(bgr)
        if ids is not None:
            for i, cid in enumerate(ids.flatten()):
                cs = corners[i].astype(int).reshape(-1, 2)
                out['aruco'].append((int(cid), cs))
    except Exception:
        pass

    return out

# ---------- 유틸 ----------
def clamp_box(x1,y1,x2,y2, W,H):
    return max(0,x1), max(0,y1), min(W,x2), min(H,y2)

def draw_poly(img, pts, color=(0,255,0), thickness=2):
    if pts is not None and len(pts) >= 4:
        cv2.polylines(img, [pts], True, color, thickness)


# #============각도 계산, x, y 거리 계산==========

# def calculate_theta(cxy_x : int) -> str:
#     frame_width = 640 # 640중 중간 픽셀
#     fov_deg = 69.4
#     fov_rad = math.radians(fov_deg)

#     # 중심 대비 상대 위치 비율 (-1.0 ~ +1.0)
#     rel = (cxy_x-frame_width/2) / (frame_width / 2)

#     # 좌우 각도 (radian)
#     theta_rad = rel * (fov_rad / 2)
#     return theta_rad
#     # 문자열로 변환해서 반환


# def calculate_vector(cxy_x, real_dist):
#     theta_rad = calculate_theta(cxy_x)
#     # real_dist의 x, y성분
#     dx = math.cos(theta_rad)*real_dist
#     dy = math.sin(theta_rad)*real_dist
#     # 1m의 x, y성분
#     dx_1m = math.cos(theta_rad)*1.00
#     dy_1m = math.sin(theta_rad)*1.00
#     return dx, dy, dx_1m, dy_1m

# def Artificial_Potention_Field(cxy_x, real_dist, k_att=1.0, stop_dist=1.0):
#     # 근데 이거 일단 attractive force만, 아직 replusive 는 구현 안함
#     dx, dy, dx_1m, dy_1m = calculate_vector(cxy_x, real_dist)
#     dist = math.hypot(dx, dy)
#     delta = dist - stop_dist
#     # Attractive force 계산
#     if delta <= 0:
#         return 0.0, 0.0  # 일정 거리 이내면 멈춤
    
#     apf_dist = k_att*delta
#     theta = calculate_theta(cxy_x)
#     apf_delta = k_att*theta

#     return apf_dist, apf_delta


# def get_apf_inputs(cxy_x, real_dist_m):
#     # cxy_x, real_dist_m은 현재 계산된 값으로 대체 가능
#     # 일단 테스트용으로 더미 값 리턴
#     return cxy_x, real_dist_m




def main():
    global stop

    # YOLO 로드 + 워밍업
    person_model = YOLO(YOLO_WEIGHTS)
    door_model = YOLO(YOLO_DOOR_WEIGHTS)
    _ = person_model.predict(np.zeros((480,640,3), dtype=np.uint8), verbose=False)
    _ = door_model.predict(np.zeros((480,640,3), dtype=np.uint8), verbose=False)



    # RealSense 파이프라인
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, FPS)
    if USE_DEPTH:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
        align = rs.align(rs.stream.color)
    profile = pipe.start(cfg)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)


    # post+mid
    post_mid = 0
    post_door = 0

    try:
        while not stop:
            # FPS 시작 시각
            start_time = time.time()

            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            frames = pipe.wait_for_frames()
            if USE_DEPTH:
                frames = align.process(frames)

            color_f = frames.get_color_frame()
            if not color_f:
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
                continue
            img = np.asanyarray(color_f.get_data())
            H, W = img.shape[:2]

            depth_f = frames.get_depth_frame() if USE_DEPTH else None

            # 1) 사람 탐지
            res = model.predict(img, classes=[0], conf=PERSON_CONF, verbose=False)
            
            for r in res:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    x1,y1,x2,y2 = clamp_box(x1,y1,x2,y2, W,H)
                    cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 2)

                # 2) 상체 ROI (환자복 판별용)
                ph = y2 - y1
                torso_y2 = y1 + int(ph*0.65)
                tx1, ty1, tx2, ty2 = x1, y1, x2, max(y1+1, torso_y2)
                torso = img[ty1:ty2, tx1:tx2]
                gown = is_patient_gown(torso)

                label = f"person ({'gown' if gown else 'clothes'})"
                cv2.putText(img, label, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                            (0,255,0) if gown else (0,165,255), 2)

                if not gown:
                    continue  # 환자복 아니면 넘어감 (정책상 필요 시 제거)

                # 3) 환자복으로 판정된 경우: 사람 ROI 전체에서 마커 탐지
                roi = img[y1:y2, x1:x2]
                # 작은 ROI는 업샘플
                scale_up = 1.0
                if max(roi.shape[:2]) < 420:
                    roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
                    scale_up = 1.5

                det = decode_markers(roi)

             
                # 3-2) ArUco 결과 
                for mid, corners in det['aruco']:
                    pts = (corners/scale_up).astype(int) + np.array([x1,y1])
                    draw_poly(img, pts, (0,255,255), 2)
                    # 중심
                    cxy = pts.mean(axis=0).astype(int)
                    dist_str = ""
                    if USE_DEPTH and depth_f:
                        d = depth_f.get_distance(int(cxy[0]), int(cxy[1]))
                        if d > 0: dist_str = f" | {d:.2f}m"
                    cv2.putText(img, f"ArUco : {mid}{dist_str}", (pts[0,0]+30, pts[0,1]-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
                    
                    cv2.circle(img, tuple(cxy), 4, (0,255,255), -1)


               
                    # myname
                    patient_data = info.get_patient_info(str(mid))

                    if not patient_data or not isinstance(patient_data, dict):
                        myname = "Unknown"
                    else:
                        myname = patient_data.get("final_name", "Unknown")
                    cv2.putText(img, f"Name : {myname}", (pts[0,0]+30, pts[0,1]-26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)


        

                if post_mid != mid:
                    crop = img[y1:y2, x1:x2]

                    # 파일 이름 구성
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"patient_{mid}_{ts}"
                    jpg_path = f"detections/{base_name}.jpg"
                    json_path = f"detections/{base_name}.json"

                    # 이미지 저장
                    cv2.imwrite(jpg_path, crop)

                    # bbox JSON 저장
                    bbox_data = {
                        f"patient_{mid}": {
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "phrase": "",
                            "caption": ""
                        }
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(bbox_data, f, ensure_ascii=False, indent=4)

                    print(f"New patient detected (ID={mid}) → Saved {jpg_path}")

                    # post_mid 업데이트
                    post_mid = mid
                    

            # (2) 문 탐지
            res_door = door_model.predict(img, conf=DOOR_CONF, verbose=False)
            for r in res_door:
                for b in r.boxes:
                    x1_d, y1_d, x2_d, y2_d = map(int, b.xyxy[0].tolist())
                    cv2.rectangle(img, (x1_d,y1_d),(x2_d,y2_d), (0,255,255), 2)
                    label = "Door"
                    cv2.putText(img, label, (x1_d,y1_d-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                if post_door != 1:
                    #capture save
                    crop_d = img[y1_d:y2_d, x1_d:x2_d]

                    # 파일 이름 구성
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_name = f"door_{ts}"
                    jpg_path = f"detections/{base_name}.jpg"
                    json_path = f"detections/{base_name}.json"

                    # 이미지 저장
                    cv2.imwrite(jpg_path, crop_d)

                    # bbox JSON 저장
                    bbox_data = {
                        f"patient_{mid}": {
                            "bbox": [float(x1_d), float(y1_d), float(x2_d), float(y2_d)],
                            "phrase": "",
                            "caption": ""
                        }
                    }
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(bbox_data, f, ensure_ascii=False, indent=4)

                    print(f"New door detected → Saved {jpg_path}")
                    post_door = 1




            # FPS 계산 및 표시 (imshow 직전)
            end_time = time.time()
            frame_time = end_time - start_time
            if frame_time > 0:
                fps_history.append(1.0 / frame_time)
                avg_fps = sum(fps_history) / len(fps_history)
            else:
                avg_fps = 0.0

            cv2.putText(img, f"FPS: {avg_fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            
                        

            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

