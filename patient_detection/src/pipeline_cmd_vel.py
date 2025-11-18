# pipeline_cmd_vel.py

import cv2
import numpy as np
import pyrealsense2 as rs
import signal, time
from ultralytics import YOLO
import math

import torch
from torchvision import models, transforms
import cv2, numpy as np

# 추가
#from groundingdino.util.inference import load_model, predict
import torchvision.transforms as T
import patient_info as info


# --- ROS2 추가 ---
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from rclpy.qos import QoSProfile, ReliabilityPolicy



# FPS 측정용
from collections import deque
import time
fps_history = deque(maxlen=10)


# ===============================================================
#                  ROS2 Node 정의
# ===============================================================
class CmdVelPublisher(Node):
    def __init__(self):
        super().__init__('gown_apf_publisher')
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', qos)

    def publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(max(min(linear, 0.5), -0.5))   # 제한
        msg.angular.z = float(max(min(angular, 1.0), -1.0))
        self.publisher_.publish(msg)
        self.get_logger().info(
            f"📤 /cmd_vel -> linear.x={msg.linear.x:.3f}, angular.z={msg.angular.z:.3f}"
        )




# 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),  
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


# 모델 불러오기
g_model = models.mobilenet_v2(pretrained=False) 
g_model.classifier[1] = torch.nn.Linear(g_model.last_channel, 2)
g_model.load_state_dict(torch.load("gown_classifier.pth", map_location="cpu"))
g_model.eval()




# ====== 설정 ======
WIN = "Patient-Gown + Marker (ESC/q)"
YOLO_WEIGHTS = "yolov8n.pt"     # n/s/m 로 교체 가능 yolov8s.pt(Small), yolov8m.pt(Medium)
PERSON_CONF = 0.5       # 최소 신뢰도(confidence) 임계값.
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
        out = g_model(tensor)
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

    # 1) QR/바코드: pyzbar 우선
    if HAVE_PYZBAR:
        try:
            for c in pyzbar.decode(bgr):
                txt = c.data.decode('utf-8', errors='ignore')
                pts = np.array(c.polygon, dtype=np.int32).reshape(-1, 2) if c.polygon else None
                out['qr'].append((txt, pts))
        except Exception:
            pass

    # 2) OpenCV QR 폴백/보강
    try:
        ok, texts, points, _ = qrd.detectAndDecodeMulti(bgr)
        if ok and texts is not None:
            for i, t in enumerate(texts):
                if t:
                    pts = points[i].astype(int) if points is not None else None
                    out['qr'].append((t, pts))
        else:
            t, pts, _ = qrd.detectAndDecode(bgr)
            if t:
                pts = pts.astype(int) if pts is not None else None
                out['qr'].append((t, pts))
    except Exception:
        pass

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


#============각도 계산, x, y 거리 계산==========

def calculate_theta(cxy_x : int) -> str:
    frame_width = 640 # 640중 중간 픽셀
    fov_deg = 69.4
    fov_rad = math.radians(fov_deg)

    # 중심 대비 상대 위치 비율 (-1.0 ~ +1.0)
    rel = (cxy_x-frame_width/2) / (frame_width / 2)

    # 좌우 각도 (radian)
    theta_rad = rel * (fov_rad / 2)
    return theta_rad
#  # 문자열로 변환해서 반환


def calculate_vector(cxy_x, real_dist):
    theta_rad = calculate_theta(cxy_x)
    # real_dist의 x, y성분
    dx = math.cos(theta_rad)*real_dist
    dy = math.sin(theta_rad)*real_dist
    # 1m의 x, y성분
    dx_1m = math.cos(theta_rad)*1.00
    dy_1m = math.sin(theta_rad)*1.00
    return dx, dy, dx_1m, dy_1m

def Artificial_Potention_Field(cxy_x, real_dist, k_att=3.0, stop_dist=1.0):
    # 근데 이거 일단 attractive force만, 아직 replusive 는 구현 안함
    dx, dy, dx_1m, dy_1m = calculate_vector(cxy_x, real_dist)
    dist = math.hypot(dx, dy)
    delta = dist - stop_dist
    # Attractive force 계산
    if delta <= 0:
        return 0.0, 0.0  # 일정 거리 이내면 멈춤
    
    apf_dist = k_att*delta
    theta = calculate_theta(cxy_x)
    apf_delta = k_att*theta

    apf_dist = k_att*delta
    theta = calculate_theta(cxy_x)
    theta_p = math.pow(abs(theta), 1.5)*(theta/abs(theta))
    apf_delta = k_att*theta_p
    return apf_dist, apf_delta


def get_apf_inputs(cxy_x, real_dist_m):
    return cxy_x, real_dist_m


def main():
    global stop

    rclpy.init()                        # --- ROS2 INIT ---
    node = CmdVelPublisher()            # --- ROS2 Node 생성 ---

    # YOLO 로드 + 워밍업
    model = YOLO(YOLO_WEIGHTS)
    _ = model.predict(np.zeros((480,640,3), dtype=np.uint8), verbose=False)

    # ==== GroundingDINO 로드 ====
    # GROUNDDINO_CONFIG = "../../../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    # GROUNDDINO_WEIGHTS = "../../../GroundingDINO/weights/groundingdino_swint_ogc.pth"
    # g_dino_model = load_model(GROUNDDINO_CONFIG, GROUNDDINO_WEIGHTS)
    # device = next(g_dino_model.parameters()).device


    # RealSense 파이프라인
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, FPS)
    if USE_DEPTH:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, FPS)
        align = rs.align(rs.stream.color)
    profile = pipe.start(cfg)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        while not stop:
            # 🔹 FPS 시작 시각
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
            # === GroundingDINO (대체) ===
            
            
            # # BGR → RGB → Tensor 변환
            # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # tensor_img = T.ToTensor()(rgb_img).to(device)



            # # 사람 탐지 수행
            # boxes, logits, phrases = predict(
            #     model=g_dino_model,
            #     image=tensor_img,
                
            #     caption="a person who is standing",
            #     box_threshold=0.35,
            #     text_threshold=0.30
            # )


            
            #print(f"[DEBUG] Detected boxes: {len(boxes)}")

            # for box in boxes:
            #     # GroundingDINO는 (cx, cy, w, h)일 수 있음 → 변환
            #     if len(box) == 4:
            #         cx, cy, w, h = box.tolist()
            #         x1 = (cx - w / 2)
            #         y1 = (cy - h / 2)
            #         x2 = (cx + w / 2)
            #         y2 = (cy + h / 2)
            #     else:
            #         x1, y1, x2, y2 = box.tolist()

                # # 정규화 좌표면 픽셀 단위로 변환
                # if 0 <= x2 <= 1 and 0 <= y2 <= 1:
                #     x1, x2 = x1 * W, x2 * W
                #     y1, y2 = y1 * H, y2 * H

                # # 정렬 및 클램프
                # x1, x2 = sorted([int(x1), int(x2)])
                # y1, y2 = sorted([int(y1), int(y2)])
                # x1, y1 = max(0, x1), max(0, y1)
                # x2, y2 = min(W, x2), min(H, y2)

                # if (x2 - x1) < 10 or (y2 - y1) < 10:
                #     continue

                # print(f"[DEBUG] Corrected box: ({x1},{y1})-({x2},{y2})")
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)





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

                    # # 3-1) QR 결과
                    # for txt, pts in det['qr']:
                    #     # 중심 계산
                    #     if pts is not None and len(pts) >= 4:
                    #         cxy = pts.mean(axis=0).astype(int)
                    #         # 원본 좌표로 보정
                    #         cp = (int(x1 + cxy[0]/scale_up), int(y1 + cxy[1]/scale_up))
                    #         cv2.circle(img, cp, 4, (0,255,0), -1)
                    #         draw_poly(img, (pts/scale_up).astype(int) + np.array([x1,y1]))
                    #     else:
                    #         # 폴리곤이 없으면 ROI 중앙
                    #         cp = (int((x1+x2)//2), int((y1+y2)//2))
                    #     # 거리 추정
                    #     dist_str = ""
                    #     if USE_DEPTH and depth_f:
                    #         d = depth_f.get_distance(cp[0], cp[1])
                    #         if d > 0:
                    #             dist_str = f" | {d:.2f}m"
                    #     cv2.putText(img, f"QR:{txt}{dist_str}", (x1+5, max(20,y1-10)),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

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


                        #is it ok???????????
                        depth = d * 100
                        real_dist = info.calculate_range(str(mid), depth)
                            # 안전 포맷 처리
                        if real_dist is None:
                            dist_text = "REAL_DISTANCE = N/A"
                        else:
                            dist_text = f"REAL_DISTANCE = {real_dist:.2f} cm"

                        cv2.putText(img, dist_text, (pts[0,0]+30, pts[0,1]+20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                        
                        # 각도 계산 printinf
                        theta = calculate_theta(cxy[0]) 
                        theta_s = f"{theta:.2f} rad"  # 문자열로 변환
                        cv2.putText(img, theta_s, (pts[0,0]+30, pts[0,1]+40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                                            
    

                        # --- 인공퍼텐셜필드(APF) 계산 결과 표시 ---
                        if real_dist is not None:
                            real_dist_m = real_dist / 100.0  # cm → m 변환
                            v, theta_rad = Artificial_Potention_Field(cxy[0], real_dist_m)
                            
                            # 🟢 ROS 퍼블리시 추가
                            node.publish_cmd(v, theta_rad)
                            # 속도와 각도 결과 문자열 생성
                            apf_text = f"APF -> v: {v:.2f}, theta: {math.degrees(theta_rad):.2f}°"

                            # 화면 표시
                            cv2.putText(img, apf_text, (pts[0,0]+30, pts[0,1]+60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)


                        
                        # myname
                        patient_data = info.get_patient_info(str(mid))

                        if not patient_data or not isinstance(patient_data, dict):
                            myname = "Unknown"
                        else:
                            myname = patient_data.get("final_name", "Unknown")
                        cv2.putText(img, f"Name : {myname}", (pts[0,0]+30, pts[0,1]-26),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

                                
            # 🔹 FPS 계산 및 표시 (imshow 직전)
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
            rclpy.spin_once(node, timeout_sec=0)  # ROS 이벤트 처리

    finally:
        pipe.stop()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

