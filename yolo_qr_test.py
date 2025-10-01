# yolo_barcode_anywhere_in_person.py
import cv2, numpy as np, pyrealsense2 as rs, signal
from ultralytics import YOLO
from pyzbar import pyzbar

WIN = "Person ROI → Barcode/QR  (ESC/q to quit)"
stop = False
def _sigint(sig, frame):
    global stop; stop = True
signal.signal(signal.SIGINT, _sigint)

qrd = cv2.QRCodeDetector()

def decode_all(img_bgr):
    """pyzbar(다중 심볼) 우선 → 실패 시 OpenCV QR 폴백"""
    out = []
    # 1) pyzbar
    for c in pyzbar.decode(img_bgr):
        txt = c.data.decode('utf-8', errors='ignore')
        pts = np.array(c.polygon, dtype=np.int32).reshape(-1,2) if c.polygon else None
        sym = c.type
        out.append((txt, pts, sym))
    if out: return out

    # 2) OpenCV QR (multi → single 폴백)
    try:
        ok, texts, points, _ = qrd.detectAndDecodeMulti(img_bgr)
        if ok and texts is not None:
            for i, t in enumerate(texts):
                if t:
                    pts = points[i].astype(int) if points is not None else None
                    out.append((t, pts, "QR"))
        if out: return out
        t, pts, _ = qrd.detectAndDecode(img_bgr)
        if t:
            pts = pts.astype(int) if pts is not None else None
            out.append((t, pts, "QR"))
    except Exception:
        pass
    return out

def clamp(x1,y1,x2,y2,w,h):
    return max(0,x1), max(0,y1), min(w,x2), min(h,y2)

def main():
    global stop
    # YOLO(person) 로드
    model = YOLO("yolov8n.pt")
    _ = model.predict(np.zeros((480,640,3),dtype=np.uint8), verbose=False)  # 워밍업

    # RealSense 컬러 (해상도 넉넉히)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipe.start(cfg)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    try:
        while not stop:
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break

            f = pipe.wait_for_frames()
            c = f.get_color_frame()
            if not c:
                if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break
                continue

            img = np.asanyarray(c.get_data())
            H, W = img.shape[:2]

            # 사람 탐지
            res = model.predict(img, classes=[0], conf=0.5, verbose=False)
            for r in res:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    x1,y1,x2,y2 = clamp(x1,y1,x2,y2,W,H)
                    cv2.rectangle(img, (x1,y1),(x2,y2), (255,0,0), 2)

                    # ★ 사람 ROI 전체에서 바코드/QR 탐색
                    roi = img[y1:y2, x1:x2]
                    # ROI가 작으면 디코더 유리하게 1.5배 업샘플
                    if max(roi.shape[:2]) < 400:
                        roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)

                    found = decode_all(roi)

                    # 결과 그리기 (좌표 원본 기준으로 보정)
                    for txt, pts, sym in found:
                        cv2.putText(img, f"{sym}: {txt}",
                                    (x1+5, max(20, y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)
                        if pts is not None:
                            # 업샘플했으면 좌표 축소 필요
                            scale = 1.0
                            # 업샘플 시 scale=1.5였으니 되돌리기
                            if max((y2-y1),(x2-x1)) < 400:
                                scale = 1/1.5
                            pts = (pts * scale).astype(int)
                            pts[:,0] += x1
                            pts[:,1] += y1
                            cv2.polylines(img, [pts], True, (0,255,0), 2)

            cv2.imshow(WIN, img)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
