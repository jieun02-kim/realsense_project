#color_test.py
#기본 realsense camera 동작

import pyrealsense2 as rs
import numpy as np
import cv2

# 파이프라인 초기화
pipeline = rs.pipeline()
config = rs.config()

# 컬러 스트림만 활성화
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 파이프라인 시작
pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 화면 출력
        cv2.imshow('RealSense Color', color_image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
