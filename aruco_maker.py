import cv2
import numpy as np

if __name__ == "__main__":
    
    board_type=cv2.aruco.DICT_4X4_50;
    MARKER_SIZE = 400;
    id_info = 1;
    
    arucoDict = cv2.aruco.getPredefinedDictionary(board_type);
    aruco_matker_img = cv2.aruco.generateImageMarker(arucoDict , id_info , MARKER_SIZE);
    
    cv2.imshow("aruco_matker_img",aruco_matker_img);
    cv2.waitKey(0);