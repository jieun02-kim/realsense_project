import cv2

WIN = "Test Window"
img = cv2.imread("lenna.jpg")

cv2.imshow(WIN, img)     # 창에 이미지 띄우기
cv2.waitKey(0)           # 키 입력 기다리기 (0 → 무한대기)
cv2.destroyAllWindows()  # 창 닫기