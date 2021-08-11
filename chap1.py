# opencv调用摄像头
import cv2

# cap = cv2.VideoCapture("Tokyo.mp4")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)
while True:
    success, img = cap.read()
    cv2.imshow("Video", img)
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
