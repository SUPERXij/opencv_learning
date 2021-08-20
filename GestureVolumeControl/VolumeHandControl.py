import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 摄像窗口设置
wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
# 调用手部检测函数
detector = htm.handDetector(maxHands=1, detectionCon=0.7)
# 音量控制函数
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
# 参数
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400
volPer = 0
volColor = 0
pTime = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHand(img)
    lmList = detector.findPosition(img, draw=False)
    # 获取食指与大拇指位置
    if len(lmList) != 0:
        # print(lmList[4], lmList[8])

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = int(math.hypot(x2 - x1, y2 - y1))
        # print(length)

        # 长度范围 50 - 250
        # 音量范围 -65 - 0
        # 范围转换
        vol = np.interp(length, [50, 250], [minVol, maxVol])
        # 拟合得出的音量函数
        volBar = int(118*math.exp(0.08*vol))
        if volBar > 100:
            volBar = 100
        # print(volBar)

        # volBar = np.interp(length, [50, 250], [400, 150])
        # volPer = np.interp(length, [50, 250], [0, 100])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        volPer = volBar
        volBar = np.interp(volBar, [0, 100], [400, 150])
        volColor = np.interp(volPer, [0, 100], [0, 255])
        # print(volColor)

        if length < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
    # 音量图
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
    # 根据音量变换颜色
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (50, 50, int(volColor)), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                1, (50, 50, int(volColor)), 3)

    # FPS计算
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                1, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
