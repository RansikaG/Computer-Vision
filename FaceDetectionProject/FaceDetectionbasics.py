import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture('Videos/3.mp4')
pTime=0
while True:
    success, img =cap.read()

    cTime=time.time()
    print(cTime,pTime)
    fps= 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}',(20,70),cv2.FONT_HERSHEY_SIMPLEX,3,(255,3,251),2)
    cv2.imshow("Image", img)

    cv2.waitKey(1)