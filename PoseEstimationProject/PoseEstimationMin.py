import cv2
import mediapipe as mp
import os
import time
os.getcwd()

cap = cv2.VideoCapture('PoseVideos/1.mp4')
pTime=0
while True:
    success, img =cap.read()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(70,50),cv2.FONT_HERSHEY_PLAIN, 3 ,(255,255,0),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)