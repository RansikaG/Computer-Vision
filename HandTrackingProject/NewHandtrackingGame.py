import HandTrackingModule as htm
import cv2
import time

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector =htm.handDetector()
while True:
    success, img = cap.read()  # gives the image
    img =detector.findHands(img,draw=False)
    lmList =detector.findPosition(img,draw=False)
    if lmList:
        print(lmList[4])

    #print(results.multi_hand_landmarks)
    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)),(10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255),3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)