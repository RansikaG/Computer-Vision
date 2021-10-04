import cv2
import mediapipe as mp
import time


class poseDetector():

    def __init__(self, mode = False, complexity = 1, smooth=True,segmentation=False,smooth_segmentation=True,detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.segmentation=segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose= mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode,self.complexity,self.smooth,self.segmentation,self.smooth_segmentation,self.detectionCon,self.trackCon)

    def findPose(self,img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw==True:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def FindPosition(self,img, draw= True):
        lmList=[]
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id,cx,cy])
                if draw==True:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0))
        return lmList
def main():
    cap = cv2.VideoCapture('PoseVideos/3.mp4')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img=detector.findPose(img)
        lmList =detector.FindPosition(img,draw=False)
        if lmList:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 0, 0))
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()