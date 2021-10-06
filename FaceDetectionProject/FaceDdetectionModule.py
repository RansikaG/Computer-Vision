import cv2
import mediapipe as mp
import time

class Facedetector():
    def __init__(self,minDetectionCon=0.5):
        self.minDetectionCon= minDetectionCon
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionCon)

    def findfaces(self,img,draw=True):

        imgRGB =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxs=[]
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                #mpDraw.draw_detection(img,detection)
                #print(id,detection)
                ih, iw ,ic =img.shape
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih), \
                       int(bboxC.width * iw),int(bboxC.height * ih)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img = self.fancydraw(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        return bboxs,img

    def fancydraw(selfself,img,bbox,l=30,t=5,rt=2):
        x,y,w,h =bbox
        x1, y1 = x+w, y+h

        cv2.rectangle(img, bbox, (0, 255, 0), rt)
        #top left
        cv2.line(img,(x,y),(x+l,y),(0,255,0),t)
        cv2.line(img, (x, y), (x , y+l), (0, 255, 0), t)

        #bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (0, 255, 0), t)
        cv2.line(img, (x1, y1), (x1 , y1-l), (0, 255, 0), t)

        # top right
        cv2.line(img, (x1, y), (x1 - l, y), (0, 255, 0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0, 255, 0), t)

        #bottom left
        cv2.line(img, (x, y1), (x + l, y1), (0, 255, 0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0, 255, 0), t)
        return img
def main():
    cap = cv2.VideoCapture('Videos/2.mp4')
    pTime = 0
    fps = 0
    detector =Facedetector()
    while True:
        success, img = cap.read()
        bboxes,img = detector.findfaces(img)
        print(bboxes)
        cTime = time.time()
        if cTime - pTime != 0:
            fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{str(int(fps))}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 3, 251), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__== "__main__":
    main()