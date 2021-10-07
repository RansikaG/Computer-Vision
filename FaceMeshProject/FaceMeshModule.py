import cv2
import mediapipe as mp
import time

class FaceMeshdetector():

    def __init__(self,staticmode = False, maxFaces =5 ,mindetectonCon =0.5, minTrackcon =0.5):
        self.staticmode = staticmode
        self.maxFaces = maxFaces
        self.mindetectonCon =mindetectonCon
        self.minTrackcon = minTrackcon

        self.mpDraw =mp.solutions.drawing_utils
        self.mpFaceMesh =mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticmode,self.maxFaces,
                                                 self.mindetectonCon,self.minTrackcon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def findFaceMesh(self,img, draw= True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face= []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)

                    #print(id, x, y)
                    face.append([x,y])

                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture('Videos/2.mp4')
    pTime = 0
    detector = FaceMeshdetector()
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img,True)
        if faces!=[]:
            print(len(faces))
        cTime = time.time()
        if cTime != pTime:
            fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()