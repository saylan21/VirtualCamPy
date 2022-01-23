import cv2
import numpy as np
import pyvirtualcam
from face import *
from facial_landmarks import *
from handTracker import *


def main():
    faceDec = faceDetector()
    faceRec = faceRecognizer()

    handTrac = HandTracker()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            # here statemachine
            img = faceRec.recognizeFace(frame) 
            #face_img = faceDec.headMovementControl(img_rgb)
            #img_hand_rgb = handTrac.trackHands(frame)
            #img_hand_bgr = cv2.cvtColor(img_hand_rgb, cv2.COLOR_BGR2RGB)
            #cv2.imshow("Frame", img_hand_bgr) # going to be virtual cam
            #img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            cv2.imshow("Frame", img)
        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()