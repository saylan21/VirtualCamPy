import cv2
from cv2 import resize
import numpy as np
import pyvirtualcam
from face import *
from facial_landmarks import *
from handTracker import *
import pyvirtualcam
from constants import *

def resizeImg(image):
    scale_percent = 60 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    return resized

def drawMenu(Flags,image):
    #Landmark
    cv2.rectangle(image, (faceMashX,faceMashY), (faceMashX+recWidth,faceMashY+recHeight), (200,0,0),-1)
    cv2.putText(image, "Active Liveness Detection", (faceMashX,faceMashY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,250), 2,cv2.LINE_AA, False)

    #Recognizer
    cv2.rectangle(image, (headRecX,headRecY), (headRecX+recWidth,headRecY+recHeight), (50,100,50),-1)
    cv2.putText(image, "Face Detection", (headRecX,headRecY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,250), 2,cv2.LINE_AA, False)

    #Exit
    if Flags[2] == 1:
        cv2.rectangle(image, (ExitX,ExitY), (ExitX+recWidth,ExitY+recHeight), (0,0,255),-1)
        cv2.putText(image, "Exit", (ExitX,ExitY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,250), 2,cv2.LINE_AA, False)
    else:
        cv2.rectangle(image, (ExitX,ExitY), (ExitX+recWidth,ExitY+recHeight), (0,255,0),-1)
        cv2.putText(image, "Start", (ExitX,ExitY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,100,250), 2,cv2.LINE_AA, False)
    return image

def drawFingerPoints(image,points):
    for point in points:
        cv2.circle(image,(point[0], point[1]), 2, (255, 255, 0), 5)
    #cv2.imshow("fingers",image)
    return image
def MenuController(Flags,finger_points):
    liveness_box = [faceMashX,faceMashY,faceMashX+recWidth,faceMashY+recHeight]
    recognizer_box = [headRecX,headRecY,headRecX+recWidth,headRecY+recHeight]
    exit_box = [ExitX,ExitY,ExitX+recWidth,ExitY+recHeight]

    if liveness_box[0]<finger_points[0][0]<liveness_box[2] and liveness_box[1]<finger_points[0][1]<liveness_box[3]:
        if liveness_box[0]<finger_points[1][0]<liveness_box[2] and liveness_box[1]<finger_points[1][1]<liveness_box[3]:
            print("liveness Detection")
            Flags[0] = 1
    if recognizer_box[0]<finger_points[0][0]<recognizer_box[2] and recognizer_box[1]<finger_points[0][1]<recognizer_box[3]:
        if recognizer_box[0]<finger_points[1][0]<recognizer_box[2] and recognizer_box[1]<finger_points[1][1]<recognizer_box[3]:
            print("Face Recognizer")
            Flags[1] = 1
    if exit_box[0]<finger_points[0][0]<exit_box[2] and exit_box[1]<finger_points[0][1]<exit_box[3]:
        if exit_box[0]<finger_points[1][0]<exit_box[2] and exit_box[1]<finger_points[1][1]<exit_box[3]:
            print("Exit/Start")
            Flags[2] ^= 1
    return Flags

    

def main():
    fmt = pyvirtualcam.PixelFormat.BGR
    Flags = [0,0,1] # Landmarks, Recognizer, Exit

    faceDec = faceDetector()
    faceRec = faceRecognizer()

    handTrac = HandTracker()
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    with pyvirtualcam.Camera(width=w, height=h, fps=20, fmt=fmt) as vcam:
        print(f'Using virtual camera: {vcam.device} at {vcam.width} x {vcam.height}')
        while True:
            ret, frame = cap.read()
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if ret:
                if Flags[2] == 1:
                    # here statemachine
                    if Flags[1]:
                        img_rgb = faceRec.recognizeFace(frame) 
                    if Flags[0]:
                        img_rgb = faceDec.headMovementControl(img_rgb)
                    #img_hand_rgb = handTrac.trackHands(frame)
                finger_points = handTrac.trackHands(img_rgb)
                #img_hand_bgr = cv2.cvtColor(img_hand_rgb, cv2.COLOR_BGR2RGB)
                #cv2.imshow("Frame", img_hand_bgr) # going to be virtual cam
                frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

                frame = drawMenu(Flags,frame)
                if finger_points:
                    frame = drawFingerPoints(frame,finger_points)
                    Flags = MenuController(Flags, finger_points)
                #cv2.imshow("Frame", frame)
                vcam.send(cv2.flip(frame, 1))


            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()