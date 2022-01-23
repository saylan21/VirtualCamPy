import cv2
import numpy as np
import pyvirtualcam


src=cv2.VideoCapture(0)
w = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
fmt = pyvirtualcam.PixelFormat.BGR
ret, frame=src.read()

font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (100, 300)
  
# fontScale
fontScale = 3
   
# Blue color in BGR
color = (255, 0, 0)

thickness = 3

with pyvirtualcam.Camera(width=w, height=h, fps=20, fmt=fmt) as vcam:
    print(f'Using virtual camera: {vcam.device} at {vcam.width} x {vcam.height}')

    while True:
        try:
            if ret:
                frame = cv2.putText(frame, 'VirtualCam', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
                frame = cv2.flip(frame, 1)
                vcam.send(frame)
                #vcam.sleep_until_next_frame()
                ret,frame = src.read()
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            break

src.release()
