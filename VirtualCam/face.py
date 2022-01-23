import cv2
import numpy as np

class faceRecognizer():
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
        self.color = (0, 255, 0) #BGR
        self.stroke = 2
    def recognizeFace(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            faces = self.face_cascade.detectMultiScale(gray)
            for x, y, w, h in faces:
                roi_color = frame[y:y+h, x:w:h]
                end_cord_x = x + w
                end_cord_y = y + h
                cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y),self.color, self.stroke)
        except:
            pass
        return frame
# face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')

# cap = cv2.VideoCapture(0)

# color = (0, 255, 0) #BGR
# stroke = 2

# while True:
#     # ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray)

#     for x, y, w, h in faces:
#         roi_color = frame[y:y+h, x:w:h]
#         end_cord_x = x + w
#         end_cord_y = y + h
#         cv2.rectangle(frame,(x,y),(end_cord_x, end_cord_y),color, stroke)


#     cv2.imshow('frame',frame)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()