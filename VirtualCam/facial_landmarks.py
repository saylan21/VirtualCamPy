import cv2
import mediapipe as mp
import math

# Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh()

# cap = cv2.VideoCapture(0)

#MOVE_THRESHOLD = 40 


class faceDetector():
    MOVE_THRESHOLD = 40 
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

        self.font = cv2.FONT_HERSHEY_SIMPLEX
  
        # org
        self.org = (50, 100)

        # fontScale
        self.fontScale = 2

        # Red color in BGR
        self.color = (0, 255, 255)

        # Line thickness of 2 px
        self.thickness = 2

    def headMovementControl(self,image):
        # Image
        #ret, image = self.cap.read()
        height, width, _ = image.shape
        #print("Height, width", height, width)
        #rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Facial landmarks
        result = self.face_mesh.process(image)
        try:
            direction = ''
            for facial_landmarks in result.multi_face_landmarks:
                # for i in range(0, 468):
                #     pt1 = facial_landmarks.landmark[i]
                #     x = int(pt1.x * width)
                #     y = int(pt1.y * height)
                #     cv2.circle(image, (x, y), 2, (100, 100, 0), -1)
                #     #cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))
                nose_points = facial_landmarks.landmark[0]
                nose_point_x = int(nose_points.x * width)
                nose_point_y = int(nose_points.y * height)
                right_side_point = facial_landmarks.landmark[234]
                right_point_x = int(right_side_point.x * width)
                right_point_y = int(right_side_point.y * height)
                left_side_point = facial_landmarks.landmark[454]
                left_point_x = int(left_side_point.x * width)
                left_point_y = int(left_side_point.y * height)
                dist_left = math.hypot(right_point_x - nose_point_x, right_point_y - nose_point_y)
                dist_right = math.hypot(left_point_x - nose_point_x, left_point_y - nose_point_y)
                if dist_left < self.MOVE_THRESHOLD:
                    print("RIGHT")
                    direction = 'RIGHT'
                    # image = cv2.putText(image,"RIGHT",self.org,self.font,
                    #             self.fontScale,self.color,self.thickness,cv2.LINE_AA,False)
                elif dist_right < self.MOVE_THRESHOLD:
                    print("LEFT")
                    direction = 'LEFT'
                    # image = cv2.putText(image,"LEFT",self.org,self.font,
                    #             self.fontScale,self.color,self.thickness)
                #print("DIST : ", dist)
        except:
            pass
        cv2.putText(image,direction,self.org,self.font,
                                self.fontScale,self.color,self.thickness,cv2.LINE_AA,False)
        return image
        # cv2.imshow("Image", image)
        # if cv2.waitKey(20) & 0xFF == ord('q'):
        #     break
            
        # self.cap.release()
        # cv2.destroyAllWindows()

def main():
    cap = cv2.VideoCapture(0)
    faceDec = faceDetector(cap,cap)
    faceDec.headMovementControl()


if __name__== "__main__":
    main()









