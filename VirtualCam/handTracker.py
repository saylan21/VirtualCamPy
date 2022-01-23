import cv2
import mediapipe as mp
import time

class HandTracker():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands( max_num_hands=1)
        self.mpDraw = mp.solutions.drawing_utils

    def trackHands(self,rgb_img):
        ct = time.time()
        results = self.hands.process(rgb_img)
        end_time = time.time()
        h, w, c = rgb_img.shape
        #print("PROCESS TIME :", end_time - ct)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                #self.mpDraw.draw_landmarks(rgb_img, handLms)
                # for id, lm in enumerate(handLms.landmark):
                #     h, w, c = rgb_img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                point_finger = handLms.landmark[8]
                cx, cy = int(point_finger.x*w), int(point_finger.y*h)
                cv2.circle(rgb_img,(cx, cy), 2, (100, 100, 0), 5)
                middle_finger = handLms.landmark[12]
                cx, cy = int(middle_finger.x*w), int(middle_finger.y*h)
                cv2.circle(rgb_img,(cx, cy), 2, (100, 100, 0), 5)
        return rgb_img

def main():
    handTrac = HandTracker()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret:
            img = handTrac.trackHands(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow("hands", img)
        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()