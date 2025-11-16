import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

detector = htm.handDetector(detectionCon=0.7, maxHands=4)

class handDetector:
    def __init__(self, mode=False, maxHands=4, detectionCon=0.7, trackCon=0.7):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPositions(self, img):
        allHands = []
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                allHands.append(lmList)
        return allHands


cap = cv2.VideoCapture(0)
detector = handDetector(detectionCon=0.7, maxHands=4)

tipIds = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    allHands = detector.findPositions(img)

    totalFingers = 0  # global counter for all hands

    for lmList in allHands:
        fingers = []

        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalFingers += fingers.count(1)  # add to global counter

    # Show finger count on screen
    cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(totalFingers), (45, 375),
                cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cv2.imshow("Finger Counter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
