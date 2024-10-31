import cv2
import mediapipe as mp

camera = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands()
skeleton = mp.solutions.drawing_utils

while True:
    _, frame = camera.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_obj = hands.process(frameRGB)

    if hand_obj.multi_hand_landmarks:
        for hand_landmarks in hand_obj.multi_hand_landmarks:
            skeleton.draw_landmarks(frame, hand_landmarks,
                                 mp.solutions.hands.HAND_CONNECTIONS)
            
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()