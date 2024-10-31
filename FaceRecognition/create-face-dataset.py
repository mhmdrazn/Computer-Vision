import cv2
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

person_id = 1
count = 0
while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, 
                                         minNeighbors=6, minSize=(50, 50))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), 
                      (255, 0, 0), 2)
        count += 1
        cv2.imwrite(dataset_path + "Person-" + str(person_id)
                    + "-" + str(count) + ".jpg", gray[y:y+h, x:x+w]) 
    
    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break
    elif count == 100:
        break

camera.release()
cv2.destroyAllWindows()

