import cv2

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-model.yml")
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ['None', 'Razan']
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
                                         minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 90:
            id = names[id]
        else:
            id = "unknown"
        confidence = "{}%".format(round(100-confidence))

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255, 0, 0), 1)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
