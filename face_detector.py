import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


#img = cv2.imread('images/ss.png')
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read , frame = webcam.read()

    greyscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_cordinates = trained_face_data.detectMultiScale(greyscaled_img)

    print(face_cordinates)

    for (x,y,w,h) in face_cordinates:

        cv2.rectangle(frame,(x ,y),(x+w,y+h), (0,255,0),2)


    cv2.imshow('face detector', frame)
    key = cv2.waitKey(1)

    if key == 32:
        break

webcam.release()