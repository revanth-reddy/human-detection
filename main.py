import numpy as np
import cv2

# in windows mention the classifier location in between " " and in linux download opencv file and place the file location
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
face2_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
face3_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
full_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_fullbody.xml')
lower_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_lowerbody.xml')
upper_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_upperbody.xml')





cap = cv2.VideoCapture('vtest.avi')
i=0
while(1):
    _, img =cap.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ##face1
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##face2
    faces = face2_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##face3
    faces = face3_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##full
    faces = full_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    ##lower
    faces = lower_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##upper
    faces = upper_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    

    cv2.imshow('img (Esc to exit)',img)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()