import cv2
import matplotlib.pyplot as plt

face = cv2.CascadeClassifier('/home/rahul/Programming/Github-Content/Chrome-Dinosaur-Pythonified/Face-detection/cascade-files/haarcascade_frontalface_alt.xml')
eyes = cv2.CascadeClassifier('/home/rahul/Programming/Github-Content/Chrome-Dinosaur-Pythonified/Face-detection/cascade-files/parojos.xml')

cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_face = face.detectMultiScale(gray, 1.1, 5)
    detected_eyes = eyes.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in detected_face:     
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    for (x1, y1, x2, y2) in detected_eyes:
        cv2.circle(frame, (x1,y1), 7, 2)
        cv2.circle(frame, (x2,y2), 7, 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''
img = cv2.imread('/home/rahul/Pictures/Webcam/2018-10-21-170626_4.jpg')
plt.imshow(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
det = eyes.detectMultiScale(gray, 1.1, 5)
cv2.circle(img, (343,99), 7, 3)
cv2.circle(img, (127, 29), 7, 3)
cv2.imshow('img', img)
print(det)
'''