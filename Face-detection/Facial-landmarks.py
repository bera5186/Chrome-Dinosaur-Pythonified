import cv2
import dlib
import utility
import imutils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/home/rahul/Programming/Github-Content/Chrome-Dinosaur-Pythonified/Face-detection/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)

while True:
    _, orig = cap.read()
    _, frame = cap.read()
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray,ratio = utility.resize(gray, width=120)

    rects = detector(gray,1)
    for (i,rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = utility.shape_to_np(shape)

        (x,y,w,h) = utility.rect_to_bb(rect)
        #cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
        cv2.rectangle(frame, (int(x/ratio), int(y/ratio)),(int(w/ratio), int(h/ratio)), (255, 0, 0), 5)

        
        for (x,y) in shape:
            cv2.circle(frame, (int(x/ratio),int(y/ratio)),2,(0,255,0), 2)
    cv2.imshow('frame', frame)
    cv2.imshow('original', orig)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()