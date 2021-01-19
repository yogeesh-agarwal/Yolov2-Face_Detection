import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
index = 1
while(cap.isOpened()):
    stime= time.time()
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame" , frame)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
