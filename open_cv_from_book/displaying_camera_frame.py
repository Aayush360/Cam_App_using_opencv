'''program to display camera frame in a window'''

import cv2
clicked = False

def onMouse(event,x,y,flags,params):
    global clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        clicked=True


cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('MyWindow')
cv2.setMouseCallback('MyWindow',onMouse)

print('Showing camera feed. Click Window or press any key to stop')

success,frame = cameraCapture.read()

# waitKey - number of millisecond to wait for keyboard input

while success and cv2.waitKey(1)==-1 and not clicked:
    cv2.imshow('MyWindow',frame)
    success,frame = cameraCapture.read()

cv2.destroyWindow('MyWindow')
