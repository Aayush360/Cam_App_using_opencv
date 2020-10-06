''' program that captures 10 seconds of video frame and writes to avi file'''

import cv2

cameraCapture = cv2.VideoCapture(0)
fps = 30 # assumption
size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fourcc = cv2.VideoWriter.fourcc('I','4','2','0')
videoWriter = cv2.VideoWriter('MyOutputVid.avi',fourcc,fps,size)
success,frame = cameraCapture.read()

numFrameRemaining = 10*fps-1

while success and numFrameRemaining>0:
    videoWriter.write(frame)
    success,frame = cameraCapture.read()
    numFrameRemaining -=1

cv2.waitKey(0)
cv2.destroyAllWindows()