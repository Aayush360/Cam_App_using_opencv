import cv2

videoCapture = cv2.VideoCapture('images/MyInputVid.avi')
# get the frame rate from the video
fps = videoCapture.get(cv2.CAP_PROP_FPS)
print(fps)

# get the frame size from the video
# size = (int(cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cv2.VideoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float
size=(width,height)
print(width,height)

fourcc = cv2.VideoWriter_fourcc('I','4','2','0')
# since our video format is .avi we are using uncompressed YUV endcoing
videoWriter = cv2.VideoWriter('MyOutputVid.avi',fourcc,fps,size)


success,frame = videoCapture.read()

while success: # read until there are no more frame

    videoWriter.write(frame)
    success,frame = videoCapture.read()