import cv2
import numpy as np
import time
import pygame
from open_cv_from_book import utils
import sys


class CaptureManager(object):

    def __init__(self, capture, previewWindowManager=None, shouldMirrorPreview=False):
        self.previewWindowManager=previewWindowManager
        self.shouldMirrorPreview = shouldMirrorPreview
        self._capture = capture
        self._channel = 0
        self._enteredFrame = False
        self._frame = None
        self._imageFilename = None
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None
        self._startTime = None
        self._framesElapsed = int(0)
        self._fpsEstimate = None

    # setting a getter using intermediery
    @property
    def channel(self):
        return self._channel

    @channel.setter
    def channel(self,value):
        if self._channel !=value:
            self._channel=value
            self._frame=None

    @property
    def frame(self):
        if self._enteredFrame and self._frame is None:
            # instead of read we use grab(tells sucess or faliure) and retrieve(gives frame)
            _,self._frame=self._capture.retrieve()
        return self._frame
    @property
    def isWritingImage(self):
        return self._imageFilename is not None
    @property
    def isWritingVideo(self):
        return self._videoFilename is not None


    def enterFrame(self):
        """capture the next frame if any"""
        # but first check that any previous frame was exited
        assert not self._enteredFrame, \
            'previous enterFrame() has no matching exitFrame()'
        if self._capture is not None:
            self._enteredFrame = self._capture.grab()

    def exitFrame(self):
        """Draw to window, write to files and release the frame"""

        # check if any grabbed frame is retrievable
        # the getter may retrive and cache the frame
        if self.frame is None:
            self._enteredFrame = False
            return

        # Update the FPS estimate and related variables
        if self._framesElapsed == 0:
            self._startTime = time.time()
        else:
            timeElapsed = time.time() - self._startTime
            self._fpsEstimate = self._framesElapsed/timeElapsed
            self._framesElapsed+=1

        # draw to window if any
        if self.previewWindowManager is not None:
            if self.shouldMirrorPreview:
                mirroredFrame = np.fliplr(self._frame).copy()
                self.previewWindowManager.show(mirroredFrame)
            else:
                self.previewWindowManager.show(self._frame)

        # write to image file if any
        if self.isWritingImage:
            cv2.imwrite(self._imageFilename,self._frame)
            self._imageFilename = None

        # write to video file, if any
        self._writeVideoFrame()

        # release the frame
        self._frame = None
        self._enteredFrame = False

    # public methods writeImage, startWritingVideo and stopWritingVideo simply record the parameters for file writing operations
    def writeImage(self,filename):
        """write the next exited frame to an image file"""
        self._imageFilename = filename

    def startWritingVideo(self, filename, encoding = cv2.VideoWriter_fourcc('I', '4', '2', '0')):
        self._videoFilename = filename
        self._videoEncoding = encoding

    def stopWritingVideo(self):
        """stop writing exited frame to a video file"""
        self._videoFilename = None
        self._videoEncoding = None
        self._videoWriter = None

    def _writeVideoFrame(self):
        '''creates or appends to a videofile'''
        if not self.isWritingVideo:
            return
        if self._videoWriter is None:
            fps = self._capture.get(cv2.CAP_PROP_FPS)
            if fps <=0.0:
                # the capture's estimate is unknown so use an estimate
                if self._framesElapsed<20:
                    # wait until more frames elapse so that the estimate is more stable
                    return
                else:
                    fps= self._fpsEstimate
            size = (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cv2.CAP_PROP_FRAME_HEIGHT))
            self._videoWriter = cv2.VideoWriter(self._videoFilename, self._videoEncoding, fps, size)
        self._videoWriter.write(self._frame)



class WindowManager(object):
    def __init__(self, windowName, keypressCallback=None):
        self.keypressCallback = keypressCallback
        self._windowName = windowName
        self._isWindowCreated = False

    @property
    def isWindowCreated(self):
        return self._isWindowCreated

    def createWindow(self):
        cv2.namedWindow(self._windowName)
        self._isWindowCreated = True

    def show(self, frame):
        cv2.imshow(self._windowName, frame)

    def destroyWindow(self):
        cv2.destroyWindow(self._windowName)
        self._isWindowCreated = False

    def processEvents(self):
        keycode = cv2.waitKey(1)
        if self.keypressCallback is not None and keycode!=-1:
            # discard any non-ascii info encoded by GTK
            #'0xFF is a hexadecimal constant which is 11111111 in binary. By using bitwise AND (&) with this constant, it leaves only the last 8 bits of the original (in this case, whatever keycode is).'
            keycode &= 0xFF
            self.keypressCallback(keycode)


class PygameWindowManager(WindowManager):

    def createWindow(self):
        pygame.display.init() # creates a window
        pygame.display.set_caption(self._windowName)
        self._isWindowCreated = True

    def show(self,frame):
        # find the frame's dimension in (w,h) format
        frameSize = frame.shape[1::-1]
        # convert the frame to RGB, which Pygame requires
        if utils.isGray(frame):
            conversionType = cv2.COLOR_GRAY2RGB
        else:
            conversionType = cv2.COLOR_BGR2RGB
        rgbFrame = cv2.cvtColor(frame, conversionType)

        # convert frame to pygame's surface type
        pygameFrame = pygame.image.frombuffer(rgbFrame.tostring(), frameSize,'RGB')
        # resize the window to match the frame

        displaySurface = pygame.display.set_mode(frameSize)

        # blit and display the frame
        displaySurface.blit(pygameFrame,(0,0))
        pygame.display.flip()

    def destroyWindow(self):
        pygame.display.quit()
        self._isWindowCreated = False

    def processEvents(self):

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and self.keypressCallback is not None:
                self.keypressCallback(event.key)
            elif event.type == pygame.QUIT:
                self.destroyWindow()
                pygame.quit()
                sys.exit()
