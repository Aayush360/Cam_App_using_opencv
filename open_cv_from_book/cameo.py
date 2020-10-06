import cv2

from open_cv_from_book.managers import PygameWindowManager as WindowManager, CaptureManager
from open_cv_from_book import filters
from open_cv_from_book import rects
from open_cv_from_book.trackers import  FaceTracker
class Cameo(object):

    def __init__(self):
        self._windowManager = WindowManager('Cameo',self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0),self._windowManager,True)
        self._curveFilter = filters.BGRVelviaCurveFilter()
        self._faceTracker = FaceTracker()
        self._shouldDrawDebugRects = False



    def run(self):
        """ run the main loop"""
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame

            self._faceTracker.update(frame)
            faces = self._faceTracker.faces
            rects.swapRects(frame,frame, [face.faceRect for face in faces])

            # TO DO: filter the frame
            filters.strokeEdges(frame,frame)
            self._curveFilter.apply(frame,frame)
            if self._shouldDrawDebugRects:
                self._faceTracker.drawDebugRects(frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        """ Handle a keypress.
        space -> take screenshot
        tab -> start or stop recording
        escape -> quit"""
        if keycode == 32: #space
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9: # tab
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWritingVideo('screencast.avi')
            else:
                self._captureManager.stopWritingVideo()
        elif keycode == 27: # escape
            self._windowManager.destroyWindow()

        elif keycode == 120: #X
            self._shouldDrawDebugRects = not self._shouldDrawDebugRects


if __name__ =="__main__":
    Cameo().run()





