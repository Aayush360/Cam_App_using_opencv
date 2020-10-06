import cv2
from open_cv_from_book import rects
from open_cv_from_book import utils
import numpy as np

# defining faces as hierarchy of rectangels
class Face(object):
    """ Data on facial features. face, eyes, nose, mouth"""
    def __init__(self):
        self.faceRect = None
        self.leftEyeRect = None
        self.rightEyeRect = None
        self.noseRect = None
        self.mouthRect = None


class FaceTracker(object):
    """A tracker for facial features: face, eyes, nose, mouth"""
    def __init__(self, scaleFactor=1.2, minNeighbors=2, flags = cv2.CASCADE_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags

        self._faces = []
        self._faceClassifier = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier('cascades/haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier('cascades/haarcascade_mcs_mouth.xml')


    @property
    def faces(self):
        """The tracked facial features. """
        return self._faces

    def update(self,image):
        """Update the tracked facial features"""
        self._faces = []
        if utils.isGray(image):
            image = cv2.equalizeHist(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.equalizeHist(image,image)
        minSize = utils.widthHeightDividedBy(image,divisor=8)
        print(minSize)
        print(type(minSize))
        # minSize = np.array(minSize)
        # print('imagw',image)
        # print('scalefactor', self.scaleFactor)
        # print('minnei',self.minNeighbors)
        faceRects = self._faceClassifier.detectMultiScale(image, self.scaleFactor, self.minNeighbors, self.flags, minSize)
        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                face.faceRect = faceRect
                x,y,w,h = faceRect
                # seek an eye in upper left part of the face
                searchRect = (int(x+w/7),y,int(w*2/7),int(h/2))
                face.leftEyeRect = self._detectOneObject(self._eyeClassifier,image, searchRect,64)

                # seek an eye in upper right part of the face

                searchRect = (int(x+w*4/7),y,int(w*2/7),int(h/2))
                face.rightEyeRect = self._detectOneObject(self._eyeClassifier,image,searchRect,64)

                # seek nose in middlepart of the face

                searchRect = (int(x+w/4),int(y+h/4),int(w/2),int(h/2))
                face.noseRect = self._detectOneObject(self._noseClassifier, image, searchRect,32)

                # seek a mouth in lower middle part of the face

                searchRect = (int(x+w/6), int(y+h*2/3), int(w*2/3), int(h/3))
                face.mouthRect = self._detectOneObject(self._mouthClassifier, image, searchRect,16)

                self._faces.append(face)

    def _detectOneObject(self, classifier, image, rect, imageSizeToMinSizeRatio):

        x,y,w,h = rect

        minSize = utils.widthHeightDividedBy(image, imageSizeToMinSizeRatio)

        subImage = image[y:y+h,x:x+w]

        subRects = classifier.detectMultiScale(subImage, self.scaleFactor, self.minNeighbors, self.flags, minSize)

        if len(subRects) == 0:
            return None

        subX,subY, subW, subH  = subRects[0]
        return (x+subX, y+subY, subW, subH)

    def drawDebugRects(self,image):
        """Draw rectangle around tracked facial features"""

        if utils.isGray(image):
            faceColor = 255
            leftEyeColor = 255
            rightEyeColor = 255
            noseColor = 255
            mouthColor = 255

        else:
            faceColor = (255,255,255) # white
            leftEyeColor = (0,0,255) # red
            rightEyeColor = (0,255,255) # yellow
            noseColor = (0,255,0) # green
            mouthColor = (255,0,0) #blue

        for face in self.faces:
            rects.outlineRect(image, face.faceRect, faceColor)
            rects.outlineRect(image, face.leftEyeRect, leftEyeColor)
            rects.outlineRect(image, face.rightEyeRect, rightEyeColor)
            rects.outlineRect(image, face.noseRect, noseColor)
            rects.outlineRect(image, face.mouthRect, mouthColor)




