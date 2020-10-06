import cv2
import numpy as np
import open_cv_from_book.utils as utils

def recolorRC(src,dest):
    """
    simulate conversion from BGR to RC : B and G mixup to give cyan.
    The source and destination image must both be in BGR format
    dst.b = dst.g = 0.5*(src.b+src.g)
    dst.r = src.r
    """
    b,g,r = cv2.split(src)
    cv2.addWeighted(b,0.5,g,0.5,0,b)
    cv2.merge((b,b,r),dest)

def recolorRGV(src,dest):
    """
    Simulate conversion from BGR to RGV.
    The source and destination image must both be in BGR format
    Blues are desaturated (take per-pixel minimum of BGR)

    dst.b = min(src.b, src.g, src.r)
    dst.g = src.g
    dst.r = src.r
    """
    b,g,r = cv2.split(src)

    # compute per-pixel minimum of first 2 and stores in 3rd argument
    cv2.min(b,g,b)
    cv2.min(b,r,b)
    cv2.merge((b,g,r),dest)

def recolorCMV(src,dest):
    """ desature yellow instead of blue by taking per-pixel maximum of b,g,r and storing in b
    simulate conversion from bgr to cmv(cyan, magenta, value)
    """
    b,g,r = cv2.split(src)
    cv2.max(b,g,b)
    cv2.max(b,r,b)
    cv2.merge((b,g,r),dest)


class VFuncFilter(object):
    """A filter that applies a function to V (or all of BGR)"""

    def __init__(self,vFunc=None, dtype = np.uint8):
        length = np.iinfo(dtype).max+1
        self._vLookupArray = utils.createLookupArray(vFunc, length)

    def apply(self, src, dest):
        """ Apply filter with BGR or Grayscale source/destination"""
        srcFlatView = utils.createFlatView(src)
        destFlatView = utils.createFlatView(dest)
        utils.applyLookupArray(self._vLookupArray,srcFlatView,destFlatView)


class VCurveFilter(VFuncFilter):
    """A filter that applies curve to to V (or all of BGR)"""

    def __init__(self, vPoints, dtype = np.uint8):
        VFuncFilter.__init__(self,utils.createCurveFunc(vPoints),dtype)


class BGRFuncFilter(object):
    """A filter that applies different function to each of BGR"""

    def __init__(self, vFunc= None, bFunc=None, gFunc = None, rFunc = None, dtype = np.uint8):
        length = np.iinfo(dtype).max+1
        self._bLookupArray = utils.createLookupArray(utils.createCompositefuntion(bFunc,vFunc),length)
        self._gLookupArray = utils.createLookupArray(utils.createCompositefuntion(gFunc,vFunc),length)
        self._rLookupArray = utils.createLookupArray(utils.createCompositefuntion(rFunc, vFunc),length)


    def apply(self,src,dest):
        """apply the filter with BGR source/destination"""
        b,g,r = cv2.split(src)
        utils.applyLookupArray(self._bLookupArray,b,b)
        utils.applyLookupArray(self._gLookupArray,g,g)
        utils.applyLookupArray(self._rLookupArray,r,r)

        cv2.merge([b,g,r],dest)

class BGRCurveFilter(BGRFuncFilter):
    """A filter that applies different curve to each of BGR"""

    def __init__(self,vPoints =None, bPoints = None, gPoints = None, rPoints = None,  dtype = np.uint8):
        BGRFuncFilter.__init__(self, utils.createCurveFunc(vPoints), utils.createCurveFunc(bPoints), utils.createCurveFunc(gPoints),
                               utils.createCurveFunc(rPoints), dtype)



class BGRPortraCurveFilter(BGRCurveFilter):
    """Emulating KODAK-PORTRA-A filter that applies portra-like curves to BGR"""
    def __init__(self, dtype = np.uint8):
        BGRCurveFilter.__init__(self,vPoints=[(0,0),(23,20),(157,173),(255,255)],
                                bPoints = [(0,0),(41,46),(231,228),(255,255)],
                                gPoints=[(0,0),(52,47),(189,196),(255,255)],
                                rPoints=[(0,0),(69,69),(213,218),(255,255)],
                                dtype=dtype)

class BGRProviaCurveFilter(BGRCurveFilter):
    """Emulating FUji- Provia - a filter that applies provia like curve to BGR
    general purpose"""

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                bPoints=[(0, 0), (35, 25), (205, 227), (255, 255)],
                                gPoints=[(0, 0), (27, 21), (196, 207), (255, 255)],
                                rPoints=[(0, 0), (59, 54), (202, 210), (255, 255)],
                                dtype=dtype)

class BGRVelviaCurveFilter(BGRCurveFilter):
    """Emulating fuji-velvia - a filter that applies velvia-like curve to BGR
    optimized for landscape"""

    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                vPoints=[(0,0),(128,118),(221,215),(255,255)],
                                bPoints=[(0,0),(25,21),(122,153),(165,206),(255,255)],
                                gPoints=[(0,0),(25,21),(95,102),(181,208),(255,255)],
                                rPoints=[(0,0),(41,28),(183,209),(255,255)],
                                dtype=dtype)

class BGRCrossProcessCurveFilter(BGRCurveFilter):
    """Produce grungy look, a filter that applies cross-process like curves to BGR"""
    def __init__(self, dtype=np.uint8):
        BGRCurveFilter.__init__(self,
                                bPoints=[(0,20),(255,255)],
                                gPoints=[(0,0),(56,39),(208,226),(255,255)],
                                rPoints=[(0,0),(56,22),(211,255),(255,255)],
                                dtype=dtype)


def strokeEdges(src, dest, blurKsize=7, edgeKsize=5):
    if blurKsize>=3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc,cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U,graySrc,edgeKsize)
    normalizedInverseAlpha = (1.0/255)*(255-graySrc) #invert to get black edges on white background and normalize
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel*normalizedInverseAlpha

    cv2.merge(channels, dest)


class VConvolutionFilter(object):
    """A filter that applies a convolution to V (or all of BGR)."""
    def __init__(self,kernel):
        self._kernel = kernel

    def apply(self,src,dest):
        """apply the filter with BGR or Gray source/destination"""
        cv2.filter2D(src,-1,self._kernel, dest) # -1 represent destination image has same depth as source image

class SharpenFilter(VConvolutionFilter):
    """A sharpen filter with a 1-pixel radius"""
    def __init__(self):
        kernel = np.array([[-1,-1,-1],
                           [-1,9,-1],
                           [-1,-1,-1]]) # sharpening filter sum to 1 or greater than 1
        VConvolutionFilter.__init__(self,kernel)

class FindEdgesFilter(VConvolutionFilter):
    """An edge finding filter with 1-px radius"""
    def __init__(self):
        kernel = np.array([[-1,-1,-1],
                           [-1,8,-1],
                           [-1,-1,-1]])
        VConvolutionFilter.__init__(self,kernel)


# making a blur filter -> weights should sum upto 1 and should be positive throughout the neighborhood

class BlurFilter(VConvolutionFilter):
    """A blur filer with 2-px radius"""
    def __init__(self,kernel):
        kernel = np.array([[0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04],
                           [0.04,0.04,0.04,0.04,0.04]])
        VConvolutionFilter.__init__(self,kernel)

# let's make less symmetric kernel that blurs on one side (with positive weights) and sharpens on the other side (with negative weights)
# it will produce ridged or embossed effect

class EmbossFilter(VConvolutionFilter):
    """An emboss filter with 1-px radius"""
    def __init__(self):
        kernel = np.array([[-2,-1,0],
                           [-1,1,1],
                           [0,1,2]])
        VConvolutionFilter.__init__(self,kernel)
