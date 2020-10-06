import cv2
import numpy as np
import scipy.interpolate

def createCurveFunc(points):
    """Return a function derived from control points"""
    if points is None:
        return None
    numPoints = len(points)
    if numPoints < 2:
        return None

    xs,ys = zip(*points)

    if numPoints < 4:
        kind = 'linear' # 'quadratic is not implemented'

    else:
        kind = 'cubic'
    return scipy.interpolate.interp1d(xs,ys,kind,bounds_error=False)
    # bounds_error = false to permit extrapolation and interpolation

# print(createCurveFunc([(1,3),(3,4),(4,7)]))

def createLookupArray(func, length=256):
    """
    returns a lookup for whole-number inputs to a function
    the lookup values are clamped to [0,length-1]
    """
    if func is None:
        return None

    lookupArray = np.empty(length)
    i=0
    while i <length:
        func_i = func(i)
        lookupArray[i] = min(max(0,func_i),length-1)
        i+=1
    return lookupArray

def applyLookupArray(lookupArray, src,dest):
    """ map a source to destination using lookup"""
    if lookupArray is None:
        return
    dest[:] = lookupArray[src]

# combining two curve function into one function before creating a lookup array (optimization)

def createCompositefuntion(func0, func1):
    """ Returns a composite of two functions."""
    if func0 is None:
        return func1
    if func1 is None:
        return func0
    return lambda x: func0(func1(x))

# for image with multiple channel, instead of splitting and merging, flatten and make it 1-d and pass to create lookup array

def createFlatView(array):
    """ Returns 1-d view of an array of any dimension """
    flatView = array.view()
    flatView.shape = array.size # size will return the mxn value of the array
    return flatView

def isGray(image):
    """return true if the image has one channel per pixel"""
    return image.ndim<3

def widthHeightDividedBy(image, divisor):
    """returns and image's dimension divided by number"""
    h,w = image.shape[:2]
    # tup = []
    # # print(type(h))
    # # h = (h/divisor).astype(int)
    # # w = (w/divisor).astype(int)
    # for i in range(len(h)):
    #     tup.append(tuple((w[i],h[i])))

    # print(tup)
    # print(w)
    # print(type(w))
    # print(h)
    # concat = np.concatenate((w,h))
    return (int(w/divisor), int(h/divisor))



