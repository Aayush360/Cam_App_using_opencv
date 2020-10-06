import cv2
# opencv rectangle() function argument represents a rectangle differently than Face does


def outlineRect(image, rect, color):
    if rect is None:
        return
    x,y,w,h = rect
    cv2.rectangle(image,(x,y),(x+w,y+h),color)

def copyRect(src, dest, srcRect, destRect, interpolation = cv2.INTER_LINEAR):
    """ copy part of source to part of destination"""
    x0,y0,w0,h0 = srcRect
    x1,y1,w1,h1 = destRect

    # resize the content of source sub-rectangle
    # put the result in destination sub-rectangle
    dest[y1:y1+h1, x1:x1+w1] = cv2.resize(src[y0:y0+h0,x0:x0+w0], (w1,h1), interpolation=interpolation)

    # for copying, swap one of the rectangle to a temporaray array before overwriting anything
def swapRects(src, dest, rects, interpolation=cv2.INTER_LINEAR):
    """copy the source with two or more sub-rectangle swapped"""
    if dest is not src:
        dest[:]=src
    numRects = len(rects)
    if numRects<2:
        return
    # copy the content of last rectangle into temporary array
    x,y,w,h = rects[numRects-1]
    temp = src[y:y+h, x:x+w].copy()

    # copy the content of each rectangle into the next
    i = numRects-2
    while i >=0:
        copyRect(src,dest, rects[i], rects[i+1],interpolation)
        i-=1
    # copy the temporarily stored content into the first rectangle
    copyRect(temp,dest,(0,0,w,h),rects[0], interpolation)
