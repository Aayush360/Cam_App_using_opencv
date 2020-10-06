import cv2


# reading an image

image = cv2.imread('images/MyPic.png')
# cv2.imshow('my_image',image)
#
# # converting same imaga form png to jpg
#
# cv2.imwrite('MyPic.jpg',image)


# reading image as grayscale and saving as grayscale png version

image_gray = cv2.imread('images/MyPic.png',0)
cv2.imshow('grayscaled verion',image_gray)

# writing the grayscale image

cv2.imwrite('myPicGray.png',image_gray)


# converting between an image and raw-bytes


cv2.waitKey(0)
cv2.destroyAllWindows()
