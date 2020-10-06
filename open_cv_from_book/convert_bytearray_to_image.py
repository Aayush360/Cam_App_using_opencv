import cv2
import numpy as np
import os

# make an array of 120,000 random bytes
random_byte_array = bytearray(os.urandom(120000))
flat_numpy_array = np.array(random_byte_array)

# convert and array to make 400x300 grayscale image

gray_image = flat_numpy_array.reshape(300,400)
cv2.imwrite('randomgray.png',gray_image)

# convert and array to make 400x100 color image

bgr_image = flat_numpy_array.reshape(100,400,3)
cv2.imwrite('randomcolor.png',bgr_image)

cv2.waitKey(0)
cv2.destroyAllWindows()