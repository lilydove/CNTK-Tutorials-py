# https://blog.csdn.net/qq_14845119/article/details/52354394
import cv2
import numpy as np
fname = 'E:\\ProgramLib\\Python\\CNTK\\testdata\\Conv2D.png'
img = cv2.imread(fname)
emptyImage = np.zeros(img.shape, np.uint8)

emptyImage2 = img.copy()

emptyImage3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imshow("EmptyImage3", emptyImage3)
cv2.waitKey (0)
cv2.destroyAllWindows()