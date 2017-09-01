import cv2
import numpy as np
import rectangles
import matplotlib.pyplot as plt

ig= cv2.imread('H:\\PyScanner\\images\\test1.jpg')

ig= cv2.resize(ig, (1500, 880))
orig = ig.copy()

gray_scaled = cv2.cvtColor(ig, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_scaled, (5, 5), 0)

cny = cv2.Canny(blurred, 0, 50)
orig_edged = cny.copy()

(contours, _) = cv2.findContours(cny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break
print target
print target.shape
approx = rectangles.rectify(target)
print '\n--------------------------------------------------------------\n'
print approx
print approx.shape
pts2 = np.float32([[0,0],[800,0],[800,800],[0,800]])
print '\n--------------------------------------------------------------\n'
print pts2
M = cv2.getPerspectiveTransform(approx, pts2)
dst = cv2.warpPerspective(orig, M, (800,800))

dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

cv2.imshow("dst.jpg", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
