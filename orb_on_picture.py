import cv2

image = cv2.imread("Images/" + "lip6kennedy_bigdoubleloop_000000.ppm")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('Image with ORB Keypoints', image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
