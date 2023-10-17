import cv2
import numpy as np

# loading the images
images = []
for i in range(0, 388):
    filename = f'lip6kennedy_bigdoubleloop_{i:06d}.ppm'
    img = cv2.imread("Images/" + filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(gray)

# initialize the instance of the ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor extractor
orb = cv2.ORB_create()

# create the FLANN (Fast Library for Approximate Nearest Neighbors) matcher with values that strike balance between
# efficiency and accuracy (higher values = better accuray but less efficiency)
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# set temporal threshold
temporal_threshold = 50

for i in range(len(images)):
    # extract keypoints and descriptors for the current image
    keypoints1, descriptors1 = orb.detectAndCompute(images[i], None)

    for j in range(i - 1, -1, -1):
        if i - j <= temporal_threshold:
            continue

        # extract keypoints and descriptors for the previous image
        keypoints2, descriptors2 = orb.detectAndCompute(images[j], None)

        # match keypoints between the current and previous images
        try:
            # matches = array of objects (with keypoint1, keypoint2 and distance)
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        except cv2.error:
            continue

        # apply ratio test to filter good matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) < 2:
                continue
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        # geometric verification
        if len(good_matches) > 10:
            # extract keypoints from the first image
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # from the 2nd image
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # estimate the homography between the current and previous images
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # if enough inliers => loop closure detected
            if np.sum(mask) > 20:
                print(f"Loop closure detected between image {i:04d}.ppm and {j:04d}.ppm")
                with open("results.txt", "a") as f:
                    f.write(str(i) + " " + str(j) + "\n")
                # Additional steps: Correct map or adjust localization estimate

