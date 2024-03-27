import cv2
import numpy as np
# Read images
ref_img = cv2.imread("E:/Computer Vision slot D/programs/pic1.png")
trans_img = cv2.imread("E:/Computer Vision slot D/programs/pic2.png")

# Convert to grayscale
ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
trans_gray = cv2.cvtColor(trans_img, cv2.COLOR_BGR2GRAY)

# Find key points and descriptors
sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
kp_trans, des_trans = sift.detectAndCompute(trans_gray, None)

# Match descriptors
matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = matcher.knnMatch(des_ref, des_trans, k=2)

# Apply ratio test
good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

# Get corresponding points
src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_trans[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calculate Homography
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply Homography
transformed_img = cv2.warpPerspective(trans_img, H, (ref_img.shape[1], ref_img.shape[0]))

# Display results
cv2.imshow('Matched Features', cv2.drawMatches(ref_img, kp_ref, trans_img, kp_trans, good_matches, None))
cv2.imshow('Transformed Image', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
