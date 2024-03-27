import cv2
import numpy as np

# Read the reference image and the image to be transformed
ref_image = cv2.imread("E:/Computer Vision slot D/programs/pic1.png")
transform_image = cv2.imread("E:/Computer Vision slot D/programs/pic2.png")

# Check if images were successfully loaded
if ref_image is None or transform_image is None:
    print("Error: Unable to read one or both images.")
else:
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    transform_gray = cv2.cvtColor(transform_image, cv2.COLOR_BGR2GRAY)

    # Perform feature detection and description
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(ref_gray, None)
    keypoints_transform, descriptors_transform = sift.detectAndCompute(transform_gray, None)

    # Perform feature matching
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    matches = matcher.knnMatch(descriptors_ref, descriptors_transform, k=2)

    # Filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    matched_image = cv2.drawMatches(ref_image, keypoints_ref, transform_image, keypoints_transform, good_matches, None)

    # Extract matched points
    src_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_transform[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Calculate Homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply Homography transformation
    transformed_image = cv2.warpPerspective(transform_image, H, (ref_image.shape[1], ref_image.shape[0]))

    # Display the original and transformed images
    cv2.imshow('Matched Features', matched_image)
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
