import cv2
import numpy as np

def high_boost_filter(image, kernel_size, boost_factor):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    # Calculate the high-pass filtered image
    high_pass = gray - blurred

    # Apply the boost factor
    boosted_high_pass = np.clip(gray + boost_factor * high_pass, 0, 255).astype(np.uint8)

    # Convert back to BGR for displaying
    sharpened_image = cv2.cvtColor(boosted_high_pass, cv2.COLOR_GRAY2BGR)

    return sharpened_image

# Read the image
image = cv2.imread("E:/Computer Vision slot D/programs/picture1.png")

# Set parameters
kernel_size = 3
boost_factor = 1.5

# Apply high-boost filter
sharpened_image = high_boost_filter(image, kernel_size, boost_factor)

# Display the original and sharpened images
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
