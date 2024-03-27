import cv2
import numpy as np

def sharpen_image_with_gradient(image, alpha):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the gradients in x and y directions
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Combine the gradients to get the overall gradient magnitude
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    # Normalize the gradient magnitude to range [0, 255]
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Add the scaled gradient to the original image
    sharpened_image = cv2.addWeighted(gray, 1 + alpha, gradient_magnitude, -alpha, 0)

    # Convert back to BGR for displaying
    sharpened_image = cv2.cvtColor(sharpened_image, cv2.COLOR_GRAY2BGR)

    return sharpened_image

# Read the image
image = cv2.imread("E:/Computer Vision slot D/programs/picture1.png")

# Set the alpha value for sharpening
alpha = 0.5

# Apply sharpening using gradient masking
sharpened_image = sharpen_image_with_gradient(image, alpha)

# Display the original and sharpened images
cv2.imshow('Original Image', image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
