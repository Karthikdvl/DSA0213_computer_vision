import cv2

# Read the main image
image = cv2.imread("E:/Computer Vision slot D/programs/picture1.png")

# Read the watermark image
watermark = cv2.imread("C:/Users/user/OneDrive/Pictures/logo1.png", cv2.IMREAD_UNCHANGED)

# Define the region of interest (ROI) where the watermark will be inserted
x_offset = 100  # Adjust as needed
y_offset = 100  # Adjust as needed
roi_height, roi_width = watermark.shape[:2]
roi = image[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width]

# Resize the watermark image to match the size of the ROI
wm = cv2.resize(watermark, (roi_width, roi_height))

# Blend the watermark with the ROI using alpha blending
result = cv2.addWeighted(roi, 1, wm, 0.3, 0)

# Replace the region of interest in the original image with the watermarked ROI
image[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = result

# Display the watermarked image
cv2.imshow('Watermarked Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
