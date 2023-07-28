import cv2
import numpy as np

# Load the hologram image
hologram_img = cv2.imread('sampleTests/docImg.png')

# Convert the image to HSV color space
hsv_img = cv2.cvtColor(hologram_img, cv2.COLOR_BGR2HSV)

# Define the lower and upper green color thresholds
lower_green = np.array([40, 50, 50])
upper_green = np.array([80, 255, 255])

# Create a mask for green color detection
mask = cv2.inRange(hsv_img, lower_green, upper_green)

# Apply morphological operations to remove noise
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Find contours in the image
contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour and draw a green dot on the hologram
for contour in contours:
    # Calculate the center of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Draw a green dot at the center of the contour
        cv2.circle(hologram_img, (cx, cy), 5, (0, 255, 0), -1)

# Display the image with detected green dots
cv2.imshow('Hologram with Green Dots', hologram_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
