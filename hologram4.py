import cv2

# Load the image
image = cv2.imread('sampleTests/docImg.png', cv2.IMREAD_GRAYSCALE)

# Convert the image to color to overlay green dots
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Define the lower and upper range of green color in HSV
lower_green = (40, 50, 50)
upper_green = (80, 255, 255)

# Convert the image to HSV
hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

# Threshold the image to obtain green regions
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and draw green dots
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.circle(color_image, (int(x + w/2), int(y + h/2)), 5, (0, 255, 0), 2)

# Display the original image with green dots
cv2.imshow('Hologram Detection', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
