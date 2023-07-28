import cv2

# Load the image
image = cv2.imread('sampleTests/docImg.png', cv2.IMREAD_GRAYSCALE)

# Convert the image to color
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Find contours in the image
contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over the contours and draw green lines
for contour in contours:
    cv2.drawContours(color_image, [contour], 0, (0, 255, 0), 2)

# Display the image with green contour lines
cv2.imshow('Contour Lines', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()