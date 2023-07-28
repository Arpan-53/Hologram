import cv2
import numpy as np
import base64

# Load the hologram image
def percentage(image):
    output_dict={}
    #image_path = "WhatsApp Image 2023-07-10 at 1.24.03 PM(2).jpeg"
    #image = cv2.imread(image_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('Gray.jpg', gray)
    lower_green = (40, 50, 50)
    upper_green = (80, 255, 255)
    #cv2.imshow("gray", gray)
    # Apply image processing techniques to enhance hologram features
    # e.g., thresholding, edge detection, etc.
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Perform contour detection
    edges = cv2.Canny(threshold, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Perform morphological operations to improve the thresholded mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # Invert the mask to select the hologram region
    hologram_mask = cv2.bitwise_not(opened)
    # Calculate the ratio of hologram pixels to the total image pixels
    total_pixels = np.prod(hologram_mask.shape[:2])
    hologram_pixels = cv2.countNonZero(hologram_mask)
    hologram_percentage = (hologram_pixels / total_pixels) * 100
    print(hologram_percentage)
    # Iterate through the contours and analyze their properties
    if len(contours) > 0:
        output_dict["hologram Detected"] = "SuccessFul"
    else:
        output_dict["hologram Detected"] = "Fail"
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        # Approximate the contour to reduce the number of vertices
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Analyze the number of vertices in the contour
        num_vertices = len(approx)
    
        # Check if the contour shape resembles a typical hologram pattern
        # You may need to define specific conditions based on the hologram design
        if num_vertices >= 3 and num_vertices <= 10:
            # Perform additional checks or analysis to verify the contour as a hologram
            # e.g., aspect ratio, area, symmetry, color properties, etc.
            # You might need to experiment and adapt these checks based on your hologram design
        
        # If the contour passes the hologram detection criteria, mark it on the image
            cv2.drawContours(color_image, [approx], -1, (0, 255, 0), 2)

    # Display the result
    cv2.imwrite("ColorImage.jpg",color_image)
    retval, buffer = cv2.imencode('.jpg', gray)
    image_bytes = buffer.tobytes()

    # Encode the image bytes as base64
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    #output_dict["hologram percentage"]=hologram_percentage
    #output_dict["hologram image"]=base64_image
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return output_dict


    