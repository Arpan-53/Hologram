import cv2
import numpy as np
from hologram2 import percentage
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from werkzeug.utils import secure_filename
import os
import logging
import base64




image = cv2.imread("sampleTests/WhatsAppImage2023-07-10at1.24.01 PM.jpeg")

        # Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to segment the hologram region
_, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Perform morphological operations to improve the thresholded mask
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
opened = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
# Invert the mask to select the hologram region
hologram_mask = cv2.bitwise_not(opened)
# Convert the mask to color format
hologram_mask = cv2.cvtColor(hologram_mask, cv2.COLOR_GRAY2BGR)
# Extract the hologram region from the original image
hologram_region = cv2.bitwise_and(image, hologram_mask)
# Convert the remaining grayscale part of the image to color
gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Combine the grayscale and hologram regions
output_image = cv2.bitwise_or(gray, hologram_region)
        # Display the result
        #cv2.imshow("Grayscale Image with Hologram", output_image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
outputDict=percentage(output_image)
gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
retval, buffer = cv2.imencode('.jpg', gray)
image_bytes = buffer.tobytes()
base64_image = base64.b64encode(image_bytes).decode('utf-8')
#output_dict["hologram percentage"]=hologram_percentage
outputDict["hologram image"]=base64_image