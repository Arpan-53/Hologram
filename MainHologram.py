import cv2
import numpy as np
from hologram2 import percentage
from flask import Flask, request, jsonify, send_from_directory, send_file, abort
from werkzeug.utils import secure_filename
import os
import logging
import base64

FORMAT = '%(asctime)-15s [%(levelname)s] [%(filename)s:%(lineno)s]: %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO, filename="logs/intelliocr.out")
outputDict={}
uploaded_folder="SampleImagesTested"
# Load the image
app = Flask(__name__)
@app.route("/")
def welcome():
    #logging.critical("Welcome msg dispatched to: {}".format(request.remote_addr))
    return "<h1>Welcome to Hologram Detection</h1>"


@app.route('/Hologram', methods=['POST'])
def imageRead():
    faceid_image = request.files['faceid_image']
    filename_faceid_img = secure_filename(faceid_image.filename)
    filename_faceid_img_name = filename_faceid_img.split(".")[0]
    idx_sel = len(filename_faceid_img.split(".")) - 1
    filename_faceid_img_ext = filename_faceid_img.split(".")[idx_sel]
    filename_faceid_img = filename_faceid_img_name+"."+"png"
    #filename_faceid_img1 = filename_faceid_img_name + str(support.id_generator())
    #logger.info("filename_faceid_img = {}".format(filename_faceid_img))
    #faceid_image.save(os.path.join(EXIF_DIR, filename_faceid_img))
    faceid_image.save(os.path.join(uploaded_folder, filename_faceid_img))
    #test_im = Image.open(os.path.join(uploaded_folder, filename_faceid_img))
    #exif_dict = test_im._getexif()

    #image_path = "sampleTests/WhatsApp Image 2023-07-10 at 1.24.04 PM(1).jpeg"
    try:
        image = cv2.imread(os.path.join(uploaded_folder, filename_faceid_img))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding to segment the hologram region
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Threshold", threshold )

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
        cv2.imwrite("Gray.jpg",gray)
        # Combine the grayscale and hologram regions
        output_image = cv2.bitwise_or(gray, hologram_region)
        #percentage_white_pixels = (np.count_nonzero(threshold) / threshold.size) * 100
        #print("New percent: ",percentage_white_pixels)
        cv2.imwrite("outputImage.jpg",output_image)
# Display the result
#cv2.imshow("Grayscale Image with Hologram", output_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
        outputDict=percentage(output_image)
        return jsonify(outputDict)
    except:
        outputDict["hologram Detected"] = "Failed"
        return jsonify(outputDict)

