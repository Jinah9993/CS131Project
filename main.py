# load config
import json
import random
import cv2
import base64
import numpy as np
import requests
import time
from simplegmail import Gmail

gmail = Gmail()

with open('config.json') as f:
    config = json.load(f)

API_KEY = config["API_KEY"]
MODEL = config["MODEL"]
SIZE = config["SIZE"]
EMAIL_CONFIG = config["simplegmail_config"]

# (if running locally replace https://detect.roboflow.com/ with eg http://127.0.0.1:9001/)
upload_url = "".join([
    "http://127.0.0.1:9001/",
    MODEL,
    "?api_key=",
    API_KEY,
    "&format=json"
])

# Get webcam interface via opencv-python
video = cv2.VideoCapture(0)

# Given an array of predictions, check if there are any 
# predictions that seem to be the target object.
def process_preds(preds, target_object):
    for pred in preds:
        if pred['class_name'] == target_object:
            return True
    return False

# Infer via the Roboflow Infer API and return the result
def infer():
    # Get the current image from the webcam
    ret, img = video.read()

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = SIZE / max(height, width)
    img = cv2.resize(img, (round(scale * width), round(scale * height)))

    # Encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw
    
    try:
        resp = json.loads(resp.read())
    except:
        print("Could not parse response.")
        
    print(resp)

    preds = resp["predictions"]
    detected = process_preds(preds, "target_object")
    
    if detected:
        execute_trigger(img)  
    
    return img

def execute_trigger(image):
    cv2.imwrite("detected_image.jpg", image)
    print("Image successfully saved! Attempting to send email.")
    
    email_config = EMAIL_CONFIG.copy()
    email_config["attachments"] = ["detected_image.jpg"]
    
    message = gmail.send_message(**email_config)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# Main loop; infers sequentially until you press "q"
while 1:
    # On "q" keypress, exit
    if(cv2.waitKey(1) == ord('q')):
        break

    # Capture start time to calculate fps
    start = time.time()

    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    cv2.imshow('image', image)

    # Print frames per second
    print((1/(time.time()-start)), " fps")

# Release resources when finished
video.release()
cv2.destroyAllWindows()
