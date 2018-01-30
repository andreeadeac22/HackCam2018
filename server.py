from flask import Flask, request, Response, redirect, url_for
import jsonpickle
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize the Flask application
app = Flask(__name__)

image = cv2.imread('antibody.jpg')

# route http posts to this method
@app.route('/', methods=['POST'])
def test():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("before imshow")
    cv2.imwrite('messigray.jpg', img)
    print("after imshow")

    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(img.shape[1], img.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route("/")
def hello():
    return "Hello World!"

# start flask app
app.run()