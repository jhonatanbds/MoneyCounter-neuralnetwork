# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages

from PIL import Image
import cv2
import tensorflow as tf
import flask
from flask import request
import numpy as np
import io
from loader import PredictionModel
from keras.models import load_model
import base64
from yolo import YOLO

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

def detect_img(yolo, img):
	
	print(img)
	# cv2.imshow("fsd", img)
	# cv2.waitKey(-1)
	r_image = yolo.detect_image(img)
	return r_image

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary
	data = {"success": False}
	
	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		model_name = request.args['model']
		image_format = request.args['format']
		image = None
		if(image_format == "b64"):
			json = flask.request.get_json()
			try:
				image = Image.open(io.BytesIO(base64.b64decode(json.get('image'))))
			except():
				data = {"success": False}
				data["message"] = "Cant Decode Base 64"
		else:
			if flask.request.files.get("image"):
				image_file = flask.request.files["image"].read()
				try:
					image = Image.open(io.BytesIO(image_file))
				except Exception:
					data = {"success": False}
					data["message"] = "Cant Decode Image"
					

		if(image != None):
			if(image.mode != "RGB"):
				data = {"success": False}
				data["message"] = "Image with wrong channels. Expected: RGB Detected: " + image.mode 
			else:
				data = {"success": True}
				data["coins"] = str(detect_img(YOLO(), image))
	# return the data dictionary as a JSON response
	print("VAI SERIALIZAR")
	return flask.jsonify(data)



# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading models and Flask starting server..."
		"please wait until server has fully started"))
	# load_models()
	app.run(host='0.0.0.0' , port=5000, debug=True )
