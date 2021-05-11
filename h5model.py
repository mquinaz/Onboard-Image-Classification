import logging
import sys
from typing import Tuple
import asyncio
import pyimc
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe, RunOnce, Periodic
import cv2
import time
import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from PIL import Image
import numpy as np

logger = logging.getLogger('examples.ImageClassification')


class tfmodel:

	def __init__(self, path_file,model_file):
		self.path_file = path_file
		self.model_file = model_file

	def create_model(self):
		with open((self.path_file + "/dict.txt")) as f:
			lines = [line.rstrip('\n') for line in f]
		print(lines)

		if os.path.isfile(self.path_file + self.model_file):
			print("h5 model")
			# self.model = h5model.model(..)
			model = keras.models.load_model(self.path_file + model_file)
			model_type = 1
			return [model,model_type,lines]

	#self.model = list_model_param[0]
	#self.model_type = list_model_param[1]
	#labels = list_model_param[2]

	def classify_Image(self,file_name,list_model_param,top_k=1):
		img = Image.open(file_name)
		img = img.convert('RGB')
		img_array = np.asarray(img) / 255
		img_array = tf.expand_dims(img_array, 0)

		predictions = list_model_param[0].predict(img_array, verbose=True)
		score = tf.nn.softmax(predictions[0])
		print(np.argmax(score) + " - " +  (100 * np.max(score)) )
		return np.argmax(score), 100 * np.max(score)
