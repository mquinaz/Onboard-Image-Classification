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


class h5model:

	def __init__(self, path_file,model_file):
		self.path_file = path_file
		self.model_type = model_file
		self.list_model_param = self.create_model()

	def create_model(self):
		with open((self.path_file + self.model_type + "/dict.txt")) as f:
			lines = [line.rstrip('\n') for line in f]
		print(lines)

		if os.path.isfile(self.path_file + self.model_type + "/model.h5"):
			model = keras.models.load_model(self.path_file + self.model_type + "/model.h5")
			return [model,lines]

	#self.model = list_model_param[0]
	#labels = list_model_param[1]

	def classify_Image(self,frame,top_k=1):
		#img = frame.convert('RGB')
		img_array = cv2.resize(frame, (224, 224))
		#img_array = np.asarray(frame) / 255
		img_array = tf.expand_dims(frame, 0)
		#input_data = input_data = np.expand_dims(frame, axis=0).astype(np.float32)

		predictions = self.list_model_param[0].predict(img_array, verbose=True)
		score = tf.nn.softmax(predictions[0])
		print(np.argmax(score) + " - " +  (100 * np.max(score)) )
		return np.argmax(score), 100 * np.max(score)
