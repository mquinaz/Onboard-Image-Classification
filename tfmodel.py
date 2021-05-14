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
		self.interpreter = tf.lite.Interpreter(model_path=self.path_file + self.model_file + "/model.tflite")
		self.interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		self.floating_model = self.input_details[0]['dtype'] == np.float32
		self.height = self.input_details[0]['shape'][1]
		self.width = self.input_details[0]['shape'][2]

	#im_rgb = cv2.cvtColor(im_cv, cv2.COLOR_BGR2RGB)
	def classify_Image(self,file_name,top_k=1):
		input_shape = self.input_details[0]['shape']
		input_data = np.expand_dims(cv2.resize( cv2	.imread(file_name) , (self.width,self.height) ) , axis=0).astype(np.float32)
		if self.floating_model:
			input_data = (np.float32(input_data) - 127.5) / 127.5
		self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

		self.interpreter.invoke()
		output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
		results = np.squeeze(output_data)
		#results = tf.nn.softmax(results).numpy()

		print(results)
		ordered = np.argpartition(-results, top_k)
		return [(i, results[i]) for i in ordered]

#recebes msg.model e acedes a pasta path_dir + msg.model
#create_model carrega path_dir + msg.model + "/model.tflite"
#path_dir + msg.model + "/dict.txt"
