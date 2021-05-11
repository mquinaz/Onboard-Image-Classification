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
			interpreter = tf.lite.Interpreter(model_path=self.path_file + self.model_file)
			interpreter.allocate_tensors()
			input_details = interpreter.get_input_details()
			output_details = interpreter.get_output_details()
			model_type = 0
			return [interpreter,input_details,output_details,model_type,lines]

	#interpreter = list_model_param[0]
	#input_details = list_model_param[1]
	#output_details = list_model_param[2]
	#model_type = list_model_param[3]
	#labels = list_model_param[4]

	def classify_Image(self,file_name,list_model_param,top_k=1):
		input_shape = list_model_param[1][0]['shape']
		input_data = np.expand_dims(cv2.resize( cv2	.imread(file_name) , (64,64) ) , axis=0).astype(np.float32)
		input_data = (np.float32(input_data) - 127.5) / 127.5
		list_model_param[0].set_tensor(list_model_param[1][0]['index'], input_data)

		list_model_param[0].invoke()
		output_data = list_model_param[0].get_tensor(list_model_param[2][0]['index'])
		results = np.squeeze(output_data)
		#results = tf.nn.softmax(results).numpy()

		print(results)
		ordered = np.argpartition(-results, top_k)
		return [(i, results[i]) for i in ordered]

#recebes msg.model e acedes a pasta path_dir + msg.model
#create_model carrega path_dir + msg.model + "/model.tflite"
#path_dir + msg.model + "/dict.txt"
