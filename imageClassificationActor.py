import logging
import sys
from typing import Tuple
import asyncio
import pyimc
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe, RunOnce, Periodic
import cv2
import time

logger = logging.getLogger('examples.ImageClassification')

from PIL import Image
import tensorflow as tf
import numpy as np



class ImageClassificationActor(DynamicActor):
	MODE_NOT_CONFIGURED = 0
	MODE_CONFIGURED = 1
	MODE_ACTIVE = 2
	
	def __init__(self, target_name,imc_id):
		super().__init__(imc_id)
		self.target_name = target_name
		self.estate = None

		# This list contains the target systems to maintain communications with
		self.heartbeat.append(target_name)
		self.mode = self.MODE_NOT_CONFIGURED
		self.vc = None
		self.model = None
		self.sample_freq = 0
		self.frame_counter = 0
		self.cam = None
		self.interpreter = None
		self.input_details = None
		self.output_details = None
		self.last_run = time.time()
		self.labels = None
        
	def create_model(self,file_model):
		self.interpreter = tf.lite.Interpreter(model_path=file_model)
		self.interpreter.allocate_tensors()

		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		return self.interpreter
	
	def classify_Image(self,file_name,top_k=1):
		input_shape = self.input_details[0]['shape']
		input_data = np.expand_dims(cv2.resize( cv2	.imread(file_name) , (64,64) ) , axis=0).astype(np.float32)
		input_data = (np.float32(input_data) - 127.5) / 127.5
		self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

		self.interpreter.invoke()
		output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
		results = np.squeeze(output_data)
		results = tf.nn.softmax(results).numpy()

		ordered = np.argpartition(-results, top_k)
		return [(i, results[i]) for i in ordered]
		
	def from_target(self, msg):
		try:
			node = self.resolve_node_id(msg)
			return node.name == self.target_name
		except KeyError:
			return False
             
	
	@Subscribe(pyimc.ImageClassificationControl)
	def on_Classification_Control(self, msg: pyimc.ImageClassificationControl):
		logger.info('Received')
		if msg.command == pyimc.ImageClassificationControl.CommandEnum.SETUP:
			self.model = self.create_model(msg.model)
			self.mode = self.MODE_CONFIGURED
			self.sample_freq = msg.sampling_freq
			file_name = msg.model.split('.')[0] + ".txt"
			with open(file_name) as f:
				lines = [line.rstrip('\n') for line in f]
			self.labels = lines
			print(self.labels)
			try:
				if self.cam != None:
					self.cam.release()
				self.cam = cv2.VideoCapture(msg.video_source)
			except:
				logger.error("could not setup video source: " + msg.video_source)
				self.mode = self.MODE_NOT_CONFIGURED
			logger.info("setup")
			
		elif msg.command == pyimc.ImageClassificationControl.CommandEnum.START:
			if self.mode == self.MODE_CONFIGURED:
				self.mode = self.MODE_ACTIVE
				logger.info("active")
			elif self.mode == self.MODE_NOT_CONFIGURED:
				logger.error("setup not configured")
			else:
				logger.error("already active")
				
		elif msg.command == pyimc.ImageClassificationControl.CommandEnum.STOP:	
			if self.mode == self.MODE_ACTIVE:	
				self.mode = self.MODE_CONFIGURED  
				logger.info("inactive")  
				cv2.destroyAllWindows()       
			else:
				logger.error("not active") 
	
	@Periodic(0.1)
	def main_Loop(self):   	
		if self.mode != self.MODE_ACTIVE:
			return

		current_time = time.time()
		if( current_time - self.last_run < 1 / self.sample_freq ):
			return
		
		self.last_run = current_time
		ret, frame = self.cam.read()
		if not ret:
			logger.error("failed to grab frame")
			self.mode = self.MODE_CONFIGURED
			return
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			return

		frame = cv2.resize(frame,(64,64))
		cv2.imshow("test", frame)
		img_name = "opencv_frame_{}.png".format(self.frame_counter)
		cv2.imwrite(img_name, frame)
		logger.info("{} written!".format(img_name))

		start_time = time.time()
		results = self.classify_Image(img_name)
		elapsed_ms = (time.time() - start_time) * 1000
		print("Elapsed time: %.2f" % (elapsed_ms))
		msg = pyimc.ImageClassification()
		print(results)

		for tempTuple in results:
			label_id, prob = tempTuple
			print("%s - %.2f" % (self.labels[label_id], prob ) )

			new_Classification = pyimc.ScoredClassification()
			prob = abs(prob)
			new_Classification.score = prob
			new_Classification.classification = self.labels[label_id]
			msg.classifications.append(new_Classification)

		msg.frameid = self.frame_counter
		#msg.data = frame.tobytes()
		msg.data = cv2.imencode('.png',frame)[1].tobytes()

		node = self.resolve_node_id(self.target_name)

		self.send(node,msg)
		self.frame_counter += 1
		
if __name__ == '__main__':
	# Setup logging level and console output
	logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
	pil_logger = logging.getLogger('PIL')
	pil_logger.setLevel(logging.INFO)

	# Create an actor, targeting the lauv-simulator-1 system
	actor = ImageClassificationActor('lauv-simulator-1',1234)

	# This command starts the asyncio event loop
	actor.run()