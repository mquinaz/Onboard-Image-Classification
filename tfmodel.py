import logging
import sys
import cv2
import time
import tensorflow as tf
import os
import numpy as np



class tfmodel:

  def __init__(self, path, model):
    model_path = '{}/{}/model.tflite'.format(path, model)
    logging.info('Loading model from ' + model_path)
    self.interpreter = tf.lite.Interpreter(model_path)
    self.interpreter.allocate_tensors()
    self.input_details = self.interpreter.get_input_details()
    self.output_details = self.interpreter.get_output_details()
    self.floating_model = self.input_details[0]['dtype'] == np.float32
    print(self.floating_model)
    self.height = self.input_details[0]['shape'][1]
    self.width = self.input_details[0]['shape'][2]

  def classify_Image(self,frame,top_k=1):
    input_shape = self.input_details[0]['shape']
    input_data = np.expand_dims(cv2.resize(frame, (self.width, self.height)), axis=0)#.astype(np.float32)

    if self.floating_model:
      input_data = (np.float32(input_data) - 127.5) / 127.5
    self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

    self.interpreter.invoke()
    output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
    results = np.squeeze(output_data)
    if not self.floating_model: 
        results = results / 255.0
    print(results)
    ordered = np.argpartition(-results, top_k)
    return [(i, results[i]) for i in ordered]
