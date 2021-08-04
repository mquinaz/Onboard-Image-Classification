#! /usr/bin/python3
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import cv2
import datetime
import logging
import os
import sys
import pyimc
import tfmodel
import time
import traceback
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe, RunOnce, Periodic
from pyimc.node import IMCService


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class ImageClassificationActor(DynamicActor):
    # Modes of operation
    MODE_NOT_CONFIGURED = 0
    MODE_CONFIGURED = 1
    MODE_ACTIVE = 2

    def __init__(self, parameters):
        super().__init__(parameters.imc_address,static_port=parameters.local_port)
        self.parameters = parameters
        self._static_transports[pyimc.ImageClassification] = \
          [IMCService(ip=parameters.static_dest_addr,
                     port=parameters.static_dest_port, 
                     scheme='udp')]
        self.reset()

    def reset(self):
        logging.info('resetting internal state')
        # Mode flag
        self.mode = self.MODE_NOT_CONFIGURED
        # ImageClasssificationControl message containing setup
        self.setup = None
        # Counter for classified frames
        self.frame_counter = 0
        # Timestamp for last classification
        self.last_run = 0
        # Model 
        self.classifier = None
        # Data dir
        self.data_dir = None
        # Log file
        if hasattr(self, 'lsf_file') and self.lsf_file != None:
            self.lsf_file.close()
        self.lsf_file = None
        # Video source
        if hasattr(self, 'video_source') and self.video_source != None:
           try:
             self.video_source.release()
           except:
             pass
        self.video_source = None

    @Subscribe(pyimc.ImageClassificationControl)
    def on_Classification_Control(self, msg: pyimc.ImageClassificationControl):
        logging.info('Received control message -- {}'.format(msg))
        if msg.command == pyimc.ImageClassificationControl.CommandEnum.SETUP:
            self.reset()
            try:
              logging.info('setting up')
              self.setup = msg
              self.classifier = tfmodel.Classifier(self.parameters.model_path, self.setup.model)
              self.video_source = cv2.VideoCapture(self.setup.video_source)
              if not self.video_source.isOpened():
                 raise Exception('Error setting up video source')
              self.dump_vc_props()
              self.mode = self.MODE_CONFIGURED
              self.data_dir = '{}/{}/{}'.format(parameters.data_path, 
                                                self.setup.model,
                                                datetime.datetime.now().strftime('%Y%m%d/%H%M%S'))
              os.makedirs(self.data_dir)
              self.lsf_file = open(self.data_dir + '/Data.lsf', 'wb')
              self.log_message(self.setup)
              logging.info('setup complete - data dir: ' + self.data_dir)
            except:
                logging.error('Error during setup')
                traceback.print_exc()
                self.reset()
        elif msg.command == pyimc.ImageClassificationControl.CommandEnum.START:
            if self.mode == self.MODE_CONFIGURED:
                self.mode = self.MODE_ACTIVE
                logging.info('now active')
            elif self.mode == self.MODE_NOT_CONFIGURED:
                logging.error('cannot start, not configured!')
            else:
                logging.error('already active')
        elif msg.command == pyimc.ImageClassificationControl.CommandEnum.STOP:    
            if self.mode == self.MODE_ACTIVE:    
                self.mode = self.MODE_CONFIGURED  
                logging.info('now inactive')  
                # cv2.destroyAllWindows()       
            else:
                logging.error('not active') 
        else:
            logging.error('invalid command received')

    @Periodic(0.001)
    def classification_loop(self):       
        # Do nothing if not active
        if self.mode != self.MODE_ACTIVE:
            return

        # Capture frame
        ret, frame = self.video_source.read()
        if not ret:
            logging.error('failed to grab frame - end of video stream reached?')
            self.reset()
            return

        # Account for sampling frequency
        current_time = time.time()
        if  current_time - self.last_run < 1 / self.setup.sampling_freq:
            return

        # Convert to RGB for proper handling by TFLite
        # cv2.imshow('test', frame)
        cl_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Invoke model to classify frame
        try:
            results = self.classifier.classify(cl_frame)
        except:
            traceback.print_exc()
            return
        self.last_run = current_time
        classification_time = time.time() - current_time
        logging.info('Classification time: %.3f s' % classification_time)
        logging.info(results)

        # Save image to disk
        self.frame_counter += 1
        img_name = '{}/f{:04d}.png'.format(self.data_dir, self.frame_counter)
        cv2.imwrite(img_name, frame)
        logging.info('{} written!'.format(img_name))

        # Build ImageClassification message
        ic_msg = pyimc.ImageClassification()
        ic_msg.frameid = self.frame_counter
        for (label, score) in results:
            sc_msg = pyimc.ScoredClassification()
            sc_msg.score = int(round(score*100))
            sc_msg.classification = label
            ic_msg.classifications.append(sc_msg)
        logging.info(ic_msg)
        compressed_frame = cv2.resize(frame, (128,128))
        ic_msg.data = cv2.imencode('.png',compressed_frame)[1].tobytes()

        # Log message
        self.log_message(ic_msg)

        # Send message
        self.send_static(ic_msg, set_timestamp=False)

        # Display image on GUI
        self.show_image(frame)

    def show_image(self, frame):
        if not self.parameters.headless:
            gui_frame = frame # cv2.resize(frame, (self.parameters.window_size, self.parameters.window_size))
            cv2.imshow('Onboard image classification', gui_frame)

    def log_message(self, message):
        message.src = self.imc_id
        message.set_timestamp_now()
        data = pyimc.Packet.serialize(message)
        self.lsf_file.write(data)

    def dump_vc_props(self):
        logging.info('CV_CAP_PROP_FRAME_WIDTH: {}'.format(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH)))
        logging.info('CV_CAP_PROP_FRAME_HEIGHT : {}'.format(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logging.info('CAP_PROP_FPS : {}'.format(self.video_source.get(cv2.CAP_PROP_FPS)))
        logging.info('CAP_PROP_POS_MSEC : {}'.format(self.video_source.get(cv2.CAP_PROP_POS_MSEC)))
        logging.info('CAP_PROP_FRAME_COUNT  : {}'.format(self.video_source.get(cv2.CAP_PROP_FRAME_COUNT)))
        logging.info('CAP_PROP_BRIGHTNESS : {}'.format(self.video_source.get(cv2.CAP_PROP_BRIGHTNESS)))
        logging.info('CAP_PROP_CONTRAST : {}'.format(self.video_source.get(cv2.CAP_PROP_CONTRAST)))
        logging.info('CAP_PROP_SATURATION : {}'.format(self.video_source.get(cv2.CAP_PROP_SATURATION)))
        logging.info('CAP_PROP_HUE : {}'.format(self.video_source.get(cv2.CAP_PROP_HUE)))
        logging.info('CAP_PROP_GAIN  : {}'.format(self.video_source.get(cv2.CAP_PROP_GAIN)))
        logging.info('CAP_PROP_CONVERT_RGB : {}'.format(self.video_source.get(cv2.CAP_PROP_CONVERT_RGB)))
    
        
if __name__ == '__main__':
    INSTALL_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) 
    DEFAULT_MODEL_PATH = INSTALL_PATH + '/models'
    DEFAULT_DATA_PATH = INSTALL_PATH + '/data'
    # Parse program parameters
    parser = argparse.ArgumentParser(description='Image classification actor.')
    parser.add_argument('-i', '--imc_address', help='local IMC address for actor', type=int, default=0x3334)
    parser.add_argument('-l', '--local_port', help='local port for incoming messages', default=6011, type=int)
    parser.add_argument('-d', '--data_path', help='path for generated data files', default=DEFAULT_DATA_PATH)
    parser.add_argument('-m', '--model_path', help='path for classification models', default=DEFAULT_MODEL_PATH)
    parser.add_argument('-a', '--static_dest_addr', help='static destination host', default='127.0.0.1')
    parser.add_argument('-p', '--static_dest_port', help='static destination port', type=int, default=6012)
    parser.add_argument('-w', '--window-size', help='GUI window size', type=int, default=800)
    parser.add_argument('-H', '--headless', help='headless mode (no GUI)', action='store_true')
    parameters = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stderr)
    logging.info('starting')
    # Create the actor and run it
    actor = ImageClassificationActor(parameters)
    actor.run()
