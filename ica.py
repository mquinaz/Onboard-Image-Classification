import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import cv2
import logging
import os
import sys
import pyimc
import tfmodel
import time
import traceback
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe, RunOnce, Periodic


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PATH_DIR = os.path.dirname(os.path.abspath(sys.argv[0])) + '/models'

class ImageClassificationActor(DynamicActor):
    MODE_NOT_CONFIGURED = 0
    MODE_CONFIGURED = 1
    MODE_ACTIVE = 2

    def __init__(self, target_name,imc_id):
        super().__init__(imc_id,static_port=6011)
        self.target_name = target_name
        self.video = None
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
        # Video source
        if self.video != None:
           try:
             self.video.release()
           except:
             pass
        self.video = None

    @Subscribe(pyimc.ImageClassificationControl)
    def on_Classification_Control(self, msg: pyimc.ImageClassificationControl):
        logging.info('Received control message -- {}'.format(msg))
        if msg.command == pyimc.ImageClassificationControl.CommandEnum.SETUP:
            self.reset()
            try:
              logging.info('setting up')
              self.setup = msg
              self.classifier = tfmodel.Classifier(PATH_DIR, self.setup.model)
              self.video = cv2.VideoCapture(self.setup.video_source)
              if not self.video.isOpened():
                 raise Exception('Error setting up video source')
              self.dump_vc_props()
              self.mode = self.MODE_CONFIGURED
              self.convert_frames_to_rgb = int(self.video.get(cv2.CAP_PROP_CONVERT_RGB)) != 0
              logging.info('setup complete')
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

    @Periodic(0.1)
    def classification_loop(self):       
        # Do nothing if not active
        if self.mode != self.MODE_ACTIVE:
            return

        # Capture frame
        ret, frame = self.video.read()
        if not ret:
            logging.error('failed to grab frame')
            self.reset()
            return

        # Account for sampling frequency
        current_time = time.time()
        if  current_time - self.last_run < 1 / self.setup.sampling_freq:
            return

        # Convert to RGB if necessary
        # cv2.imshow('test', frame)
        if self.convert_frames_to_rgb:
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Invoke model to classify image
        try:
            results = self.classifier.classify(frame)
        except:
            traceback.print_exc()
            return
        self.last_run = current_time
        classification_time = time.time() - current_time
        logging.info('Classification time: %.3f s' % classification_time)
        logging.info(results)

        # Save image to disk
        self.frame_counter += 1
        img_name = 'opencv_frame_{}.png'.format(self.frame_counter)
        cv2.imwrite(img_name, frame)
        logging.info('{} written!'.format(img_name))

        # Build ImageClassification message
        msg = pyimc.ImageClassification()
        msg.frameid = self.frame_counter
        for (label, score) in results:
            sc_msg = pyimc.ScoredClassification()
            sc_msg.score = int(round(score*100))
            sc_msg.classification = label
            msg.classifications.append(sc_msg)
        logging.info(msg)
        msg.data = cv2.imencode('.png',frame)[1].tobytes()

        # Send message
        if  self.target_name in  self._nodes.keys():
            node = self.resolve_node_id(self.target_name)
            self.send(node,msg)

    def dump_vc_props(self):
        logging.info("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)))
        logging.info("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logging.info("CAP_PROP_FPS : '{}'".format(self.video.get(cv2.CAP_PROP_FPS)))
        logging.info("CAP_PROP_POS_MSEC : '{}'".format(self.video.get(cv2.CAP_PROP_POS_MSEC)))
        logging.info("CAP_PROP_FRAME_COUNT  : '{}'".format(self.video.get(cv2.CAP_PROP_FRAME_COUNT)))
        logging.info("CAP_PROP_BRIGHTNESS : '{}'".format(self.video.get(cv2.CAP_PROP_BRIGHTNESS)))
        logging.info("CAP_PROP_CONTRAST : '{}'".format(self.video.get(cv2.CAP_PROP_CONTRAST)))
        logging.info("CAP_PROP_SATURATION : '{}'".format(self.video.get(cv2.CAP_PROP_SATURATION)))
        logging.info("CAP_PROP_HUE : '{}'".format(self.video.get(cv2.CAP_PROP_HUE)))
        logging.info("CAP_PROP_GAIN  : '{}'".format(self.video.get(cv2.CAP_PROP_GAIN)))
        logging.info("CAP_PROP_CONVERT_RGB : '{}'".format(self.video.get(cv2.CAP_PROP_CONVERT_RGB)))
    
        
if __name__ == '__main__':
    # Setup logging level and console output
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S', stream=sys.stderr)
    logging.info('started')
    
    # Create an actor, targeting the lauv-simulator-1 system
    actor = ImageClassificationActor('lauv-simulator-1',1234)

    # This command starts the asyncio event loop
    actor.run()
