import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import cv2
import logging
import os
import sys
import pyimc
import time
import traceback
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe, RunOnce, Periodic



import tfmodel

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
PATH_DIR = os.path.dirname(os.path.abspath(sys.argv[0])) + '/models'
print(PATH_DIR)

class ImageClassificationActor(DynamicActor):
    MODE_NOT_CONFIGURED = 0
    MODE_CONFIGURED = 1
    MODE_ACTIVE = 2

    def __init__(self, target_name,imc_id):
        super().__init__(imc_id,static_port=6011)
        self.target_name = target_name
        self.estate = None

        # This list contains the target systems to maintain communications with
        self.heartbeat.append(target_name)
        self.mode = self.MODE_NOT_CONFIGURED
        self.vc = None
        self.sample_freq = 0
        self.frame_counter = 0
        self.cam = None
        self.last_run = time.time()
        self.labels = None

        self.model = None

    def from_target(self, msg):
        try:
            node = self.resolve_node_id(msg)
            return node.name == self.target_name
        except KeyError:
            return False

    @Subscribe(pyimc.ImageClassificationControl)
    def on_Classification_Control(self, msg: pyimc.ImageClassificationControl):
        logging.info('Received control message -- {}'.format(msg))
        if msg.command == pyimc.ImageClassificationControl.CommandEnum.SETUP:
            try:
              logging.info('setting up')
              self.model = tfmodel.tfmodel(PATH_DIR, msg.model)
              with open((PATH_DIR + '/' + msg.model + '/dict.txt')) as f:
                lines = [line.rstrip('\n') for line in f]
              print(lines)
              self.labels = lines
              self.sample_freq = msg.sampling_freq

              if self.cam != None:
                 self.cam.release()
              self.cam = cv2.VideoCapture(msg.video_source)
              if not self.cam.isOpened():
                 raise Exception('Error setting up video source')
                 self.cam.release()
              self.dump_vc_props()
              self.mode = self.MODE_CONFIGURED
              self.convert_frames_to_rgb = int(self.cam.get(cv2.CAP_PROP_CONVERT_RGB)) != 0
              logging.info('setup complete')
            except:
                logging.error('Error during setup')
                traceback.print_exc()
                self.mode = self.MODE_NOT_CONFIGURED
            
        elif msg.command == pyimc.ImageClassificationControl.CommandEnum.START:
            if self.mode == self.MODE_CONFIGURED:
                self.mode = self.MODE_ACTIVE
                logging.info('active')
            elif self.mode == self.MODE_NOT_CONFIGURED:
                logging.error('setup not done')
            else:
                logging.error('already active')
                
        elif msg.command == pyimc.ImageClassificationControl.CommandEnum.STOP:    
            if self.mode == self.MODE_ACTIVE:    
                self.mode = self.MODE_CONFIGURED  
                logging.info('inactive')  
                cv2.destroyAllWindows()       
            else:
                logging.error('not active') 

    def dump_vc_props(self):
        logging.info("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)))
        logging.info("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        logging.info("CAP_PROP_FPS : '{}'".format(self.cam.get(cv2.CAP_PROP_FPS)))
        logging.info("CAP_PROP_POS_MSEC : '{}'".format(self.cam.get(cv2.CAP_PROP_POS_MSEC)))
        logging.info("CAP_PROP_FRAME_COUNT  : '{}'".format(self.cam.get(cv2.CAP_PROP_FRAME_COUNT)))
        logging.info("CAP_PROP_BRIGHTNESS : '{}'".format(self.cam.get(cv2.CAP_PROP_BRIGHTNESS)))
        logging.info("CAP_PROP_CONTRAST : '{}'".format(self.cam.get(cv2.CAP_PROP_CONTRAST)))
        logging.info("CAP_PROP_SATURATION : '{}'".format(self.cam.get(cv2.CAP_PROP_SATURATION)))
        logging.info("CAP_PROP_HUE : '{}'".format(self.cam.get(cv2.CAP_PROP_HUE)))
        logging.info("CAP_PROP_GAIN  : '{}'".format(self.cam.get(cv2.CAP_PROP_GAIN)))
        logging.info("CAP_PROP_CONVERT_RGB : '{}'".format(self.cam.get(cv2.CAP_PROP_CONVERT_RGB)))
    
    @Periodic(0.1)
    def main_Loop(self):       
        if self.mode != self.MODE_ACTIVE:
            return

        current_time = time.time()

        ret, frame = self.cam.read()
        if not ret:
            logging.error('failed to grab frame')
            self.mode = self.MODE_NOT_CONFIGURED
            return

        if self.convert_frames_to_rgb:
           frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('test', frame)
        if( current_time - self.last_run < 1 / self.sample_freq ):
            return
        self.last_run = current_time

        start_time = time.time()

        try:
            results = self.model.classify_Image(frame)
        except:
            traceback.print_exc()
            return

        elapsed_time = time.time() - start_time
        logging.info('Classification time: %.3d s' % elapsed_time)
        logging.info(results)
        msg = pyimc.ImageClassification()
        for (label, score) in results:
            new_Classification = pyimc.ScoredClassification()
            new_Classification.score = int(round(score*100))
            new_Classification.classification = label
            msg.classifications.append(new_Classification)

        msg.frameid = self.frame_counter
        logging.info(msg)
        msg.data = cv2.imencode('.png',frame)[1].tobytes()
        img_name = 'opencv_frame_{}.png'.format(self.frame_counter)
        cv2.imwrite(img_name, frame)
        logging.info('{} written!'.format(img_name))

        if  self.target_name in  self._nodes.keys():
            node = self.resolve_node_id(self.target_name)
            self.send(node,msg)

        self.frame_counter += 1
        
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
