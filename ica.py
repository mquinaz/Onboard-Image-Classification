import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import cv2
import os
import sys
import time
import logging
import pyimc
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
              self.mode = self.MODE_CONFIGURED
              logging.info('setup complete')
            except:
                logging.error('Error during setup')
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

        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('test', frame)
        img_name = 'opencv_frame_{}.png'.format(self.frame_counter)
        if( current_time - self.last_run < 1 / self.sample_freq ):
            return
        self.last_run = current_time

        start_time = time.time()

        results = self.model.classify_Image(frame)

        elapsed_ms = (time.time() - start_time) * 1000
        print('Elapsed time: %.2f' % (elapsed_ms))
        msg = pyimc.ImageClassification()
        print(results)


        for tempTuple in results:
            label_id, prob = tempTuple
            print('%s - %.2f - ' % (self.labels[label_id], prob ) , end='')
            new_Classification = pyimc.ScoredClassification()
            prob = abs(prob)
            new_Classification.score = prob
            new_Classification.classification = self.labels[label_id]
            msg.classifications.append(new_Classification)

        msg.frameid = self.frame_counter
        #frame = cv2.resize(frame,(128,128))
        #msg.data = frame.tobytes()
        msg.data = cv2.imencode('.png',frame)[1].tobytes()
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
