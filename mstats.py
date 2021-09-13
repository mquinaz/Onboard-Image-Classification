#! /usr/bin/python3
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import cv2
import math
import os
import sys
import time
import tfmodel

parser = argparse.ArgumentParser(description='Derive model metrics')
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + '/models'
parser.add_argument('-p', '--model_path', help='path for classification models', default=DEFAULT_MODEL_PATH)
parser.add_argument('model', help='model id')
parser.add_argument('dir', help='directory containing data set')
args = parser.parse_args()

classifier = tfmodel.Classifier(args.model_path, args.model)
correct = 0
total = 0
loss = 0
ctime = 0
for c in classifier.labels:
    for f in sorted(os.listdir(args.dir + '/' + c.lower())):
        total +=1
        file = args.dir + '/' + c.lower() + '/' + f
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        t = time.time()
        results = classifier.classify(image)
        ctime += time.time() - t
        # print('--- {} ---'.format(file)) 
        if results[0][0] == c:
            correct +=1
        for (i,r) in enumerate(results,1):
            if r[0] == c:
                loss += - math.log(r[1])
            # print('{}: {} {}'.format(i, r[0], int(round(100*r[1]))))

accuracy = correct / total
loss = loss / total
ctime = ctime / total
print('Files: %d' % (total))
print('Accuracy: %.2f' % (accuracy))
print('Loss: %.2f' % (loss))
print('Time p/image (ms): %.2f' %(ctime*1000))
