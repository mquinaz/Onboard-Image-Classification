#! /usr/bin/python3
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import argparse
import cv2
import os
import sys
import tfmodel

parser = argparse.ArgumentParser(description='Test classification models for given images')
DEFAULT_MODEL_PATH = os.path.dirname(os.path.abspath(sys.argv[0])) + '/models'
parser.add_argument('-p', '--model_path', help='path for classification models', default=DEFAULT_MODEL_PATH)
parser.add_argument("-m", "--model", help="model id",  default="autoML")
parser.add_argument('files', nargs='+', help='files to classify')
args = parser.parse_args()

classifier = tfmodel.Classifier(args.model_path, args.model)

for file in args.files:
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = classifier.classify(image)
    print('--- {} ---'.format(file))
    for (i,r) in enumerate(results,1):
        print('{}: {} {}'.format(i, r[0], int(round(100*r[1]))))
