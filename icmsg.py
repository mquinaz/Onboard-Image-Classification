#! /usr/bin/python3
import argparse
import sys
import pyimc
from pyimc.network.udp import IMCSenderUDP

parser = argparse.ArgumentParser(description='Send message to image classification actor.')
parser.add_argument("-a", "--address", help="host name or IP address", default="127.0.0.1")
parser.add_argument("-p", "--port", help="host port", default=6011, type=int)
parser.add_argument("-m", "--model", help="model id (for setup command)", default="autoML")
parser.add_argument("-s", "--sampling_freq", help="sampling frequency (for setup command)", default="1.0", type=float)
parser.add_argument("-v", "--video_source", help="video source (for setup command)", default="example.mjpg")
parser.add_argument("command", help="Control command", choices=['setup', 'start', 'stop'])
args = parser.parse_args()

message = pyimc.ImageClassificationControl()
if args.command == 'setup':
    message.command = 2
    message.model = args.model
    message.sampling_freq = args.sampling_freq
    message.video_source = args.video_source
elif args.command == 'start':
    message.command = 0
elif args.command == 'stop':
    message.command = 1
else:
    raise Exception('Internal error!') 

message.set_timestamp_now()
with IMCSenderUDP(args.address, None) as socket:
   socket.send(message, args.port) 
   print('Message sent to {}:{} ...'.format(args.address, args.port))
   print(message)
   
