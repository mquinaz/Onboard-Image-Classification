import sys

import pyimc
from pyimc.network.udp import IMCSenderUDP
import socket, struct, asyncio, logging
from pyimc.common import multicast_ip

if __name__ == '__main__':
	print(sys.argv)
	#./dune-sendmsg localhost 6011 ImageClassificationControl 2 tflite /home/miguel/Downloads/example.mjpg 0.2
	if sys.argv[3] ==  "ImageClassificationControl" :
		message = pyimc.ImageClassificationControl()
		message.command = int(sys.argv[4])
		message.model = sys.argv[5]
		message.video_source = sys.argv[6]
		message.sampling_freq = float(sys.argv[7])


		with IMCSenderUDP(sys.argv[1], None) as socket:
			socket.send(message, int(sys.argv[2]) )