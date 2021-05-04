import sys

import pyimc
from pyimc.network.udp import IMCSenderUDP
import socket, struct, asyncio, logging
from pyimc.common import multicast_ip

class ImageClassificationMessage:

	def __init__(self,imc_id,host,port):
		self.imc_id = imc_id
		self.host = host
		self.port = port

	def send_message(self,msg):
		if msg == None :
			print("ERROR: unknown message: " + sys.argv[3])
			return

		socket = IMCSenderUDP(self.host,self.port)
		#socket.__enter__()
		socket.send(msg,self.port)

if __name__ == '__main__':
	control = ImageClassificationMessage('1234', sys.argv[1], sys.argv[2])
	print(sys.argv)
	#./dune-sendmsg localhost 6001 ImageClassificationControl 2 newModel.tflite /home/miguel/Downloads/example.mjpg 0.2
	if sys.argv[3] ==  "ImageClassificationControl" :
		message = pyimc.ImageClassificationControl()
		message.command = int(sys.argv[4])
		message.model = sys.argv[5]
		message.video_source = sys.argv[6]
		message.sampling_freq = float(sys.argv[7])

		control.send_message(message)