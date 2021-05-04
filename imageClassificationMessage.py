import sys
from pyimc.network.udp import IMCSenderUDP

class ImageClassificationMessage:

	def __init__(self,imc_id):
		self.imc_id = imc_id
		self.host = ''  # Symbolic name meaning the local host
		self.port = 0  #	 Arbitrary non-privileged port
		self.command = 0
		self.model = None
		self.video_source = None
		self.sampling_freq = 0

	def send_message(self):
		if message == None :
			print("ERROR: unknown message: " + sys.argv[3])
			return

		# change "test" to IMC message
		socket = IMCSenderUDP(self.host,self.port)
		socket.send("test",self.port)

if __name__ == '__main__':
	message = ImageClassificationMessage(1234)
	print(sys.argv)
	#./dune-sendmsg localhost 6001 ImageClassificationControl 2 newModel.tflite /home/miguel/Downloads/example.mjpg 0.2
	if sys.argv[3] ==  "ImageClassificationControl" :
		message.host = sys.argv[1]
		message.port = sys.argv[2]
		message.command = sys.argv[4]
		message.model = sys.argv[5]
		message.video_source = sys.argv[6]
		message.sampling_freq = sys.argv[7]

		message.send_message()