import sys
import socket

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

		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
		#while True:
		#	s.sendto(bytes(message.video_source).encoding('utf-8'), (self.host, self.port))
		#	print("Sent : ", send_data, "\n\n")

		# uint16_t rv = IMC::Packet::serialize(msg, bfr, sizeof(bfr));
		# IMC::Message * msg = NULL;
		# std::list<IMC::Message*>* msg_list = new std::list<IMC::Message*>();
		s.close()

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