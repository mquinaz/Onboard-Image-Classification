#! /usr/bin/python3
import argparse
import sys
import pyimc
from pyimc.actors.dynamic import DynamicActor
from pyimc.decorators import Subscribe


class ListenActor(DynamicActor):
  def __init__(self, parameters):
    super().__init__(parameters.imc_address, static_port=parameters.port)

  @Subscribe(pyimc.Message)
  def on_message(self, msg: pyimc.Message):
    print(msg)

parser = argparse.ArgumentParser(description='Receive messages from image classification actor.')
parser.add_argument("-i", "--imc-address", help="IMC address", default=0x3335, type=int)
parser.add_argument("-p", "--port", help="host port", default=6012, type=int)
parameters = parser.parse_args()

actor = ListenActor(parameters)
actor.run()
   
