__author__='Alan Anderson'

import grpc
from concurrent import futures
import time

from wp_predict import WpRecService
from wprecservice_pb2_grpc import add_WpRecServiceServicer_to_server

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
add_WpRecServiceServicer_to_server(WpRecService(),server)

server.add_insecure_port('[::]:50051')
server.start()
try:
  while True:
    time.sleep(_ONE_DAY_IN_SECONDS)
except KeyboardInterrupt:
  server.stop(0)