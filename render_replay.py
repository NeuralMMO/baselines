import time
import argparse

from nmmo.render.render_client import WebsocketRenderer
from nmmo.render.replay_helper import ReplayFileHelper

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    "--file", dest="file_name", type=str,
    help="path to the replay file to render")

  args = parser.parse_args()

  if args.file_name is None:
    raise ValueError("Please specify a replay file to render.")

  # open a client
  renderer = WebsocketRenderer()
  time.sleep(3)

  replay = ReplayFileHelper.load(args.file_name)

  # run the replay
  for packet in replay:
    renderer.render_packet(packet)
    time.sleep(1)
