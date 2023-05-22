import colorsys
import json
import lzma
import logging
import os
import pickle
from typing import Dict

from nmmo.render.render_utils import np_encoder, patch_packet
from nmmo.render.replay_helper import FileReplayHelper

class TeamReplayHelper(FileReplayHelper):
  def __init__(self, team_helper) -> None:
    super().__init__()
    self._team_helper = team_helper
    self._colormap = self._rainbow_colormap(team_helper.num_teams)

  def _packet(self) -> Dict:
    packet = super()._packet()

    for ent_id, player in packet['player'].items():
      team_id, pos = self._team_helper.team_and_position_for_agent[ent_id]
      player['base']['name'] = f'Team{team_id:02d}-{pos}'
      player['base']['population'] = team_id
      player['base']['color'] = self._colormap[team_id]

    return packet

  def _metadata(self) -> Dict:
    return {
      **super()._metadata(),
      "teams": self._team_helper.teams
    }

  def _rainbow_colormap(self, num_teams):
    colormap = []
    for i in range(num_teams):
      hue = i / float(num_teams)
      r, g, b = tuple(int(255 * x) for x in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
      hexcode = f'#{r:02x}{g:02x}{b:02x}'
      colormap.append(hexcode)
    return colormap
