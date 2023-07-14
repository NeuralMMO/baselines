import torch

from lib.prioritized_level_replay.level import Level
from lib.prioritized_level_replay.replay import (
    PrioritizedReplay,
    PrioritizedReplayDistribution,
)


def test_prioritized_replay_distribution():
  level_1 = Level("level_1")
  level_2 = Level("level_2")
  level_3 = Level("level_3")

  score_levels = {level_1: 0.2, level_2: 0.1, level_3: 0.9}
  last_episode_levels = {level_1: 3, level_2: 1, level_3: 4}
  last_episode = 4
  dist = PrioritizedReplayDistribution()

  prioritized_dist = dist.create(
      score_levels, last_episode_levels, last_episode)

  assert isinstance(prioritized_dist, torch.Tensor)
  assert len(prioritized_dist) == len(score_levels)
  assert prioritized_dist.sum(dim=-1) == 1


def test_prioritized_replay_2():
  level_1 = Level("level_1")
  level_2 = Level("level_2")
  level_3 = Level("level_3")
  level_4 = Level("level_4")
  level_5 = Level("level_5")

  levels = [level_1, level_2, level_3, level_4, level_5]
  visited_levels = [level_1, level_3]
  score_levels = {level_1: 0.2, level_3: 0.1}
  last_episode_levels = {level_1: 3, level_3: 1}
  last_episode = 3

  replay = PrioritizedReplay(levels)

  assert replay.levels == levels

  next_level = replay.sample_next_level(
      visited_levels, score_levels, last_episode_levels, last_episode
  )

  assert isinstance(next_level, Level)
  assert next_level in levels
