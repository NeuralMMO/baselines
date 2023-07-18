from lib.prioritized_level_replay.level import BaseLevel, Level


def test_base_level():
  id = "level_1"

  level = BaseLevel(id)

  assert level.id == id
  assert str(level) == id
  assert repr(level) == id


def test_level():
  id = "level_1"

  level = Level(id)

  assert level.id == id
  assert str(level) == id
  assert repr(level) == id
