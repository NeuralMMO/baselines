from nmmo.task.base_predicates import (
    AllMembersWithinRange,
    InventorySpaceGE,
    TickGE,
    norm as norm_progress,
)


def PracticeFormation(gs, subject, dist, num_tick):
  return norm_progress(
      AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick)
  )


def PracticeInventoryManagement(gs, subject, space, num_tick):
  return norm_progress(
      InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)
  )


def PracticeEating(gs, subject):
  """The progress, the max of which is 1, should
  * increase small for each eating
  * increase big for the 1st and 3rd eating
  * reach 1 with 10 eatings
  """
  num_eat = len(subject.event.EAT_FOOD)
  progress = num_eat * 0.06
  if num_eat >= 1:
    progress += 0.1
  if num_eat >= 3:
    progress += 0.3
  return norm_progress(progress)
