from pdb import set_trace as T
import numpy as np
import random as rand

from queue import PriorityQueue, Queue

import nmmo
from nmmo.lib import material

from scripted import utils

def adjacentPos(pos):
   r, c = pos
   return [(r - 1, c), (r, c - 1), (r + 1, c), (r, c + 1)]

def inSight(dr, dc, vision):
    return (
          dr >= -vision and
          dc >= -vision and
          dr <= vision and
          dc <= vision)

def vacant(tile):
   Tile     = nmmo.Serialized.Tile
   occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)
   matl     = nmmo.scripting.Observation.attribute(tile, Tile.Index)

   return matl in material.Habitable and not occupied

def random(config, ob, actions):
   direction                 = rand.choice(nmmo.action.Direction.edges)
   actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

def towards(direction):
   if direction == (-1, 0):
      return nmmo.action.North
   elif direction == (1, 0):
      return nmmo.action.South
   elif direction == (0, -1):
      return nmmo.action.West
   elif direction == (0, 1):
      return nmmo.action.East
   else:
      return rand.choice(nmmo.action.Direction.edges)

def pathfind(config, ob, actions, rr, cc):
   direction = aStar(config, ob, actions, rr, cc)
   direction = towards(direction)
   actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

def meander(config, ob, actions):
   agent  = ob.agent
   Entity = nmmo.Serialized.Entity
   Tile   = nmmo.Serialized.Tile

   r = nmmo.scripting.Observation.attribute(agent, Entity.R)
   c = nmmo.scripting.Observation.attribute(agent, Entity.C)

   cands = []
   if vacant(ob.tile(-1, 0)):
      cands.append((-1, 0))
   if vacant(ob.tile(1, 0)):
      cands.append((1, 0))
   if vacant(ob.tile(0, -1)):
      cands.append((0, -1))
   if vacant(ob.tile(0, 1)):
      cands.append((0, 1))
   if not cands:
      return (-1, 0)

   direction = rand.choices(cands)[0]
   direction = towards(direction)
   actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

def explore(config, ob, actions, spawnR, spawnC):
   vision = config.NSTIM
   sz     = config.TERRAIN_SIZE
   Entity = nmmo.Serialized.Entity
   Tile   = nmmo.Serialized.Tile

   agent  = ob.agent
   r      = nmmo.scripting.Observation.attribute(agent, Entity.R)
   c      = nmmo.scripting.Observation.attribute(agent, Entity.C)

   centR, centC = sz//2, sz//2
   vR, vC = centR-spawnR, centC-spawnC

   mmag = max(abs(vR), abs(vC))
   rr   = int(np.round(vision*vR/mmag))
   cc   = int(np.round(vision*vC/mmag))
   pathfind(config, ob, actions, rr, cc)

def evade(config, ob, actions, attacker):
   Entity = nmmo.Serialized.Entity

   sr     = nmmo.scripting.Observation.attribute(ob.agent, Entity.R)
   sc     = nmmo.scripting.Observation.attribute(ob.agent, Entity.C)

   gr     = nmmo.scripting.Observation.attribute(attacker, Entity.R)
   gc     = nmmo.scripting.Observation.attribute(attacker, Entity.C)

   rr, cc = (2*sr - gr, 2*sc - gc)

   pathfind(config, ob, actions, rr, cc)

def forageDijkstra(config, ob, actions, food_max, water_max, cutoff=100):
   vision = config.NSTIM
   Entity = nmmo.Serialized.Entity
   Tile   = nmmo.Serialized.Tile

   agent  = ob.agent
   food   = nmmo.scripting.Observation.attribute(agent, Entity.Food)
   water  = nmmo.scripting.Observation.attribute(agent, Entity.Water)

   best      = -1000 
   start     = (0, 0)
   goal      = (0, 0)

   reward    = {start: (food, water)}
   backtrace = {start: None}

   queue = Queue()
   queue.put(start)

   while not queue.empty():
      cutoff -= 1
      if cutoff <= 0:
         break

      cur = queue.get()
      for nxt in adjacentPos(cur):
         if nxt in backtrace:
            continue

         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = nmmo.scripting.Observation.attribute(tile, Tile.Index)
         occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)

         if not vacant(tile):
            continue

         food, water = reward[cur]
         food  = max(0, food - 1)
         water = max(0, water - 1)

         if matl == material.Forest.index:
            food = min(food+food_max//2, food_max)
         for pos in adjacentPos(nxt):
            if not inSight(*pos, vision):
               continue

            tile = ob.tile(*pos)
            matl = nmmo.scripting.Observation.attribute(tile, Tile.Index)
 
            if matl == material.Water.index:
               water = min(water+water_max//2, water_max)
               break

         reward[nxt] = (food, water)

         total = min(food, water)
         if total > best or (
                 total == best and max(food, water) > max(reward[goal])):
            best = total
            goal = nxt

         queue.put(nxt)
         backtrace[nxt] = cur

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   direction = towards(goal)
   actions[nmmo.action.Move] = {nmmo.action.Direction: direction}

def aStar(config, ob, actions, rr, cc, cutoff=100):
   Entity = nmmo.Serialized.Entity
   Tile   = nmmo.Serialized.Tile
   vision = config.NSTIM

   start = (0, 0)
   goal  = (rr, cc)

   if start == goal:
      return (0, 0)

   pq = PriorityQueue()
   pq.put((0, start))

   backtrace = {}
   cost = {start: 0}

   closestPos = start
   closestHeuristic = utils.l1(start, goal)
   closestCost = closestHeuristic

   while not pq.empty():
      # Use approximate solution if budget exhausted
      cutoff -= 1
      if cutoff <= 0:
         if goal not in backtrace:
            goal = closestPos
         break

      priority, cur = pq.get()

      if cur == goal:
         break

      for nxt in adjacentPos(cur):
         if not inSight(*nxt, vision):
            continue

         tile     = ob.tile(*nxt)
         matl     = nmmo.scripting.Observation.attribute(tile, Tile.Index)
         occupied = nmmo.scripting.Observation.attribute(tile, Tile.NEnts)

         #if not vacant(tile):
         #   continue

         if occupied:
            continue

         #Omitted water from the original implementation. Seems key
         if matl in material.Impassible:
            continue

         newCost = cost[cur] + 1
         if nxt not in cost or newCost < cost[nxt]:
            cost[nxt] = newCost
            heuristic = utils.lInfty(goal, nxt)
            priority = newCost + heuristic

            # Compute approximate solution
            if heuristic < closestHeuristic or (
                    heuristic == closestHeuristic and priority < closestCost):
               closestPos = nxt
               closestHeuristic = heuristic
               closestCost = priority

            pq.put((priority, nxt))
            backtrace[nxt] = cur

   #Not needed with scuffed material list above
   #if goal not in backtrace:
   #   goal = closestPos

   while goal in backtrace and backtrace[goal] != start:
      goal = backtrace[goal]

   return goal

