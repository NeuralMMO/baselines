
from lib.agent.agent import Agent

class BaselineAgent(Agent):

  def __init__(self, agent_id):
    super().__init__(agent_id)

  def act(self, observation):
    pass
