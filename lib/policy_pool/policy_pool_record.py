# Stores the model weights path and the rewards history for a policy

class PolicyPoolRecord():

  NUM_SAMPLES_TO_KEEP = 25

  def __init__(self, model_weights_path: str):
    self._model_weights_path = model_weights_path
    self._rewards = []

  def record_reward(self, reward: float):
    self._rewards.append(reward)
    if len(self._rewards) > self.NUM_SAMPLES_TO_KEEP:
      self._rewards.pop(0)

  def mean_reward(self) -> float:
    if len(self._rewards) == 0:
      return 0
    return sum(self._rewards) / len(self._rewards)

  def to_dict(self):
    return {
      'model_weights_path': self._model_weights_path,
      'rewards': self._rewards,
    }

  @classmethod
  def from_dict(cls, data):
    policy = cls(data['model_weights_path'])
    policy._rewards = data['rewards']
    return policy
