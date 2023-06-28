# Stores the model weights path and the rewards history for a policy

class PolicyPoolRecord():
  def __init__(self, model_weights_path: str):
    self._model_weights_path = model_weights_path
    self._num_samples = 0

  def record_sample(self):
    self._num_samples += 1

  def num_samples(self) -> int:
    return self._num_samples

  def to_dict(self):
    return {
      'model_weights_path': self._model_weights_path,
      'num_samples': self._num_samples,
      'rewards': []
    }

  @classmethod
  def from_dict(cls, data):
    policy = cls(data['model_weights_path'])
    policy._num_samples = data.get("num_samples", 1)
    return policy
