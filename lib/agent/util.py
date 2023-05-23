import numpy as np

# Loads a state dict into a model, skipping parameters that don't match in shape.
def load_matching_state_dict(model, state_dict):
  upgrade_required = False
  model_state_dict = model.state_dict()
  for name, param in state_dict.items():
    if name in model_state_dict:
      if model_state_dict[name].shape == param.shape:
        model_state_dict[name].copy_(param)
      else:
        upgrade_required = True
        print(f"Skipping {name} due to shape mismatch. " \
              f"Model shape: {model_state_dict[name].shape}, checkpoint shape: {param.shape}")
    else:
      upgrade_required = True
      print(f"Skipping {name} as it is not found in the model's state_dict")
  model.load_state_dict(model_state_dict, strict=False)
  return upgrade_required

def softmax(x, temperature=1.0):
    e_x = np.exp(x / temperature)
    return e_x / e_x.sum(axis=0)
