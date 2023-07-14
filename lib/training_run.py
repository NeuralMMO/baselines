import logging
import os

import torch
from attr import dataclass

import wandb


@dataclass
class RunState:
  run_name: str
  wandb_project: str = None
  wandb_entity: str = None
  wandb_run_id: str = None
  policy_checkpoint_name: str = None

  trainer: dict = None
  num_epochs_trained: int = 0
  args: dict = {}


class TrainingRun:
  def __init__(self, name, runs_dir: str, args: dict):
    self.name = name
    self._data_dir = os.path.join(runs_dir, name, "data")
    self._persist_path = os.path.join(runs_dir, name, "run.pt")
    self._state = RunState(name, args=args)
    self._use_wandb = False

    logging.info(
        f"TrainingRun: {self.name} data_dir {self._data_dir} args {args}")

    os.makedirs(self._data_dir, exist_ok=True)
    self._load()

  def data_dir(self):
    return self._data_dir

  def has_policy_checkpoint(self):
    return self._state.policy_checkpoint_name is not None

  def latest_policy_name(self):
    return self._state.policy_checkpoint_name

  def save_checkpoint(self, trainer):
    policy_name = f"{self.name}.{trainer.update:06d}"
    logging.info(f"TrainingRun: {self.name} saving checkpoint {policy_name}")

    self._state.trainer = trainer.get_trainer_state()
    self._state.num_epochs_trained = self._state.trainer.update
    self._state.policy_checkpoint_name = policy_name
    self._save()

  def resume_training(self, trainer):
    trainer.load_trainer_state(self._state.trainer)
    if self._use_wandb:
      trainer.wandb_initialized = True

  def enable_wandb(self, project: str, entity: str):
    if project is None:
      return

    self._state.wandb_project = project
    self._state.wandb_entity = entity
    if self._state.wandb_run_id is None:
      self._state.wandb_run_id = wandb.util.generate_id()
    self._use_wandb = True

    wandb.init(
        id=self._state.wandb_run_id,
        project=project,
        entity=entity,
        config=self._state.args,
        sync_tensorboard=True,
        name=self._state.run_name,
        monitor_gym=True,
        save_code=True,
        resume="allow",
    )

  def _save(self):
    tmp_path = self._persist_path + ".tmp"
    torch.save(self._state.__dict__, tmp_path)
    os.rename(tmp_path, self._persist_path)

  def _load(self):
    if os.path.exists(self._persist_path):
      self._state = RunState(**torch.load(self._persist_path))
