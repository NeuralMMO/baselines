import os

from attr import dataclass
import torch
import wandb

from clean_pufferl import CleanPuffeRL
import logging

@dataclass
class RunState():
  run_name: str
  wandb_project: str = None
  wandb_entity: str = None
  wandb_run_id: str = None
  policy_checkpoint_name: str = None

  trainer: dict = None
  num_epochs_trained: int = 0
  args: dict = dict()

class TrainingRun():
  def __init__(self, name, run_state: RunState = None):
    self.name = name
    self._state = run_state or RunState(name)
    self._persist_path = None
    self._use_wandb = False

  def has_policy_checkpoint(self):
    return self._state.policy_checkpoint_name is not None

  def latest_policy_name(self):
    return self._state.policy_checkpoint_name

  def save_checkpoint(self, trainer):
    logging.info(f"TrainingRun: {self.name} saving checkpoint {trainer.update}")

    policy_name = f'{self.name}.{trainer.update}'

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
      id = self._state.wandb_run_id,
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
    if self._persist_path:
      tmp_path = self._persist_path + ".tmp"
      torch.save(self._state.__dict__, tmp_path)
      os.rename(tmp_path, self._persist_path)

  @staticmethod
  def load_or_create(name: str, runs_dir: str = None):
    if runs_dir is None:
      logging.info("No runs directory specified, creating a temporary run")
      return TrainingRun(name)

    os.makedirs(runs_dir, exist_ok=True)
    run_path = os.path.join(runs_dir, name + ".pt")

    if os.path.exists(run_path):
      logging.info(f"Loading run {name} from {run_path}")
      run_state = RunState(**torch.load(run_path))
    else:
      logging.info(f"Creating run {name} at {run_path}")
      run_state = RunState(name)

    tr = TrainingRun(name, run_state=run_state)
    tr._persist_path = run_path
    return tr


  def initialize_trainer(self, trainer: CleanPuffeRL):
    self.trainer_state = trainer.allocate_storage()
    trainer.init_wandb(
      wandb_project_name=self._wandb_project,
      wandb_entity=self._wandb_entity,
      wandb_run_id=self._wandb_run_id(),
      wandb_run_name=self._run_name,
      extra_data=vars(self._kwargs)
    )

    # if run_name is None:
    #   existing = os.listdir(args.experiments_dir)
    #   prefix_pattern = re.compile(f'^{prefix}(\\d{{4}})$')
    #   existing_numbers = [int(match.group(1)) for name in existing for match in [prefix_pattern.match(name)] if match]
    #   next_number = max(existing_numbers, default=0) + 1
    #   run_name = f"{prefix}{next_number:04}"

    # experiment_dir = os.path.join(args.experiments_dir, args.experiment_name)

    # os.makedirs(experiment_dir, exist_ok=True)

  # def resume_from_checkpoint(trainer):
  #   resume_from_path = None
  #   checkpoins = [cp for cp in os.listdir(experiment_dir) if cp.endswith(".pt")]
  #   if len(checkpoins) > 0:
  #     resume_from_path = os.path.join(experiment_dir, max(checkpoins))
  #     trainer.resume_model(resume_from_path)

  #   if args.wandb_project is not None:

  # def policy_db(self):
  #   policy_db = PolicyDatabase(
  #     backend=SQLiteBackend(training_run.policy_db()),
  #     policy_selector=OpenSkillPolicySelector(),
  #     policy_loader=model.policy_loader.PolicyLoader()
  #   )

  # def learner_policy(self):
  #   return PolicyLoader.policy_class(args.model_type)(binding)


