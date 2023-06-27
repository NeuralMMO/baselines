import re

from openelm import ELM
from openelm.configs import ELMConfig, PromptModelConfig, MAPElitesConfig
from openelm.environments import ENVS_DICT

from curriculum.train_helper import SimpleTaskGenerator

from elm_for_nmmo.elm_helper import task_spec_to_str
from elm_for_nmmo.nmmo_env import NMMOConfig, NMMOEnvironment
from elm_for_nmmo.sample_tasks import import_str


class OpenELMTaskGenerator(SimpleTaskGenerator):
  """Container class to include all the configs and generate tasks"""
  def __init__(self, task_spec, checkpoint,
               temperature=1.1,
               batch_size=1,
               gen_fn_name="training_task"):
    pattern = r"Salesforce/codegen-(350M|2B|6B)-mono"
    assert re.match(pattern, checkpoint), "Provided model not supported"
    assert 0.9<=temperature<=1.4, "temperature should be between 0.9 and 1.4"
    super().__init__(task_spec)

    self.config = ELMConfig()
    self.config.batch_size = batch_size

    self.config.env = NMMOConfig()
    self.config.env.impr = import_str["short_import"]
    self.config.env.init_prompt = task_spec_to_str(task_spec)
    self.config.env.mutate = True
    self.config.env.batch_size = batch_size
    self.gen_fn_name = gen_fn_name
    self.config.env.gen_fn_name = gen_fn_name

    self.config.qd = MAPElitesConfig()

    self.config.model = PromptModelConfig()
    self.config.model.temp = temperature
    self.config.model.batch_size = batch_size
    self.config.model.model_path = checkpoint

    ENVS_DICT["NMMO"] = NMMOEnvironment

  def evolve_tasks(self, task_spec, num_tasks, steps=10):
    """Evolve the given task specs for the given number of steps
          and return the num_tasks task specs
    """
    # NOTE: evolve task to generate a function, then generate parameters to deliver num_tasks
    self.config.env.init_prompt = task_spec_to_str(task_spec)
    elm = ELM(self.config)

    best_task = None
    while best_task is None:
      elm.run(init_steps = 2, total_steps = steps)
      # for now, just use the maximum fitness genome
      # TODO: we may want to sample best ones
      best_task = elm.qd_algorithm.current_max_genome

    return best_task.generate_task_spec(num_tasks)

  # NOTE: For now, ELM does NOT remember what it has produced before
  #   Does adding "memory" to ELM help? Something like below?

  # generate the new eval functions and add these to the inventory
  # TODO: consider separating the "active" vs. "reserve" eval fns
  #   -- "active" are the ones used in task_spec, so going into LLM
  #   -- "reserve" are the ones NOT currently used, but can be used in future
