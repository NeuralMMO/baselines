import re


from openelm import ELM
from openelm.configs import ELMConfig, PromptModelConfig, MAPElitesConfig
from openelm.environments import ENVS_DICT

from curriculum.train_helper import SimpleTaskGenerator

from elm_for_nmmo.elm_helper import task_spec_to_str, str_to_task_spec
from elm_for_nmmo.nmmo_env import NMMOConfig, NMMOEnvironment
from elm_for_nmmo.sample_tasks import import_str


class OpenELMCurriculumGenerator:
  """Container class to include all the configs and generate tasks"""
  def __init__(self, temperature, model, batch_size, task_specs):
    pattern = r"Salesforce/codegen-(350M|2B|6B)-mono"
    assert re.match(pattern, model), "Provided model not supported"
    assert 0.9<=temperature<=1.4, "temperature should be between 0.9 and 1.4"

    self.config = ELMConfig()
    self.config.batch_size = batch_size

    self.config.env = NMMOConfig()
    self.config.env.impr = import_str["short_import"]
    self.config.env.init_prompt = task_spec_to_str(task_specs)
    self.config.env.mutate = True
    self.config.env.batch_size = batch_size

    self.config.qd = MAPElitesConfig()

    self.config.model = PromptModelConfig()
    self.config.model.temp = temperature
    self.config.model.batch_size = batch_size
    self.config.model.model_path = model

    ENVS_DICT["NMMO"] = NMMOEnvironment

  def evolve_tasks(self, steps, task_spec):
    """Evolve the given task specs for the given number of steps
          and return the evolved task specs
    """
    self.config.env.init_prompt = task_spec_to_str(task_spec)
    elm = ELM(self.config)
    elm.run(init_steps = 2, total_steps = steps)
    # flatten genomes to a list of tasks
    return str_to_task_spec(list(elm.qd_algorithm.genomes.array.flatten()))


class OpenELMTaskGenerator(SimpleTaskGenerator):
  def __init__(self, task_spec, checkpoint):
    # OpenELM task generator uses the task_spec to produce new things
    #   and does NOT have to keep track
    super().__init__(task_spec)
    # OpenELM default is "Salesforce/codegen-2B-mono"
    # self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    self.elm_curriculum_generator = OpenELMCurriculumGenerator(
      temperature=1.1,
      model=checkpoint,
      batch_size=1,
      task_specs=task_spec
    )

  def _add_eval_fn(self, fn_code):
    self.eval_fn_code += '\n'+fn_code

  def _evolve_eval_fn(self) -> str:
    # return new eval fn code
    return ''

  @property
  def active_fn_code(self) -> str:
    # TODO: return only the actively used eval fn code
    return self.eval_fn_code

  # pylint: disable=unused-argument
  def evolve_tasks(self, num_tasks, task_spec):
    # generate the new eval functions and add these to the inventory
    # TODO: consider separating the "active" vs. "reserve" eval fns
    #   -- "active" are the ones used in task_spec, so going into LLM
    #   -- "reserve" are the ones NOT currently used, but can be used in future
    # self.eval_fn_code += self._evolve_eval_fn()

    evolved_task_spec = self.elm_curriculum_generator.evolve_tasks(steps=10, task_spec=task_spec)
    return evolved_task_spec


if __name__ == '__main__':
  # simulation participant's custom functions
  from submission import custom_curriculum as cc

  LLM_CHECKPOINT = "Salesforce/codegen-350M-mono"
  NUM_TRAIN_TASKS = 5
  NUM_TEST_TASKS = 5
  NUM_NEW_TASKS = 5

  task_generator = OpenELMTaskGenerator(cc.task_spec, LLM_CHECKPOINT)

  train_task_spec = task_generator.generate_tasks(NUM_TRAIN_TASKS)
  eval_task_spec = task_generator.generate_tasks(NUM_TEST_TASKS)
  new_task_spec = task_generator.evolve_tasks(NUM_NEW_TASKS, train_task_spec)
