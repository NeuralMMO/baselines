import inspect
import random
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer

import nmmo.task

######################################################################
# to be provided by Joseph
#from cleanrl_ppo_lstm import train_on_tasks, evaluate_on_tasks
class DummyAgent:
  pass

dummy_stat = {'stat': np.nan}

def train_on_tasks(agent_model, task_spec_with_embedding):
  return DummyAgent(), dummy_stat

def evaluate_on_tasks(agent_model, task_spec_with_embedding):
  return dummy_stat

def load_agent_model(model_path):
  return DummyAgent()

######################################################################
# assuming something like this
# CHECK ME: currently assuming each agent has ONLY ONE task assigned during training,
#   so that we don't have to add multiple task embeddings to feed into one agent
# TODO: when given multiple tasks, can agents prioritize and/or multi-task?
#   It seems to be a research questions.
class TaskEmbeddingGenerator:
  def __init__(self, checkpoint): # OpenELM default
    # https://huggingface.co/docs/transformers/model_doc/codegen#how-to-use
    self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # eval func definitions used in task spec, necessary for encoding the task
    # should include both pre-built and submitted
    self.eval_fn_code = None
    self.prompt_template = None

  def update_context(self, eval_fn_code):
    self.eval_fn_code = eval_fn_code
    self.prompt_template = self._contruct_prompt_template()

  def _contruct_prompt_template(self):
    assert self.eval_fn_code is not None, 'eval_fn_code must be provided'
    # something like this?
    self.prompt_template = f"""
      Your goal is to explain what an agent must accomplish in the neural nmmo,
      which is expressed as a python function, <the function name>

      The neural mmo is ... explain the game

      <the function name> can use these data structure: gs, subject

      <the function name> can use these other functions
      {self.eval_fn_code}

      The source code of <the function name> is:

      The agent's goal has been defined with <the function name> and
      these parameters: <kwargs that go into the function>

      Explain step by step, and as accurately as possible,

      The agent's goal is"""
    print()

  def _construct_prompt(self, reward_to, eval_fn, eval_fn_kwargs):
    # plug in these args to the prompt template, which will be fed into the model
    pass

  def get_task_embedding(self, task_spec):
    #assert self.prompt_template is not None, 'prompt_template must be set'
    task_spec_with_embedding = []
    for reward_to, eval_fn, kwargs in task_spec:
      # TODO: make the below lines run
      #prompt = self._construct_prompt(reward_to, eval_fn, kwargs)
      #embedding = self.model.get_hidden_layer(prompt) # something like this? 
      embedding = np.array([1, 2, 3, 4]) # dummy
      task_spec_with_embedding.append((reward_to, eval_fn, kwargs, embedding))
    return task_spec_with_embedding


######################################################################
# NOTE: this is actually a random task sampler, which sample with replacement
class SimpleTaskGenerator:
  def __init__(self, task_spec):
    self.task_spec = task_spec
    self.eval_fn_code = self._get_eval_fn_code()

  def _get_eval_fn_code(self):
    # get the whole pre-built eval functions
    code = inspect.getsource(nmmo.task.base_predicates)
    # go through the task_spec and include the code of new functions
    for _, eval_fn, _ in self.task_spec:
      if not hasattr(nmmo.task.base_predicates, eval_fn.__name__):
        code += '\n' + inspect.getsource(eval_fn)
    return code

  def generate_tasks(self, num_tasks):
    # returning the task spec, which is sampled with replacement
    # CHECK ME: do we need to provide a random task generator?
    #   providing a manually curated task could do
    return random.choices(self.task_spec, k=num_tasks)


# Nishaanth's OpenELM task generator, assuming something like this
class OpenELMTaskGenerator(SimpleTaskGenerator):
  def __init__(self, task_spec, checkpoint):
    # OpenELM task generator uses the task_spec to produce new things
    #   and does NOT have to keep track
    super().__init__(task_spec)
    # OpenELM default is "Salesforce/codegen-2B-mono"
    self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

  def _add_eval_fn(self, fn_code):
    self.eval_fn_code += '\n'+fn_code

  def _evolve_eval_fn(self) -> str:
    # return new eval fn code
    return ''

  @property
  def active_fn_code(self) -> str:
    # TODO: return only the actively used eval fn code
    return self.eval_fn_code

  def evolve_tasks(self, num_tasks, task_spec, weights=None):
    if weights is not None:
      assert len(task_spec) == len(weights), 'cannot use weights'
    # TODO: actually generate valid functions and task by evolution
    #   perhaps, the weights could be helpful?

    # generate the new eval functions and add these to the inventory
    # TODO: consider separating the "active" vs. "reserve" eval fns
    #   -- "active" are the ones used in task_spec, so going into LLM
    #   -- "reserve" are the ones NOT currently used, but can be used in future
    self.eval_fn_code += self._evolve_eval_fn()

    # generate new task spec, but for now...
    new_task_spec = self.generate_tasks(num_tasks)
    return new_task_spec


# how to load functions from str and get the src of fn subset
"""
def load_functions(s):
    # Create an empty dictionary to store the functions
    functions = {}

    # Execute the string as code in a new local namespace
    exec(s, {}, functions)

    # Return the functions dictionary
    return functions

# Define a function string
func_string = "def func1(x):\n    return x + 1\n\n def func2(y):\n    return y * 2"

# Load the functions from the string
loaded_functions = load_functions(func_string)

# Get the source of a subset of functions
function_names = ['func1']
for name in function_names:
    if name in loaded_functions:
        source_code = inspect.getsource(loaded_functions[name])
        print(source_code)
"""



######################################################################
# Ryan's syllabus task sampler, assuming something like this
class SyllabusTaskSampler:
  def __init__(self, task_spec_with_embedding):
    self.task_spec_with_embedding = task_spec_with_embedding
    # something like this? to indicate which tasks to focus on currently
    self.sample_weights = None
    self._dummy_update_weight()
    print('Number of tasks:', len(self.task_spec_with_embedding))

  def _dummy_update_weight(self):
    num_task = len(self.task_spec_with_embedding)
    weight = np.random.random(num_task)
    self.sample_weights = weight / np.sum(weight)

  # just adding new tasks, not replacing the whole tasks
  def add_new_tasks(self, task_spec_with_embedding):
    # TODO: deduplication, the embeddings must be different
    self.task_spec_with_embedding += task_spec_with_embedding
    print('Number of tasks:', len(self.task_spec_with_embedding))

  def sample_tasks(self, num_tasks):
    # TODO: return meaningful task specs
    return random.choices(self.task_spec_with_embedding, k=num_tasks)

  def update(self, task_spec_with_embedding, train_stats):
    self._dummy_update_weight()
