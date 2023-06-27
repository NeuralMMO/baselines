import inspect
import random
import numpy as np
import ast
from tqdm import tqdm
import json

from elm_curriculum_gen import OpenELMCurriculumGenerator

from transformers import AutoModelForCausalLM, AutoTokenizer, CodeGenModel

import nmmo.task
from static_src_dict import src_mapping

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
    # self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    self.model = CodeGenModel.from_pretrained(checkpoint)
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # eval func definitions used in task spec, necessary for encoding the task
    # should include both pre-built and submitted
    self.eval_fn_code = None
    self.prompt_template = None

  def update_context(self, eval_fn_code):
    self.eval_fn_code = eval_fn_code

  def _construct_prompt(self, reward_to, eval_fn, eval_fn_kwargs):
    eval_src = inspect.getsource(eval_fn)
    called_functions = get_called_functions(eval_src)
    aux_src = "\n\n".join([src_mapping.get(call, '') for call in called_functions])

    eval_fn_kwargs = str(eval_fn_kwargs)
    # plug in these args to the prompt template, which will be fed into the model

    task_specific_prompt = f"""Your goal is to explain what an agent must accomplish in the neural nmmo,
      which is expressed as a python function.
      Neural MMO is a computationally accessible, open-source research platform that 
      simulates populations of agents in virtual worlds. We challenge you to train a
      team of agents to complete tasks they have never seen before against opponents
        they have never seen before on maps they have never seen before.
    
      The reward from this function goes to {reward_to}.
      
      The function name is {eval_fn.__name__}. These are the arguments that the function takes {eval_fn_kwargs}.
    
      The function source code is {eval_src}.

      This function calls these other functions {aux_src}.

      Explain step by step, and as accurately as possible,

      The agent's goal is"""
    
    return task_specific_prompt

  def get_task_embedding(self, task_spec, to_file=False):
    if to_file:
      output_file = open("task_embedding_file.json", "w+")

    task_spec_with_embedding = []
    for single_spec in tqdm(task_spec):
      if len(single_spec) == 3:
        reward_to, eval_fn, eval_kwargs = single_spec
        task_kwargs = {}
      elif len(single_spec) == 4:
        reward_to, eval_fn, eval_kwargs, task_kwargs = single_spec
        assert isinstance(task_kwargs, dict), 'task_kwargs must be a dict'
      else:
        raise ValueError('len(single_spec) must be either 3 or 4')    

      # TODO: make the below lines run
      prompt = self._construct_prompt(reward_to, eval_fn, eval_kwargs)
      tokens =  self.tokenizer(prompt, return_tensors="pt", truncation=True)
      embedding = self.model(**tokens)[0].mean(dim=1)
      task_kwargs['embedding'] = embedding
      # task_spec_with_embedding.append((reward_to, eval_fn, eval_kwargs, task_kwargs))

      if to_file:
        line_data = {
                'reward_to': reward_to,
                'eval_fn': eval_fn.__name__,
                'eval_kwargs': str(eval_kwargs),
                'embedding': embedding[0].tolist()
            }
        output_file.write(json.dumps(line_data) + '\n')

    if to_file:
      output_file.close()

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
    # self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
    # self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    self.elm_curriculum_generator = OpenELMCurriculumGenerator(
      temperature=1.1,
      model="2B",
      batch_size=4,
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

  def evolve_tasks(self, num_tasks, task_spec, weights=None):
    # if weights is not None:
    #   assert len(task_spec) == len(weights), 'cannot use weights'
    # TODO: actually generate valid functions and task by evolution
    #   perhaps, the weights could be helpful?

    # generate the new eval functions and add these to the inventory
    # TODO: consider separating the "active" vs. "reserve" eval fns
    #   -- "active" are the ones used in task_spec, so going into LLM
    #   -- "reserve" are the ones NOT currently used, but can be used in future
    # self.eval_fn_code += self._evolve_eval_fn()

    evolved_task_spec = self.elm_curriculum_generator.evolve_tasks(steps=100, task_spec=task_spec)
    return evolved_task_spec

def get_called_functions(source_code):
    tree = ast.parse(source_code)
    funcs = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return funcs

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
