from types import ModuleType
import ast
import inspect
import json
from typing import List
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, CodeGenModel


def is_function_type(obj):
  return inspect.isfunction(obj) and not inspect.isbuiltin(obj)

def extract_module_fn(module: ModuleType):
  fn_dict = {}
  for name, fn in module.__dict__.items():
    if is_function_type(fn) and not name.startswith('_'):
      fn_dict[name] = fn
  return fn_dict

######################################################################
# Task embedding assumptions
# CHECK ME: currently assuming each agent has ONLY ONE task assigned during training,
#   so that we don't have to add multiple task embeddings to feed into one agent
# TODO: when given multiple tasks, can agents prioritize and/or multi-task?
#   It seems to be a research questions.
class TaskEncoder:
  def __init__(self, checkpoint: str, context: ModuleType, batch_size=64):
    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.model = CodeGenModel.from_pretrained(checkpoint).to(self.device)
    self.model.eval()
    self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.batch_size = batch_size

    blank_embedding = self._get_embedding(["# just to get the embedding size"])
    self.embed_dim = len(blank_embedding[0])

    # context is usually the module where the task_spec is defined,
    # so this assumes that it imports and contains all the necessary functions,
    # including the pre-built and custom functions
    self._fn_dict = extract_module_fn(context)

  def update_context(self, context: ModuleType):
    self._fn_dict = extract_module_fn(context)

  def _get_embedding(self, prompts: List[str]):
    all_embeddings = []
    for i in range(0, len(prompts), self.batch_size):
      batch = prompts[i:i+self.batch_size]
      tokens = self.tokenizer(batch, return_tensors="pt",
                              padding=True, truncation=True).to(self.device)
      embeddings = self.model(**tokens)[0].mean(dim=1).detach().cpu().numpy()
      all_embeddings.extend(embeddings)
    return all_embeddings

  def _get_task_deps_src(self, eval_fn):
    eval_src = inspect.getsource(eval_fn)
    deps_fns = [node.func.id for node in ast.walk(ast.parse(eval_src))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    deps_src = "\n".join([inspect.getsource(self._fn_dict[fn_name]) for fn_name in deps_fns
                          if fn_name in self._fn_dict])
    return eval_src, deps_src

  def _construct_prompt(self, reward_to, eval_fn, eval_fn_kwargs):
    # pylint: disable=line-too-long
    eval_src, deps_src = self._get_task_deps_src(eval_fn)
    # plug in these args to the prompt template, which will be fed into the model
    # TODO: make it easy to test various prompts
    task_specific_prompt = f"""Your goal is to explain what an agent must accomplish in the neural nmmo,
      which is expressed as a python function.
      Neural MMO is a computationally accessible, open-source research platform that 
      simulates populations of agents in virtual worlds. We challenge you to train a
      team of agents to complete tasks they have never seen before against opponents
        they have never seen before on maps they have never seen before.
    
      The reward from this function goes to {reward_to}.
      
      The function name is {eval_fn.__name__}. These are the arguments that the function takes {eval_fn_kwargs}.
    
      The function source code is \n####\n{eval_src}#### .

      This function calls these other functions \n####\n{deps_src}#### .

      Explain step by step, and as accurately as possible,

      The agent's goal is"""

    return task_specific_prompt

  def get_task_embedding(self, task_spec, save_to_file:str =None):
    task_spec_with_embedding = []
    prompts = []
    for single_spec in tqdm(task_spec):
      if len(single_spec) == 3:
        reward_to, eval_fn, eval_kwargs = single_spec
        task_kwargs = {}
      elif len(single_spec) == 4:
        reward_to, eval_fn, eval_kwargs, task_kwargs = single_spec
        assert isinstance(task_kwargs, dict), 'task_kwargs must be a dict'
      else:
        raise ValueError('len(single_spec) must be either 3 or 4')

      prompt = self._construct_prompt(reward_to, eval_fn, eval_kwargs)
      prompts.append(prompt)
      task_spec_with_embedding.append((reward_to, eval_fn, eval_kwargs, task_kwargs))

    embeddings = self._get_embedding(prompts)
    for embedding, single_spec in zip(embeddings, task_spec_with_embedding):
      single_spec[3]['embedding'] = embedding

    if save_to_file: # use save_to_file as the file name, and assume it's json
      with open(save_to_file, "w+", encoding="utf-8") as f:
        for reward_to, eval_fn, eval_kwargs, task_kwargs in task_spec_with_embedding:
          line_data = {
            'reward_to': reward_to,
            'eval_fn': eval_fn.__name__,
            'eval_kwargs': str(eval_kwargs),
            'embedding': task_kwargs['embedding'].tolist()}
          f.write(json.dumps(line_data) + '\n')
      f.close()

    return task_spec_with_embedding
