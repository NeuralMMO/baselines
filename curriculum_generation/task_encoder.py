import ast
import inspect
import os
import gc
from types import ModuleType
from typing import List

import dill
import torch
import numpy as np
from nmmo.task import task_spec as ts
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_module_fn(module: ModuleType):
  fn_dict = {}
  for name, fn in module.__dict__.items():
    if inspect.isfunction(fn) and not inspect.isbuiltin(fn) and not name.startswith("_"):
      fn_dict[name] = fn
  return fn_dict


class TaskEncoder:
    """A class for encoding tasks into embeddings using a pretrained model."""
    
    def __init__(self, checkpoint: str, context: ModuleType, batch_size=2, tmp_file_path="tmp_task_encoder.pkl"):
        """
        Initialize the TaskEncoder.

        Args:
        checkpoint: Path to the pretrained model.
        context: Python module context in which tasks are defined.
        batch_size: Size of each batch during embedding computation.
        tmp_file_path: Temporary file path for saving intermediate data.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                              trust_remote_code=True,
                                                              device_map="auto",
                                                              load_in_8bit=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint,
                                                              trust_remote_code=True).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.temp_file_path = tmp_file_path
        self._fn_dict = extract_module_fn(context)

        blank_embedding = self._get_embedding(["# just to get the embedding size"])
        self.embed_dim = len(blank_embedding[0])

    def update_context(self, context: ModuleType):
        """Update the module context, extracting function dictionary."""
        self._fn_dict = extract_module_fn(context)

    def _get_embedding(self, prompts: List[str]) -> list:
        """
        Compute the embeddings of tasks.

        Args:
        prompts: List of tasks defined as prompts.

        Returns:
        A list of embeddings corresponding to input tasks.
        """
        all_embeddings = []
        with torch.no_grad():
            for i in tqdm(range(0, len(prompts), self.batch_size)):
                batch = prompts[i: i + self.batch_size]
                tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.device)
                outputs = self.model(**tokens, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1].mean(dim=1).detach().cpu().numpy()
                all_embeddings.extend(embeddings.astype(np.float16))
        return all_embeddings

    def _get_task_deps_src(self, eval_fn) -> tuple:
        """
        Extract source code and dependent functions of the evaluation function.

        Args:
        eval_fn: Function for task evaluation.

        Returns:
        A tuple with source code and dependencies of eval_fn.
        """
        eval_src = inspect.getsource(eval_fn)
        deps_fns = [node.func.id for node in ast.walk(ast.parse(eval_src)) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        deps_src = "\n".join([inspect.getsource(self._fn_dict[fn_name]) for fn_name in deps_fns if fn_name in self._fn_dict])
        return eval_src, deps_src

    def _construct_prompt(self, reward_to, eval_fn, eval_fn_kwargs) -> str:
        """
        Construct a task-specific prompt.

        Args:
        reward_to: Reward given to the agent upon successful completion of the task.
        eval_fn: Function for task evaluation.
        eval_fn_kwargs: Keyword arguments for eval_fn.

        Returns:
        A string representing the task prompt.
        """
        eval_src, deps_src = self._get_task_deps_src(eval_fn)
        task_specific_prompt = f"""Your goal is to explain what an agent must accomplish in the neural MMO.
        Neural MMO is a research platform simulating populations of agents in virtual worlds.
        The reward from this function goes to {reward_to}.
        The function name is {eval_fn.__name__}. These are the arguments that the function takes {eval_fn_kwargs}.
        The function source code is \n####\n{eval_src}#### .
        This function calls these other functions \n####\n{deps_src}#### .
        The agent's goal is"""
        return task_specific_prompt

    def get_task_embedding(self, task_spec_list: List[ts.TaskSpec], save_to_file: str = None):
        """
        Compute embeddings for given task specifications and save them to file.

        Args:
        task_spec_list: List of task specifications.
        save_to_file: Name of the file where the results should be saved.

        Returns:
        Updated task specifications with embeddings.
        """
        assert self.model is not None, "Model has been unloaded. Re-initialize the TaskEncoder."
        prompts = [self._construct_prompt(single_spec.reward_to, single_spec.eval_fn, single_spec.eval_fn_kwargs) for single_spec in task_spec_list]
        embeddings = self._get_embedding(prompts)

        for single_spec, embedding in zip(task_spec_list, embeddings):
            single_spec.embedding = embedding

        if save_to_file:  # use save_to_file as the file name
            with open(self.temp_file_path, "wb") as f:
                dill.dump(task_spec_list, f)
            os.replace(self.temp_file_path, save_to_file)

        return task_spec_list

    def close(self):
        # free up gpu memory
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


if __name__ == "__main__":
    import curriculum_generation.manual_curriculum as curriculum
    LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"
    CURRICULUM_FILE_PATH = "reinforcement_learning/curriculum_with_embedding.pkl"

    with TaskEncoder(LLM_CHECKPOINT, curriculum, batch_size=6) as task_encoder:
        task_encoder.get_task_embedding(
            curriculum.curriculum,
            save_to_file=CURRICULUM_FILE_PATH
        )
