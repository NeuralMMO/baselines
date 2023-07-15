import unittest

from openelm import ELM

from elm_for_nmmo.elm_curriculum_gen import OpenELMTaskGenerator
from elm_for_nmmo.elm_helper import (
    generate_task_spec,
    is_task_spec_valid,
    task_spec_to_str,
)
from elm_for_nmmo.nmmo_env import NMMOTaskFn
from submission import custom_curriculum as cc

LLM_CHECKPOINT = "Salesforce/codegen-350M-mono"
NUM_TRAIN_TASKS = 5
NUM_TEST_TASKS = 5
NUM_NEW_TASKS = 5

GEN_FN_NAME = "training_task"
VALID_TASK_FN = """def training_task(gs, subject, dist, num_tick):
  return norm(AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick))"""


class TestElmForNmmo(unittest.TestCase):
  def test_task_generator_api(self):
    # pylint: disable=unused-variable
    task_generator = OpenELMTaskGenerator(cc.task_spec, LLM_CHECKPOINT)
    train_task_spec = task_generator.generate_tasks(NUM_TRAIN_TASKS)
    eval_task_spec = task_generator.generate_tasks(NUM_TEST_TASKS)

    # to actually run elm, remove debug=True
    new_task_spec = task_generator.evolve_tasks(train_task_spec, NUM_NEW_TASKS,
                                                debug=True)

  def test_gnereate_task_spec(self):
    # see also elm_helper.sample_parameter

    # the simplest task without any pre-built functions
    # still, the env has to know what GameState and Group are
    test_str = """def training_task(gs: GameState, subject: Group, N: int):
      return 0"""
    task_spec = generate_task_spec(test_str, GEN_FN_NAME, num_sample=3)
    # the kwargs must have N
    self.assertTrue("N" in task_spec[0].eval_fn_kwargs)

    # the tasks with some pre-built functions
    task_spec = generate_task_spec(VALID_TASK_FN, GEN_FN_NAME, num_sample=3)
    # the kwargs must have dist and num_tick
    self.assertTrue("dist" in task_spec[0].eval_fn_kwargs)
    self.assertTrue("num_tick" in task_spec[0].eval_fn_kwargs)

  def test_is_task_spec_valid(self):
    # the tasks with some pre-built functions
    task_spec = generate_task_spec(VALID_TASK_FN, GEN_FN_NAME, num_sample=3)
    self.assertTrue(is_task_spec_valid(task_spec))

    # this is an invalid python function
    test_str = """def training_task(gs, subject,"""
    task_spec = generate_task_spec(test_str, GEN_FN_NAME, num_sample=3)
    self.assertFalse(is_task_spec_valid(task_spec))

    # this function uses halluciated function/predicates
    test_str = """def training_task(gs, subject, dist, num_tick):
      return norm(NonExistentFunc(gs, subject, dist) * TickGE(gs, subject, num_tick))"""
    task_spec = generate_task_spec(test_str, GEN_FN_NAME, num_sample=3)
    self.assertFalse(is_task_spec_valid(task_spec))

    # this is an infinite loop, so it should be invalid
    test_str = """
import time
def training_task(gs, subject):
  while True:
    time.sleep(0.1)
  return 0"""
    task_spec = generate_task_spec(test_str, GEN_FN_NAME)
    self.assertFalse(is_task_spec_valid(task_spec))

  def test_nmmo_genotype(self):
    gene = NMMOTaskFn(VALID_TASK_FN, GEN_FN_NAME)
    self.assertTrue(gene.valid)

    num_sample = 10
    task_spec = gene.generate_task_spec(num_sample)
    self.assertEqual(len(task_spec), num_sample)
    for single_spec in task_spec:
      self.assertTrue("dist" in single_spec.eval_fn_kwargs)
      self.assertTrue("num_tick" in single_spec.eval_fn_kwargs)

  def test_elm_prompt(self):
    # pylint: disable=protected-access,bad-builtin
    # NOTE: this is to test different elm prompt
    task_generator = OpenELMTaskGenerator(cc.task_spec, LLM_CHECKPOINT)
    train_task_spec = task_generator.generate_tasks(NUM_TRAIN_TASKS)

    elm_config = task_generator.config
    # NOTE: if init_prompt is long, it will cause CUDA out of memory error
    elm_config.env.init_prompt = task_spec_to_str(train_task_spec)
    elm = ELM(elm_config)
    nmmo_elm_env = elm.qd_algorithm.env
    code_batch = [nmmo_elm_env.construct_prompt()]  # batch of 1

    # generate!
    elm_results = nmmo_elm_env._generate_task_fn(code_batch)
    print(elm_results[0])


if __name__ == "__main__":
  unittest.main()
