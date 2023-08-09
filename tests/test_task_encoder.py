import random
import unittest

import curriculum_generation.manual_curriculum
from curriculum_generation.task_encoder import TaskEncoder

LLM_CHECKPOINT = "Salesforce/codegen25-7b-instruct"
CURRICULUM_FILE_PATH = "curriculum_generation/curriculum_with_embedding.pkl"

# NOTE: models that are not Salesforce/codegen-350M-mono may give different number
EMBEDDING_DIM = 4096

class TestTaskEncoder(unittest.TestCase):
  # pylint: disable=protected-access,bad-builtin
  @classmethod
  def setUpClass(cls):
    cls.task_encoder = TaskEncoder(
        LLM_CHECKPOINT, curriculum_generation.manual_curriculum, batch_size=4
    )

  @classmethod
  def tearDownClass(cls):
    cls.task_encoder.close()

  def test_embed_dim(self):
    self.assertEqual(self.task_encoder.embed_dim, EMBEDDING_DIM)

  def test_task_encoder_api(self):
    task_spec_with_embedding = self.task_encoder.get_task_embedding(
      curriculum_generation.manual_curriculum.task_spec,
      save_to_file=CURRICULUM_FILE_PATH
    )

    for single_spec in task_spec_with_embedding:
      self.assertFalse(sum(single_spec.embedding) == 0)

  def test_get_task_deps_src(self):
    custom_fn = curriculum_generation.manual_curriculum.PracticeInventoryManagement
    fn_src, deps_src = self.task_encoder._get_task_deps_src(custom_fn)

    self.assertEqual(
        fn_src,
        "def PracticeInventoryManagement(gs, subject, space, num_tick):\n  "
        +
        "return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)\n",
    )
    self.assertTrue("def InventorySpaceGE(" in deps_src)
    self.assertTrue("def TickGE(" in deps_src)

  def test_contruct_prompt(self):
    single_spec = random.choice(curriculum_generation.manual_curriculum.task_spec)
    prompt = self.task_encoder._construct_prompt(
        single_spec.reward_to, single_spec.eval_fn, single_spec.eval_fn_kwargs
    )
    print(prompt)


if __name__ == "__main__":
  unittest.main()
