from typing import List
import inspect
import random

import nmmo
from nmmo.task import task_spec as ts
import nmmo.task.base_predicates

class RandomTaskSampler:
    def __init__(self, task_spec: List[ts.TaskSpec]):
        self.task_spec = task_spec
        self.eval_fn_code = self._get_eval_fn_code()

    def _get_eval_fn_code(self):
        # get the whole pre-built eval functions
        code = inspect.getsource(nmmo.task.base_predicates)
        # go through the task_spec and include the code of new functions
        for spec in self.task_spec:
            if not hasattr(nmmo.task.base_predicates, spec.eval_fn.__name__):
                code += "\n" + inspect.getsource(spec.eval_fn)
        return code

    def sample_tasks(self, num_tasks):
        # returning the task spec, which is sampled with replacement
        # CHECK ME: do we need to provide a random task generator?
        #   providing a manually curated task could do
        return random.choices(self.task_spec, k=num_tasks)
