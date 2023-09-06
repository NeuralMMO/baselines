from typing import List
from collections import defaultdict
import numpy as np

from nmmo.task import task_spec as ts

class LearnableTaskSampler:
    def __init__(self,
                 task_spec: List[ts.TaskSpec],
                 average_window = 50):
        self.task_spec = task_spec
        self.name_to_spec = {single_spec.name: single_spec for single_spec in self.task_spec}
        self.task_stats = {}
        self.average_window = average_window

    def reset(self):
        self.task_stats = {}

    def add_tasks(self, task_spec: List[ts.TaskSpec]):
        # so that the stats for the new tasks can be tracked
        for new_spec in task_spec:
            if new_spec.name not in self.name_to_spec:
                self.task_spec.append(new_spec)
                self.name_to_spec[new_spec.name] = new_spec

    def update(self, infos, prefix="curriculum/"):
        for key, val in infos.items():
            # Process the new infos
            if key.startswith(prefix):
                spec_name = key.replace(prefix,"")
                completed, prog_over_10pcnt, rcnt_over_2 = [], [], []
                for sublist in val:
                    for prog, rcnt in sublist:
                        completed.append(float(prog >= 1))
                        rcnt_over_2.append(float(rcnt >= 2)) # rewarded >= 2 times

                # Add to the task_stats
                if spec_name not in self.task_stats:
                    self.task_stats[spec_name] = defaultdict(list)
                self.task_stats[spec_name]["completed"] += completed
                self.task_stats[spec_name]["rcnt_over_2"] += rcnt_over_2

                # Keep only the recent values -- self.average_window (50)
                for key, vals in self.task_stats[spec_name].items():
                    self.task_stats[spec_name][key] = vals[-self.average_window:]

    def get_learnable_tasks(self, num_tasks,
                               max_completed = 0.8, # filter out easy tasks
                               min_completed = 0.05, # filter out harder tasks
                               min_rcnt_rate = 0.1, # reward signal generating
    ) -> List[ts.TaskSpec]:
        learnable = []
        for spec_name, stat in self.task_stats.items():
            completion_rate = np.mean(stat["completed"])
            rcnt_over2_rate = np.mean(stat["rcnt_over_2"])
            if completion_rate < max_completed and\
              (completion_rate >= min_completed or rcnt_over2_rate >= min_rcnt_rate):
                learnable.append(self.name_to_spec[spec_name])

        if len(learnable) > num_tasks:
            return list(np.random.choice(learnable, num_tasks))
        return learnable

    def sample_tasks(self, num_tasks,
                     random_ratio = 0.5,
                     reset_sampling_weight = True,
    ) -> List[ts.TaskSpec]:
        task_spec = []
        if 0 <= random_ratio < 1:
            num_learnable = round(num_tasks * (1-random_ratio))
            task_spec = self.get_learnable_tasks(num_learnable)

        # fill in with the randomly-sampled tasks
        # TODO: sample more "less-sampled" tasks (i.e., no or little stats)
        num_sample = num_tasks - len(task_spec)
        task_spec += list(np.random.choice(self.task_spec, num_sample))

        if reset_sampling_weight:
            for single_spec in task_spec:
                single_spec.sampling_weight = 1

        return task_spec
