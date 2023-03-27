from typing import Dict, Union, List

import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torchtyping import TensorType

from model.level import Level


class PrioritizedReplayDistribution:
    """Prioritize Replay Distribution for seen training levels."""
    def __init__(
        self,
        staleness_coeff: float = 0.1,
        temperature: float = 0.1, # the beta coefficient for the score-prioritized distribution
    ) -> None:
        self.staleness_coeff = staleness_coeff
        self.temperature = temperature

    def create(
        self,
        score_levels: Dict[Level, Union[int, float]],
        last_episode_levels: Dict[Level, int], # the last episode that each level was played,
        last_episode: int # the last episode
    ) -> TensorType["n_visited_levels"]:
        """Create a prioritized level distribution."""

        score_levels = torch.tensor([v for v in score_levels.values()])
        last_episode_levels = torch.tensor([v for v in last_episode_levels.values()])

        level_scores = torch.pow(
            input=F.normalize(score_levels, dim=-1),
            exponent=1/self.temperature
        )
        score_dist = level_scores / level_scores.sum(dim=-1)

        stale_scores = last_episode - last_episode_levels
        stale_dist = stale_scores / stale_scores.sum(dim=-1)

        prioritized_dist = (1 - self.staleness_coeff) * score_dist + self.staleness_coeff * stale_dist

        return prioritized_dist


class PrioritizedReplay:
    def __init__(
        self,
        levels: List[Level],
    ) -> None:
        self.levels = levels
        self.visited_count_levels: Dict[Level, int] = {}

        self.prioritized_dist = PrioritizedReplayDistribution()

    def sample_next_level(
        self,
        visited_levels: List[Level],
        score_levels: Dict[str, Union[int, float]],
        last_episode_levels: Dict[str, int],
        last_episode: int
    ) -> Level:
        """Sampling a level from the replay distribution."""
        decision_dist = Bernoulli(probs=0.5)
        unseen_levels = [level for level in self.levels if level not in visited_levels]

        if decision_dist.sample() == 0 and len(unseen_levels) > 0:
            # sample an unseen level
            uniform_dist = torch.rand(len(unseen_levels))
            selected_index = torch.argmax(uniform_dist)
            next_level = unseen_levels[selected_index]

            self.visited_count_levels[next_level] = 1
        else:
            # sample a level for replay
            prioritized_dist = self.prioritized_dist.create(
                score_levels,
                last_episode_levels,
                last_episode
            )
            visited_idx = torch.multinomial(prioritized_dist, num_samples=1)
            next_level = visited_levels[visited_idx]

        return next_level
