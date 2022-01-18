import nmmo

import config


def test_baselines():
    nmmo.Env(config.baselines.Medium()).reset()
    nmmo.Env(config.baselines.Debug()).reset()

def test_competition():
    nmmo.Env(config.competition.CompetitionRound1()).reset()
    nmmo.Env(config.competition.CompetitionRound2()).reset()

def test_neurips():
    nmmo.Env(config.neurips.SmallMultimodalSkills()).reset()
    nmmo.Env(config.neurips.DomainRandomization()).reset()
    nmmo.Env(config.neurips.MagnifyExploration()).reset()
    nmmo.Env(config.neurips.TeamBased()).reset()
