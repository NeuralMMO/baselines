# '''Manual test for rollout with baseline agents'''

# import nmmo
# from pufferlib.emulation import Binding

# from lib.team.team_helper import TeamHelper
# from model.realikun.policy import BaselinePolicy
# from model.realikun.baseline_agent import BaselineAgent

# def test_baseline_rollout():
#   model_weights = 'model_weights/realikun.001470.pt'
#   num_teams = 16
#   team_size = 8

#   class ReplayConfig(
#     nmmo.config.Medium,
#     nmmo.config.Terrain,
#     nmmo.config.Resource,
#     nmmo.config.NPC,
#     nmmo.config.Progression,
#     nmmo.config.Equipment,
#     nmmo.config.Item,
#     nmmo.config.Exchange,
#     nmmo.config.Profession,
#     nmmo.config.Combat,
#   ):
#     SAVE_REPLAY = True
#     PROVIDE_ACTION_TARGETS = True
#     PLAYER_N = num_teams * team_size

#   config = ReplayConfig()
#   team_helper = TeamHelper({
#     i: [i*team_size+j+1 for j in range(team_size)]
#     for i in range(num_teams)}
#   )

#   binding = Binding(
#     env_creator=BaselinePolicy.env_creator(config, team_helper),
#     env_name="Neural MMO",
#     suppress_env_prints=False,
#   )

#   agent_list = []
#   for _ in range(num_teams):
#     agent_list.append(BaselineAgent(model_weights, binding))

#   puffer_env = binding.env_creator()
#   team_obs = puffer_env.reset()

#   # Rollout
#   step = 0
#   while puffer_env.done is False:
#     if step == 0:
#       for team_id in team_obs:
#         agent_list[team_id].reset()

#     else:
#       team_obs, _, _, _ = puffer_env.step(actions)

#     # get actions for the next tick
#     actions = {}
#     for team_id, obs in team_obs.items():
#       actions[team_id] = agent_list[team_id].act(obs)

#     step += 1

#   print("Rollout done.")

# if __name__ == '__main__':
#   test_baseline_rollout()
