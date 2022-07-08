from pdb import set_trace as T

import os
import sys
import random
import time

import wandb
import gym
import numpy as np
import supersuit as ss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import nmmo
import evaluate
import tasks

from scripted import baselines
from neural import policy, io, subnets


# Switch for Debug mode (fast, low hardware usage)
if len(sys.argv) == 2 and sys.argv[1] == 'debug':
    from config.cleanrl import DebugEval as Eval
    from config.cleanrl import Debug as Train
else:
    from config.cleanrl import Eval
    from config.cleanrl import Train

class Agent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input  = io.Input(config,
                embeddings=io.MixedEmbedding,
                attributes=subnets.SelfAttention)
        self.output = io.Output(config)
        self.value  = nn.Linear(config.HIDDEN, 1)
        self.policy = policy.Simple(config)

        self.lstm = nn.LSTM(config.HIDDEN, config.HIDDEN)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

    def get_initial_state(self, batch, device='cpu'):
        return (
            torch.zeros(batch, self.lstm.num_layers, self.lstm.hidden_size).to(device),
            torch.zeros(batch, self.lstm.num_layers, self.lstm.hidden_size).to(device),
        )
 
    def _compute_hidden(self, x, lstm_state, done):
        x         = nmmo.emulation.unpack_obs(self.config, x)
        lookup    = self.input(x)
        hidden, _ = self.policy(lookup)

        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        return new_hidden, lookup, lstm_state

    def forward(self, x, lstm_state, done=None, action=None, value_only=False):
        if done is None:
            done = torch.zeros(len(x)).to(lstm_state[0].device)

        lstm_state = (lstm_state[0].transpose(0, 1), lstm_state[1].transpose(0, 1))

        if value_only:
            return self.get_value(x, lstm_state, done)
        action, logprob, entropy, value, lstm_state = self.get_action_and_value(x, lstm_state, done, action)
        lstm_state = (lstm_state[0].transpose(0, 1), lstm_state[1].transpose(0, 1))
        return action, logprob, entropy, value, lstm_state

    def get_value(self, x, lstm_state, done):
        x, _, _ = self._compute_hidden(x, lstm_state, done)
        return self.value(x)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        x, lookup, lstm_state = self._compute_hidden(x, lstm_state, done)
        logits                = self.output(x, lookup)
        value                 = self.value(x)

        flat_logits = []
        for atn in nmmo.Action.edges(self.config):
            for arg in atn.edges:
                flat_logits.append(logits[atn][arg]) 

        mulit_categorical = [Categorical(logits=l) for l in flat_logits]

        if action is None:
            action = torch.stack([c.sample() for c in mulit_categorical])
        else:
            action = action.view(-1, action.shape[-1]).T

        logprob = torch.stack([c.log_prob(a) for c, a in zip(mulit_categorical, action)]).T
        entropy = torch.stack([c.entropy() for c in mulit_categorical]).T

        return action.T, logprob.sum(1), entropy.sum(1), value, lstm_state


if __name__ == "__main__":
    config        = Train()
    eval_config   = Eval()

    # WanDB integration                                                       
    with open('wandb_api_key') as key:                                        
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')

    run_name = f"{config.ENV_ID}__{config.EXP_NAME}__{config.SEED}__{int(time.time())}"
    wandb.init(
        project=config.WANDB_PROJECT_NAME,
        entity=config.WANDB_ENTITY,
        sync_tensorboard=True,
        config=vars(config),
        name=run_name,
        monitor_gym=True,
        save_code=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = config.TORCH_DETERMINISTIC

    # This is part of the new (not yet documented) integrations API that makes NMMO
    # look like a simple environment from the perspective of infra frameworks while
    # actually maintaining all the same internal complexity. For now, just pass it a config
    # Note that it relies on config.NUM_CPUS and config.PLAYER_N to define scale
    # Agent increment per env (i.e. 4, 8, ..., 128)
    step = config.PLAYER_N / config.NUM_CPUS
    assert step == int(step)
    step = int(step)
    env_configs = []
    NUM_ENVS = config.NUM_CPUS * (config.PLAYER_N + step) #Number of training envs

    # Gross hack to make all the env templates picklable
    for core in range(config.NUM_CPUS):
        n1 = f'TrainSubenv_{core}_1'
        n2 = f'TrainSubenv_{core}_1'

        env_configs.append(type(n1, (Train, ), {'NUM_CPUS': 1, 'PLAYER_N': step*(core+1)}))
        env_configs.append(type(n2, (Train, ), {'NUM_CPUS': 1, 'PLAYER_N': step*(config.NUM_CPUS - core)}))

        globals()[n1] = env_configs[-2]
        globals()[n2] = env_configs[-1]

    env_configs.append(Eval)
    envs = nmmo.integrations.cleanrl_vec_envs(env_configs)

    agent = Agent(config)
    if config.CUDA:
        agent = torch.nn.DataParallel(agent.cuda(), device_ids=config.CUDA)

    ratings = nmmo.OpenSkillRating(eval_config.PLAYERS, baselines.Combat)

    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
    
    # ALGO Logic: Storage setup
    BATCH_SIZE = NUM_ENVS * config.NUM_STEPS

    obs = torch.zeros((config.NUM_STEPS, NUM_ENVS) + envs.observation_space.shape)
    actions = torch.zeros((config.NUM_STEPS, NUM_ENVS) + (config.NUM_ARGUMENTS,))
    logprobs = torch.zeros((config.NUM_STEPS, NUM_ENVS))
    rewards = torch.zeros((config.NUM_STEPS, NUM_ENVS))
    dones = torch.zeros((config.NUM_STEPS, NUM_ENVS))
    values = torch.zeros((config.NUM_STEPS, NUM_ENVS))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset())
    next_done = torch.zeros(NUM_ENVS + eval_config.NUM_ENVS)

    if config.CUDA:
        next_lstm_state = agent.module.get_initial_state(NUM_ENVS + eval_config.NUM_ENVS)
    else:
        next_lstm_state = agent.get_initial_state(NUM_ENVS + eval_config.NUM_ENVS)

    num_updates = config.TOTAL_TIMESTEPS // BATCH_SIZE
    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0][:NUM_ENVS].clone(), next_lstm_state[1][:NUM_ENVS].clone())
        # Annealing the rate if instructed to do so.
        if config.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.NUM_STEPS):
            global_step += NUM_ENVS
            obs[step] = next_obs[:NUM_ENVS]
            dones[step] = next_done[:NUM_ENVS]

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent(next_obs, next_lstm_state, next_done)
                values[step] = value[:NUM_ENVS].flatten()
            actions[step] = action[:NUM_ENVS]
            logprobs[step] = logprob[:NUM_ENVS]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            # Training logs
            for e in info[:NUM_ENVS]:
                if 'logs' not in e:
                    continue

                stats = {}
                for k, v in e['logs'].items():
                    stats[k] = np.mean(v).item()

                wandb.log(stats)

            # Evaluation logs
            for e in info[NUM_ENVS:]:
                if 'logs' not in e:
                    continue

                stats = {}
                for k, v in e['logs'].items():
                    stats['evaluation_' + k] = np.mean(v).item()

                ratings.update(
                    policy_ids=e['logs']['PolicyID'],
                    scores=e['logs']['Task_Reward'])

                stats = {**stats, **ratings.stats}
                wandb.log(stats)


            rewards[step] = torch.tensor(reward)[:NUM_ENVS].view(-1)
            next_obs, next_done = torch.Tensor(next_obs), torch.Tensor(done)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent(
                next_obs,
                next_lstm_state,
                next_done,
                value_only=True,
            )[:NUM_ENVS].reshape(1, -1).cpu()

            if config.GAE:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(config.NUM_STEPS)):
                    if t == config.NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done[:NUM_ENVS]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config.GAMMA * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config.GAMMA * config.GAE_LAMBDA * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(config.NUM_STEPS)):
                    if t == config.NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + config.GAMMA * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (Train.NUM_ARGUMENTS,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert NUM_ENVS % config.NUM_MINIBATCHES == 0
        envsperbatch = NUM_ENVS // config.NUM_MINIBATCHES
        envinds = np.arange(NUM_ENVS)
        flatinds = np.arange(BATCH_SIZE).reshape(config.NUM_STEPS, NUM_ENVS)
        clipfracs = []
        for epoch in range(config.UPDATE_EPOCHS):
            np.random.shuffle(envinds)
            for start in range(0, NUM_ENVS, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _ = agent(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][mbenvinds, :], initial_lstm_state[1][mbenvinds, :]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )

                # Must be done on CPU for multiGPU
                newlogprob = newlogprob.cpu()
                entropy    = entropy.cpu()
                newvalue   = newvalue.cpu()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.CLIP_COEF).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.NORM_ADV:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.CLIP_COEF, 1 + config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -config.CLIP_COEF,
                        config.CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ENT_COEF * entropy_loss + v_loss * config.VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

            if config.TARGET_KL is not None:
                if approx_kl > config.TARGET_KL:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f'Update: {update}, SPS: {int(global_step / (time.time() - start_time))}')
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        torch.save(agent.state_dict(), 'model.pt')

    envs.close()
    writer.close()

import rllib_wrapper as wrapper
import config as base_config
from config import scale


class ConsoleLog(CLIReporter):
   def report(self, trials, done, *sys_info):
      os.system('cls' if os.name == 'nt' else 'clear') 
      print(nmmo.motd + '\n')
      super().report(trials, done, *sys_info)


def run_tune_experiment(config, trainer_wrapper, rllib_env=wrapper.RLlibEnv):
   '''Ray[RLlib, Tune] integration for Neural MMO

   Setup custom environment, observations/actions, policies,
   and parallel training/evaluation'''

   #Round and round the num_threads flags go
   #Which are needed nobody knows!
   torch.set_num_threads(1)
   os.environ['MKL_NUM_THREADS']     = '1'
   os.environ['OMP_NUM_THREADS']     = '1'
   os.environ['NUMEXPR_NUM_THREADS'] = '1'
 
   ray.init(local_mode=config.LOCAL_MODE)

   #Register custom env and policies
   ray.tune.registry.register_env("Neural_MMO",
         lambda config: rllib_env(config))

   rllib.models.ModelCatalog.register_custom_model(
         'godsword', wrapper.RLlibPolicy)

   mapPolicy = lambda agentID : 'policy_{}'.format(
         agentID % config.NPOLICIES)

   policies = {}
   env = nmmo.Env(config)
   for i in range(config.NPOLICIES):
      params = {
            "agent_id": i,
            "obs_space_dict": env.observation_space(i),
            "act_space_dict": env.action_space(i)}
      key           = mapPolicy(i)
      policies[key] = (None, env.observation_space(i), env.action_space(i), params)

   #Evaluation config
   eval_config = deepcopy(config)
   eval_config.EVALUATE = True
   eval_config.PLAYERS  = eval_config.EVAL_PLAYERS

   trainer_cls, extra_config = trainer_wrapper(config)

   #Create rllib config
   rllib_config = {
      'num_workers': config.NUM_WORKERS,
      'num_gpus_per_worker': config.NUM_GPUS_PER_WORKER,
      'num_gpus': config.NUM_GPUS,
      'num_envs_per_worker': 1,
      'train_batch_size': config.TRAIN_BATCH_SIZE,
      'rollout_fragment_length': config.ROLLOUT_FRAGMENT_LENGTH,
      'num_sgd_iter': config.NUM_SGD_ITER,
      'framework': 'torch',
      'horizon': np.inf,
      'soft_horizon': False, 
      'no_done_at_end': False,
      'env': 'Neural_MMO',
      'env_config': {
         'config': config
      },
      'evaluation_config': {
         'env_config': {
            'config': eval_config
         },
      },
      'multiagent': {
         'policies': policies,
         'policy_mapping_fn': mapPolicy,
         'count_steps_by': 'agent_steps'
      },
      'model': {
         'custom_model': 'godsword',
         'custom_model_config': {'config': config},
         'max_seq_len': config.LSTM_BPTT_HORIZON
      },
      'render_env': config.RENDER,
      'callbacks': wrapper.RLlibLogCallbacks,
      'evaluation_interval': config.EVALUATION_INTERVAL,
      'evaluation_num_episodes': config.EVALUATION_NUM_EPISODES,
      'evaluation_num_workers': config.EVALUATION_NUM_WORKERS,
      'evaluation_parallel_to_training': config.EVALUATION_PARALLEL,
   }

   #Alg-specific params
   rllib_config = {**rllib_config, **extra_config}
 
   restore     = None
   config_name = config.__class__.__name__
   algorithm   = trainer_cls.name()
   if config.RESTORE:
      if config.RESTORE_ID:
         config_name = '{}_{}'.format(config_name, config.RESTORE_ID)

      restore   = '{0}/{1}/{2}/checkpoint_{3:06d}/checkpoint-{3}'.format(
            config.EXPERIMENT_DIR, algorithm, config_name, config.RESTORE_CHECKPOINT)

   callbacks = []
   wandb_api_key = 'wandb_api_key'
   if os.path.exists(wandb_api_key):
       callbacks=[WandbLoggerCallback(
               project = 'NeuralMMO-v1.6',
               api_key_file = 'wandb_api_key',
               log_config = False)]
   else:
       print('Running without WanDB. Create a file baselines/wandb_api_key and paste your API key to enable')

   tune.run(trainer_cls,
      config    = rllib_config,
      name      = trainer_cls.name(),
      verbose   = config.LOG_LEVEL,
      stop      = {'training_iteration': config.TRAINING_ITERATIONS},
      restore   = restore,
      resume    = config.RESUME,
      local_dir = config.EXPERIMENT_DIR,
      keep_checkpoints_num = config.KEEP_CHECKPOINTS_NUM,
      checkpoint_freq = config.CHECKPOINT_FREQ,
      checkpoint_at_end = True,
      trial_dirname_creator = lambda _: config_name,
      progress_reporter = ConsoleLog(),
      reuse_actors = True,
      callbacks=callbacks,
      )


class CLI():
   '''Neural MMO CLI powered by Google Fire

   Main file for the RLlib demo included with Neural MMO.

   Usage:
      python main.py <COMMAND> --config=<CONFIG> --ARG1=<ARG1> ...

   The User API documents core env flags. Additional config options specific
   to this demo are available in projekt/config.py. 

   The --config flag may be used to load an entire group of options at once.
   Select one of the defaults from projekt/config.py or write your own.
   '''
   def __init__(self, **kwargs):
      if 'help' in kwargs:
         return 

      config = 'baselines.Medium'
      if 'config' in kwargs:
          config = kwargs.pop('config')

      config = attrgetter(config)(base_config)()
      config.override(**kwargs)

      if 'scale' in kwargs:
          config_scale = kwargs.pop('scale')
          config = getattr(scale, config_scale)()
          config.override(config_scale)

      assert hasattr(config, 'NUM_GPUS'), 'Missing NUM_GPUS (did you specify a scale?)'
      assert hasattr(config, 'NUM_WORKERS'), 'Missing NUM_WORKERS (did you specify a scale?)'
      assert hasattr(config, 'EVALUATION_NUM_WORKERS'), 'Missing EVALUATION_NUM_WORKERS (did you specify a scale?)'
      assert hasattr(config, 'EVALUATION_NUM_EPISODES'), 'Missing EVALUATION_NUM_EPISODES (did you specify a scale?)'

      self.config = config
      self.trainer_wrapper = wrapper.PPO

   def generate(self, **kwargs):
      '''Manually generates maps using the current --config setting'''
      nmmo.MapGenerator(self.config).generate_all_maps()

   def train(self, **kwargs):
      '''Train a model using the current --config setting'''
      run_tune_experiment(self.config, self.trainer_wrapper)

   def evaluate(self, **kwargs):
      '''Evaluate a model against EVAL_PLAYERS models'''
      self.config.TRAINING_ITERATIONS     = 0
      self.config.EVALUATE                = True
      self.config.EVALUATION_NUM_WORKERS  = self.config.NUM_WORKERS
      self.config.EVALUATION_NUM_EPISODES = self.config.NUM_WORKERS

      run_tune_experiment(self.config, self.trainer_wrapper)

   def render(self, **kwargs):
      '''Start a WebSocket server that autoconnects to the 3D Unity client'''
      self.config.RENDER                  = True
      self.config.NUM_WORKERS             = 1
      self.evaluate(**kwargs)


if __name__ == '__main__':
   def Display(lines, out):
        text = "\n".join(lines) + "\n"
        out.write(text)

   from fire import core
   core.Display = Display
   Fire(CLI)
