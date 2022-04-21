from pdb import set_trace as T

import argparse
import os
import random
import time
from distutils.util import strtobool

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

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nmmo",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500_000_000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32*Config.NENT,
        help="the number of parallel game environments")
    parser.add_argument("--num-cpus", type=int, default=16,
        help="the number of parallel CPU cores")
    parser.add_argument("--num-steps", type=int, default=512,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=1.0,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=512,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=1,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.3,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=1.0,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    # fmt: on
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input  = io.Input(config,
                embeddings=io.MixedEmbedding,
                attributes=subnets.SelfAttention)
        #self.output = nn.Linear(config.HIDDEN, 4)# 304)
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
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
 
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
            done = torch.zeros(len(x)).to('cuda:0')

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
        for atn in nmmo.Action.edges:
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

class Config(nmmo.config.Medium, nmmo.config.AllGameSystems):
    HIDDEN             = 64
    EMBED              = 64

    NENT = 128

    #Large map pool
    NMAPS = 256

    #Enable task-based
    TASKS              = tasks.All

    @property
    def SPAWN(self):
        return self.SPAWN_CONCURRENT

    #Set a unique path for demo maps
    PATH_MAPS = 'maps/demos'

    #Force terrain generation -- avoids unexpected behavior from caching
    FORCE_MAP_GENERATION = False #True

    NUM_ARGUMENTS = 3

if __name__ == "__main__":
    args = parse_args()

    # WanDB integration                                                       
    with open('wandb_api_key') as key:                                        
        os.environ['WANDB_API_KEY'] = key.read().rstrip('\n')

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb.init(
        project=args.wandb_project_name,
        entity=args.wandb_entity,
        sync_tensorboard=True,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # NMMO Integration
    config = Config()
    envs = nmmo.integrations.cleanrl_vec_envs(Config, args.num_envs // Config.NENT, args.num_cpus)
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    agent = Agent(config).cuda()
    agent = torch.nn.DataParallel(agent, device_ids=[0])

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape)
    actions = torch.zeros((args.num_steps, args.num_envs) + (Config.NUM_ARGUMENTS,))
    logprobs = torch.zeros((args.num_steps, args.num_envs))
    rewards = torch.zeros((args.num_steps, args.num_envs))
    dones = torch.zeros((args.num_steps, args.num_envs))
    values = torch.zeros((args.num_steps, args.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset())
    next_done = torch.zeros(args.num_envs)
    next_lstm_state = agent.module.get_initial_state(args.num_envs)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent(next_obs, next_lstm_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            for e in info:
                if 'logs' not in e:
                    continue

                stats = {}
                for k, v in e['logs'].items():
                    stats[k] = np.mean(v).item()

                wandb.log(stats)

            rewards[step] = torch.tensor(reward).view(-1)
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
            ).reshape(1, -1).cpu()

            if args.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (Config.NUM_ARGUMENTS,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
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
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Sync policy evaluations
        ratings = np.load('ratings.npy', allow_pickle=True).item()
        wandb.log(ratings)

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
