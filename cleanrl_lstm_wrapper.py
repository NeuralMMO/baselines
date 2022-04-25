from pdb import set_trace as T

import os
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

from config.bases import CleanRL as Config
from scripted import baselines
from neural import policy, io, subnets


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


if __name__ == "__main__":
    config = Config()

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

    # NMMO Integration
    envs = nmmo.integrations.cleanrl_vec_envs(Config, (config.NUM_ENVS + config.NUM_EVAL_ENVS) // config.NENT, config.NUM_CPUS)
    envs = gym.wrappers.RecordEpisodeStatistics(envs)

    agent = Agent(config).cuda()
    agent = torch.nn.DataParallel(agent, device_ids=[0])

    ratings = nmmo.OpenSkillRating(config.AGENTS, baselines.Combat)

    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
    
    # ALGO Logic: Storage setup
    obs = torch.zeros((config.NUM_STEPS, config.NUM_ENVS) + envs.observation_space.shape)
    actions = torch.zeros((config.NUM_STEPS, config.NUM_ENVS) + (config.NUM_ARGUMENTS,))
    logprobs = torch.zeros((config.NUM_STEPS, config.NUM_ENVS))
    rewards = torch.zeros((config.NUM_STEPS, config.NUM_ENVS))
    dones = torch.zeros((config.NUM_STEPS, config.NUM_ENVS))
    values = torch.zeros((config.NUM_STEPS, config.NUM_ENVS))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset())
    next_done = torch.zeros(config.NUM_ENVS + config.NUM_EVAL_ENVS)
    next_lstm_state = agent.module.get_initial_state(config.NUM_ENVS + config.NUM_EVAL_ENVS)
    num_updates = config.TOTAL_TIMESTEPS // config.BATCH_SIZE

    for update in range(1, num_updates + 1):
        initial_lstm_state = (next_lstm_state[0][:config.NUM_ENVS].clone(), next_lstm_state[1][:config.NUM_ENVS].clone())
        # Annealing the rate if instructed to do so.
        if config.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, config.NUM_STEPS):
            global_step += 1 * config.NUM_ENVS
            obs[step] = next_obs[:config.NUM_ENVS]
            dones[step] = next_done[:config.NUM_ENVS]

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent(next_obs, next_lstm_state, next_done)
                values[step] = value[:config.NUM_ENVS].flatten()
            actions[step] = action[:config.NUM_ENVS]
            logprobs[step] = logprob[:config.NUM_ENVS]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            # Training logs
            for e in info[:config.NUM_ENVS]:
                if 'logs' not in e:
                    continue

                stats = {}
                for k, v in e['logs'].items():
                    stats[k] = np.mean(v).item()

                wandb.log(stats)

            # Evaluation logs
            for e in info[config.NUM_ENVS:]:
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


            rewards[step] = torch.tensor(reward)[:config.NUM_ENVS].view(-1)
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
            )[:config.NUM_ENVS].reshape(1, -1).cpu()

            if config.GAE:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(config.NUM_STEPS)):
                    if t == config.NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_done[:config.NUM_ENVS]
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
        b_actions = actions.reshape((-1,) + (Config.NUM_ARGUMENTS,))
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert config.NUM_ENVS % config.NUM_MINIBATCHES == 0
        envsperbatch = config.NUM_ENVS // config.NUM_MINIBATCHES
        envinds = np.arange(config.NUM_ENVS)
        flatinds = np.arange(config.BATCH_SIZE).reshape(config.NUM_STEPS, config.NUM_ENVS)
        clipfracs = []
        for epoch in range(config.UPDATE_EPOCHS):
            np.random.shuffle(envinds)
            for start in range(0, config.NUM_ENVS, envsperbatch):
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

        # Sync policy evaluations
        #ratings = np.load('ratings.npy', allow_pickle=True).item()
        #wandb.log(ratings)

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