from pdb import set_trace as T

import os
import sys
import random
import time
import random

from collections import defaultdict

import wandb
import gym
import numpy as np
import supersuit as ss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import rnn

import nmmo
import evaluate
import tasks

from scripted import baselines
from neural import policy, io, subnets

### Switch for Debug mode (fast, low hardware usage)
if len(sys.argv) > 1 and sys.argv[1] == 'debug':
    from config.cleanrl import Debug as Train
    from config.cleanrl import DebugEval as Eval
else:
    from config.cleanrl import Train
    from config.cleanrl import Eval

def pack_indices(dones):
    steps, ents = dones.shape
    flat_indices, traj_lens = [], []
    idx = 0
    for ent in range(ents):
        read_zero = False
        traj = []

        for step in range(steps):
            # Got a step of agent data
            if dones[step][ent] == 0:
                traj.append(idx)
                read_zero = True

            #Trajectory bound
            elif dones[step][ent] == 1 and read_zero:
                traj_lens.append(len(traj))
                flat_indices += traj

                read_zero = False
                traj = []

            idx += 1

        if read_zero:
            traj_lens.append(len(traj))
            flat_indices += traj

    assert len(flat_indices) == sum(traj_lens), f'{len(trajectories)}, {sum(traj_lens)}'
    return flat_indices, traj_lens


def pad_to_pack(tensor, dones):
    steps, ents = dones.shape
    trajectories = []
    traj_lens = []
    for ent in range(ents):
        read_zero = False
        traj = []

        for step in range(steps):
            # Got a step of agent data
            if dones[step][ent] == 0:
                traj.append(tensor[step][ent])
                read_zero = True

            #Trajectory bound
            elif dones[step][ent] == 1 and read_zero:
                traj_lens.append(len(traj))
                trajectories += traj

                read_zero = False
                traj = []

        if read_zero:
            traj_lens.append(len(traj))
            trajectories += traj

    assert len(trajectories) == sum(traj_lens), f'{len(trajectories)}, {sum(traj_lens)}'
    return torch.stack(trajectories), traj_lens


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

        self.lstm = subnets.RaggedLSTM(config.HIDDEN, config.HIDDEN)
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
 
    def _compute_hidden(self, x, lstm_state, lens):
        x         = nmmo.emulation.unpack_obs(self.config, x)
        lookup    = self.input(x)
        hidden, _ = self.policy(lookup)

        new_hidden, lstm_state = self.lstm(hidden, lstm_state, lens)
        return new_hidden, lookup, lstm_state

    def forward(self, x, lstm_state, action=None, value_only=False, lens=None):
        lstm_state = (lstm_state[0].transpose(0, 1), lstm_state[1].transpose(0, 1))

        if value_only:
            return self.get_value(x, lstm_state)
        action, logprob, entropy, value, lstm_state = self.get_action_and_value(x, lstm_state, action, lens)
        lstm_state = (lstm_state[0].transpose(0, 1), lstm_state[1].transpose(0, 1))
        return action, logprob, entropy, value, lstm_state

    def get_value(self, x, lstm_state, lens=None):
        x, _, _ = self._compute_hidden(x, lstm_state, lens)
        return self.value(x)

    def get_action_and_value(self, x, lstm_state, action=None, lens=None):
        x, lookup, lstm_state = self._compute_hidden(x, lstm_state, lens)
        logits                = self.output(x, lookup)
        value                 = self.value(x)

        flat_logits = []
        for atn in nmmo.Action.edges:
            for arg in atn.edges:
                flat_logits.append(logits[atn][arg]) 

        try:
            mulit_categorical = [Categorical(logits=l) for l in flat_logits]
        except:
            T()

        if action is None:
            action = torch.stack([c.sample() for c in mulit_categorical])
        else:
            action = action.view(-1, action.shape[-1]).T

        logprob = torch.stack([c.log_prob(a) for c, a in zip(mulit_categorical, action)]).T
        entropy = torch.stack([c.entropy() for c in mulit_categorical]).T

        return action.T, logprob.sum(1), entropy.sum(1), value, lstm_state


if __name__ == "__main__":
    config      = Train()

    #DISABLE FOR NOW DUE TO SS BUG
    #eval_config = Eval()
    class eval_config:
        NUM_ENVS = 0

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

    # Create environment and agents -- DISABLE EVAL FOR NOW DUE TO SS BUG
    # This is part of the new (not yet documented) integrations API that makes NMMO
    # look like a simple environment from the perspective of infra frameworks while
    # actually maintaining all the same internal complexity. For now, just pass it a config
    # Note that it relies on config.NUM_CPUS and config.NENT to define scale
    envs = nmmo.integrations.cleanrl_vec_envs(Train)#, Eval)

    agent = Agent(config)
    #agent.load_state_dict({k.lstrip('module')[1:]: v for k, v in torch.load('model_flatlr.pt').items()})
    if config.CUDA:
        agent = agent.to('cuda:1')
        agent = torch.nn.DataParallel(agent, device_ids=config.CUDA)

    #ratings = nmmo.OpenSkillRating(eval_config.AGENTS, baselines.Combat)

    optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE, eps=1e-5)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset())
    next_done = torch.zeros(config.NUM_ENVS)
    if config.CUDA:
        next_lstm_state = agent.module.get_initial_state(config.NUM_ENVS + eval_config.NUM_ENVS)
    else:
        next_lstm_state = agent.get_initial_state(config.NUM_ENVS + eval_config.NUM_ENVS)

    resets = [0 for _ in range(config.NUM_ENVS)]
    num_updates = config.TOTAL_TIMESTEPS // config.BATCH_SIZE
    for update in range(1, num_updates + 1):
        env_keys = [f'env_{i}_reset_{r}' for i, r in enumerate(resets)]

        obs = defaultdict(list)
        actions = defaultdict(list)
        logprobs = defaultdict(list)
        rewards = defaultdict(list)
        dones = defaultdict(list)
        values = defaultdict(list)
        advantages = defaultdict(list)
     
        initial_lstm_state = {f'env_{idx}_reset_{resets[idx]}': [next_lstm_state[0][idx], next_lstm_state[1][idx]] for idx in range(len(next_obs))}
 
        # Annealing the rate if instructed to do so.
        if config.ANNEAL_LR:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config.LEARNING_RATE
            optimizer.param_groups[0]["lr"] = lrnow

        samples_collected = 0
        while samples_collected < config.BATCH_SIZE:
            print('Sample: ', samples_collected)
            agent_steps = len(next_done) - sum(next_done)
            global_step += agent_steps
            samples_collected += agent_steps

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent(next_obs, next_lstm_state)

            assert len(next_done) == config.NUM_ENVS
            for idx, done in enumerate(next_done):
                key = f'env_{idx}_reset_{resets[idx]}'
                obs[key].append(next_obs[idx])
                dones[key].append(next_done[idx])
                values[key].append(value[idx].cpu())
                actions[key].append(action[idx])
                logprobs[key].append(logprob[idx])

                if key not in initial_lstm_state:
                    initial_lstm_state[key] = [next_lstm_state[0][idx], next_lstm_state[1][idx]]

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())

            for idx, done in enumerate(next_done):
                key = f'env_{idx}_reset_{resets[idx]}'
                rewards[key].append(reward[idx])
                if done:
                    resets[idx] += 1

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


            next_obs = torch.Tensor(next_obs)
            next_done = torch.tensor([i['done'] for i in info])[:config.NUM_ENVS]

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
                value_only=True,
            )[:config.NUM_ENVS].cpu()

            for idx, done in enumerate(next_done):
                key = f'env_{idx}_reset_{resets[idx]}'
                if done:
                    continue
 
                dones[key].append(next_done[idx])
                values[key].append(next_value[idx])

            if config.GAE:
                lastgaelam = 0
                returns = defaultdict(list)
                for idx, traj in enumerate(obs.keys()):
                    traj_len = len(obs[traj])
                    advantages[traj] = torch.zeros(traj_len)
                    returns[traj] = torch.zeros(traj_len)
                    for t in reversed(range(traj_len-1)):
                        nextnonterminal = 1.0 - dones[traj][t + 1].item()
                        nextvalues = values[traj][t + 1]
                        delta = rewards[traj][t] + config.GAMMA * nextvalues * nextnonterminal - values[traj][t]
                        advantages[traj][t] = lastgaelam = delta + config.GAMMA * config.GAE_LAMBDA * nextnonterminal * lastgaelam
                        returns[traj][t] = advantages[traj][t] + values[traj][t]
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(config.NUM_STEPS)):
                    if t == config.NUM_STEPS - 1:
                        nextnonterminal = 1.0 - next_ent_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - ent_dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + config.GAMMA * nextnonterminal * next_return
                advantages = returns - values

        clipfracs = []
        for epoch in range(config.UPDATE_EPOCHS):
            shuffled_keys = random.sample(env_keys, len(env_keys))

            b_obs = torch.cat([torch.stack(obs[k]) for k in shuffled_keys])
            b_logprobs = torch.cat([torch.stack(logprobs[k]) for k in shuffled_keys])
            b_actions = torch.cat([torch.stack(actions[k]) for k in shuffled_keys])
            b_dones = torch.cat([torch.stack(dones[k]) for k in shuffled_keys])
            b_values = torch.cat([torch.stack(values[k]) for k in shuffled_keys])

            b_initial_h = torch.stack([initial_lstm_state[k][0] for k in shuffled_keys])
            b_initial_c = torch.stack([initial_lstm_state[k][1] for k in shuffled_keys])

            b_advantages = torch.cat([advantages[k] for k in shuffled_keys])
            b_returns = torch.cat([returns[k] for k in shuffled_keys])

            advantage_mean, advantage_std = b_advantages.mean(), b_advantages.std()

            traj_lens = [len(obs[k]) for k in shuffled_keys]
            cum_sum = np.cumsum(traj_lens)

            start_sample = 0
            start_traj   = 0
            while start_sample < len(b_obs):
                print('Optimize: ', start_sample)
                end_traj = np.argmax(cum_sum > (start_sample + config.MINIBATCH_SIZE))
                if end_traj == 0:
                    end_traj = len(traj_lens)
                end_sample = cum_sum[end_traj - 1]

                _, newlogprob, entropy, newvalue, _ = agent(
                    b_obs[start_sample:end_sample],
                    [b_initial_h[start_traj:end_traj], b_initial_c[start_traj:end_traj]],
                    b_actions[start_sample:end_sample],
                    lens = traj_lens[start_traj:end_traj],
                  )

                # Perform loss computation on CPU. DataParallel will correctly bptt to GPU.
                newlogprob    = newlogprob.cpu()
                entropy       = entropy.cpu().mean()
                newvalue      = newvalue.view(-1).cpu()
                mb_values     = b_values[start_sample:end_sample].cpu()
                mb_logprobs   = b_logprobs[start_sample:end_sample].cpu()
                mb_advantages = b_advantages[start_sample:end_sample]#.cpu()
                mb_returns    = b_returns[start_sample:end_sample]#.cpu()
 
                logratio = newlogprob - mb_logprobs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config.CLIP_COEF).float().mean().item()]

                if config.NORM_ADV:
                    mb_advantages = (mb_advantages - advantage_mean) / (advantage_std + 1e-8)
                    #mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.CLIP_COEF, 1 + config.CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values + torch.clamp(
                        newvalue - mb_values,
                        -config.CLIP_COEF,
                        config.CLIP_COEF,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                loss = pg_loss - config.ENT_COEF * entropy + config.VF_COEF * v_loss

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(agent.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

                start_sample = end_sample
                start_traj = end_traj

            if config.TARGET_KL is not None:
                if approx_kl > config.TARGET_KL:
                    break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()

        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        #writer.add_scalar("charts/data_frac", b_data_frac, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(f'Update: {update}, SPS: {int(global_step / (time.time() - start_time))}')
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        torch.save(agent.state_dict(), 'model_test.pt')

    envs.close()
    writer.close()


