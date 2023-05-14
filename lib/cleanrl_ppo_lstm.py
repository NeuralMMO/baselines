# PufferLib's customized CleanRL PPO + LSTM implementation
# Adapted from https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpoolpy

from collections import defaultdict
from pdb import set_trace as T
import os
import psutil
import random
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pufferlib
import pufferlib.frameworks.cleanrl
import pufferlib.vectorization.multiprocessing
import pufferlib.vectorization.serial

from lib.agent.util import load_matching_state_dict


def train(
        binding,
        agent,
        exp_name=os.path.basename(__file__),
        seed=1,
        torch_deterministic=True,
        cuda=True,
        track=False,
        wandb_project_name='cleanRL',
        wandb_entity=None,
        total_timesteps=10000000,
        learning_rate=2.5e-4,
        num_buffers=1,
        num_envs=8,
        num_agents=1,
        num_cores=psutil.cpu_count(logical=False),
        num_steps=128,
        bptt_horizon=16,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        num_minibatches=4,
        update_epochs=4,
        norm_adv=True,
        clip_coef=0.1,
        clip_vloss=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        epoch_end_callback=None,
        resume_from_path=None,
        run_name=None,
        use_serial_vecenv=False,
    ):
    program_start = time.time()
    env_id = binding.env_name
    args = pufferlib.utils.dotdict(locals())
    args = {k: str(v)[:50] for k,v in args.items()}
    batch_size = int(num_envs * num_agents * num_buffers * num_steps)
    assert num_steps % bptt_horizon == 0, "num_steps must be divisible by bptt_horizon"

    resume_state = None
    wandb_run_id = None
    if resume_from_path is not None:
        print(f"Resuming from from {resume_from_path}...")
        resume_state = torch.load(resume_from_path)
        wandb_run_id = resume_state.get('wandb_run_id')
        print(f"Resuming from run wandb_run_id={wandb_run_id}")

    if run_name is None:
        run_name = f"{env_id}__{exp_name}__{seed}__{int(time.time())}"

    if track:
        import wandb
        wandb_run_id = wandb_run_id or wandb.util.generate_id()

        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            sync_tensorboard=True,
            config=args,
            name=run_name,
            id=wandb_run_id,
            monitor_gym=True,
            save_code=True,
            resume="allow",
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # Note: Must recompute num_envs for multiagent envs
    envs_per_worker = num_envs / num_cores
    assert envs_per_worker == int(envs_per_worker)
    assert envs_per_worker >= 1

    buffers = []

    vec_impl = pufferlib.vectorization.multiprocessing.VecEnv
    if use_serial_vecenv:
        vec_impl = pufferlib.vectorization.serial.VecEnv

    for i in range(num_buffers):
        buffers.append(
                vec_impl(
                    binding,
                    num_workers=num_cores,
                    envs_per_worker=int(envs_per_worker),
                )
        )

    agent = agent.to(device)
    optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    if resume_state is not None:
        upgrade_req = load_matching_state_dict(agent, resume_state['agent_state_dict'])
        if not upgrade_req:
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])

    # ALGO Logic: Storage setup
    obs = torch.zeros((num_steps, num_buffers, num_envs * num_agents) + binding.single_observation_space.shape).to(device)
    actions = torch.zeros((num_steps, num_buffers, num_envs * num_agents) + binding.single_action_space.shape, dtype=int).to(device)
    logprobs = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    rewards = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    dones = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    values = torch.zeros((num_steps, num_buffers, num_envs * num_agents)).to(device)
    env_profiles = [defaultdict(float) for e in range(num_buffers*num_envs)]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    agent_step = 0
    if resume_state is not None:
        global_step = resume_state.get('global_step', 0)
        agent_step = resume_state.get('agent_step', 0)

    next_obs, next_done, next_lstm_state = [], [], []
    for i, envs in enumerate(buffers):
        # envs.async_reset(seed=seed + int(i*num_cores*envs_per_worker*num_agents))
        envs.async_reset()
        o, _, _, info = envs.recv()
        next_obs.append(torch.Tensor(o).to(device))
        next_done.append(torch.zeros((num_envs * num_agents,)).to(device))

        next_lstm_state.append((
            torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, num_envs * num_agents, agent.lstm.hidden_size).to(device),
        ))  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)

    num_updates = total_timesteps // batch_size
    update_from = 0
    if resume_state is not None:
        update_from = resume_state['update']

    for update in range(update_from+1, num_updates + 1):
        epoch_lengths = []
        epoch_returns = []
        epoch_time = time.time()
        epoch_step = 0

        initial_lstm_state = [
            torch.cat([e[0].clone() for e in next_lstm_state], dim=1),
            torch.cat([e[1].clone() for e in next_lstm_state], dim=1)
        ]

        # Annealing the rate if instructed to do so.
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        env_step_time = 0
        inference_time = 0
        for step in range(0, num_steps + 1):
            for buf, envs in enumerate(buffers):
                global_step += num_envs * num_agents

                # TRY NOT TO MODIFY: Receive from game and log data
                if step == 0:
                    obs[step, buf] = next_obs[buf]
                    dones[step, buf] = next_done[buf]
                else:
                    start = time.time()
                    o, r, d, i = envs.recv()
                    env_step_time += time.time() - start

                    next_obs[buf] = torch.Tensor(o).to(device)
                    next_done[buf] = torch.Tensor(d).to(device)

                    if step != num_steps:
                        obs[step, buf] = next_obs[buf]
                        dones[step, buf] = next_done[buf]


                    rewards[step - 1, buf] = torch.tensor(r).float().to(device).view(-1)

                    for item in i:
                        if "episode" in item.keys():
                            er = sum(item["episode"]["r"]) / len(item["episode"]["r"])
                            el = sum(item["episode"]["l"]) / len(item["episode"]["l"])
                            epoch_returns.append(er)
                            epoch_lengths.append(el)
                            writer.add_scalar("charts/episodic_return", er, global_step)
                            writer.add_scalar("charts/episodic_length", el, global_step)
                            print(f"xcxc end of episode. {er} {el} {item.values()}")

                        for agent_info in item.values():
                            if "stats" in agent_info.keys():
                                for name, stat in agent_info["stats"].items():
                                    writer.add_scalar("charts/info/{}/sum".format(name), stat["sum"], global_step)
                                    writer.add_scalar("charts/info/{}/count".format(name), stat["count"], global_step)
                                    print("charts/info/{}/sum".format(name), stat["sum"], global_step)

                if step == num_steps:
                    continue

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    start = time.time()
                    action, logprob, _, value, next_lstm_state[buf] = agent.get_action_and_value(next_obs[buf], next_lstm_state[buf], next_done[buf])
                    #action, logprob, _, value, _ = agent.get_action_and_value(next_obs[buf], None, next_done[buf])
                    inference_time += time.time() - start
                    values[step, buf] = value.flatten()

                actions[step, buf] = action
                logprobs[step, buf] = logprob

                # TRY NOT TO MODIFY: execute the game
                start = time.time()
                envs.send(action.cpu().numpy(), None)
                env_step_time += time.time() - start

        # bootstrap value if not done
        with torch.no_grad():
            for buf in range(num_buffers):
                next_value = agent.get_value(
                    next_obs[buf],
                    next_lstm_state[buf],
                    next_done[buf],
                ).reshape(1, -1)

                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done[buf]
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((num_minibatches, bptt_horizon, -1) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(num_minibatches, bptt_horizon, -1)
        b_actions = actions.reshape((num_minibatches, bptt_horizon, -1) + envs.single_action_space.shape)
        b_dones = dones.reshape(num_minibatches, bptt_horizon, -1)
        b_advantages = advantages.reshape(num_minibatches, bptt_horizon, -1)
        b_returns = returns.reshape(num_minibatches, -1)
        b_values = values.reshape(num_minibatches, -1)

        # Optimizing the policy and value network
        train_time = time.time()
        clipfracs = []
        for epoch in range(update_epochs):
            initial_initial_lstm_state = initial_lstm_state
            for minibatch in range(num_minibatches):
                initial_lstm_state = initial_initial_lstm_state
                _, newlogprob, entropy, newvalue, initial_lstm_state = agent.get_action_and_value(
                    b_obs[minibatch], initial_lstm_state, b_dones[minibatch], b_actions[minibatch])
                initial_lstm_state = (initial_lstm_state[0].detach(), initial_lstm_state[1].detach())
                logratio = newlogprob - b_logprobs[minibatch].reshape(-1)
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[minibatch].reshape(-1)
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[minibatch]) ** 2
                    v_clipped = b_values[minibatch] + torch.clamp(
                        newvalue - b_values[minibatch],
                        -clip_coef,
                        clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[minibatch]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[minibatch]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None:
                if approx_kl > target_kl:
                    break

        mean_reward = float(torch.mean(rewards))
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TIMING: performance metrics to evaluate cpu/gpu usage
        epoch_step += batch_size

        train_time = time.time() - train_time
        epoch_time = time.time() - epoch_time
        extra_time = epoch_time - train_time - env_step_time

        env_sps = int(epoch_step / env_step_time)
        inference_sps = int(epoch_step / inference_time)
        train_sps = int(epoch_step / train_time)
        epoch_sps = max(1, int(epoch_step / epoch_time))

        remaining = timedelta(seconds=int((total_timesteps - global_step) / epoch_sps))
        uptime = timedelta(seconds=int(time.time() - program_start))
        completion_percentage = 100 * global_step / total_timesteps

        if len(epoch_returns) > 0:
            epoch_return = np.mean(epoch_returns)
            epoch_length = int(np.mean(epoch_lengths))
        else:
            epoch_return = 0.0
            epoch_length = 0.0

        print(
            f'Epoch: {update} - Mean Return: {epoch_return:<5.4}, Episode Length: {epoch_length}\n'
            f'\t{completion_percentage:.3}% / {global_step // 1000}K steps - {uptime} Elapsed, ~{remaining} Remaining\n'
            f'\tSteps Per Second: Overall={epoch_sps}, Env={env_sps}, Inference={inference_sps}, Train={train_sps}\n'
        )

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("performance/epoch_sps", epoch_sps, global_step)
        writer.add_scalar("performance/env_sps", env_sps, global_step)
        writer.add_scalar("performance/inference_sps", inference_sps, global_step)
        writer.add_scalar("performance/train_sps", epoch_sps, train_sps)

        writer.add_scalar("performance/env_time", env_step_time, global_step)
        writer.add_scalar("performance/inference_time", inference_time, global_step)
        writer.add_scalar("performance/train_time", train_time, global_step)
        writer.add_scalar("performance/epoch_time", epoch_time, global_step)
        writer.add_scalar("performance/extra_time", extra_time, global_step)

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/reward", mean_reward, global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        profiles = envs.profile()
        prof_deltas = defaultdict(list)
        for env_idx, profile in enumerate(profiles):
            for k, v in profile.items():
                prof_deltas[k].append(v.elapsed - env_profiles[env_idx][k])
                env_profiles[env_idx][k] = v.elapsed

        for k, v in prof_deltas.items():
            writer.add_scalar(f'performance/env/{k}', np.mean(v), global_step)

        if epoch_end_callback is not None:
            epoch_end_callback({
                'update': update,
                'global_step': global_step,
                'agent_step': agent_step,
                'optimizer_state_dict': optimizer.state_dict(),
                'agent_state_dict': agent.state_dict(),
                'wandb_run_id': wandb_run_id,
                "mean_reward": mean_reward * num_steps,
            })

    envs.close()
    writer.close()

    if track:
        wandb.finish()

