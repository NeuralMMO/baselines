from pdb import set_trace as T
import numpy as np
import torch

tensor = torch.Tensor([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1],
    ])

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

trajectories, traj_lens = pad_to_pack(tensor, tensor)
print(trajectories)
print(traj_lens)
