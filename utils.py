import h5py
import torch
import shutil
import numpy as np
import os

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)

def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth'):
    # Ensure folder exists
    os.makedirs(f"checkpoints/{task_id}", exist_ok=True)

    # Save checkpoint
    full_path = os.path.join("checkpoints", task_id, filename)
    torch.save(state, full_path)

    # Optionally save best model under a distinct name
    if is_best:
        best_path = os.path.join("checkpoints", task_id, f"best_model_{task_id}.pth")
        shutil.copyfile(full_path, best_path)
