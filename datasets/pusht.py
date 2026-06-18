import torch
import einops
import pickle
from pathlib import Path
from typing import Optional
from datasets.core import TrajectoryDataset
import numpy as np
import decord
from decord import VideoReader
decord.bridge.set_bridge("torch")


class PushTDataset(TrajectoryDataset):
    def __init__(
        self,
        data_directory,
        subset_fraction: Optional[float] = None,
        relative=False,
        prefetch: bool = True,
        state_based: bool = False,
        use_sin_cos: bool = False,
        with_velocity: bool = False,
    ):
        self.data_directory = Path(data_directory)
        self.relative = relative
        self.state_based = state_based
        self.use_sin_cos = use_sin_cos
        self.with_velocity = with_velocity
        self.states = torch.load(self.data_directory / "states.pth")
        if relative:
            self.actions = torch.load(self.data_directory / "rel_actions.pth")
        else:
            self.actions = torch.load(self.data_directory / "abs_actions.pth")
        with open(self.data_directory / "seq_lengths.pkl", "rb") as f:
            self.seq_lengths = pickle.load(f)

        self.subset_fraction = subset_fraction
        if self.subset_fraction:
            assert self.subset_fraction > 0 and self.subset_fraction <= 1
            n = int(len(self.states) * self.subset_fraction)
        else:
            n = len(self.states)
        self.states = self.states[:n]
        self.actions = self.actions[:n]
        self.seq_lengths = self.seq_lengths[:n]

        for i in range(n):
            T = self.seq_lengths[i]
            self.actions[i, T:] = 0  # redo zero padding

        if state_based:
            # preprocess states: apply use_sin_cos and with_velocity transforms
            pos = self.states[:, :, :4]  # N T 4
            if use_sin_cos:
                angles = self.states[:, :, 4:5]  # N T 1
                angle_repr = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # N T 2
            else:
                angle_repr = self.states[:, :, 4:5]  # N T 1
            self.states = torch.cat([pos, angle_repr], dim=-1)  # N T (5 or 6)
            if with_velocity:
                velocities = torch.load(self.data_directory / "velocities.pth")
                velocities = velocities[:n]
                self.states = torch.cat([self.states, velocities], dim=-1)  # N T (5+2 or 6+2)

        self.prefetch = prefetch
        if self.prefetch and not state_based:
            self.obses = []
            for i in range(n):
                vid_dir = self.data_directory / "obses"
                reader = VideoReader(str(vid_dir / f"episode_{i:03d}.mp4"), num_threads=1)
                frames = np.arange(len(reader))
                image = reader.get_batch(frames)  # THWC
                obs = image
                self.obses.append(obs)  # THWC

    def get_seq_length(self, idx):
        return self.seq_lengths[idx]

    def get_all_actions(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_states(self):
        result = []
        for i in range(len(self.seq_lengths)):
            T = self.seq_lengths[i]
            result.append(self.states[i, :T])
        return torch.cat(result, dim=0)  # (total_steps, state_dim)

    def get_frames(self, idx, frames):
        if self.state_based:
            obs = self.states[idx][frames]  # T state_dim
            obs = obs.unsqueeze(1)  # T 1 state_dim
        else:
            if self.prefetch:
                obs = self.obses[idx][frames]
            else:
                vid_dir = self.data_directory / "obses"
                reader = VideoReader(str(vid_dir / f"episode_{idx:03d}.mp4"), num_threads=1)
                image = reader.get_batch(frames)  # THWC
                obs = image
            obs = einops.rearrange(obs, "T H W C -> T 1 C H W") / 255.0  # T V C H W, 1 view
        act = self.actions[idx, frames]
        mask = torch.ones(len(act)).bool()
        dummy_goal = torch.ones([obs.shape[0], 1, 1, 1]) # dummy goal, T V P E
        # return obs, act, mask, goal
        return obs, act, dummy_goal

    def __getitem__(self, idx):
        return self.get_frames(idx, range(self.get_seq_length(idx)))

    def __len__(self):
        return len(self.seq_lengths)

def load_pusht_train_val_split(
    data_directory,
    subset_fraction=None,
    relative=False,
    prefetch=True,
    state_based=False,
    use_sin_cos=False,
    with_velocity=False,
):
    train_dataset = PushTDataset(data_directory=data_directory + "/train",
                                subset_fraction=subset_fraction,
                                relative=relative,
                                prefetch=prefetch,
                                state_based=state_based,
                                use_sin_cos=use_sin_cos,
                                with_velocity=with_velocity)
    val_dataset = PushTDataset(data_directory=data_directory + "/val",
                                subset_fraction=subset_fraction,
                                relative=relative,
                                prefetch=prefetch,
                                state_based=state_based,
                                use_sin_cos=use_sin_cos,
                                with_velocity=with_velocity)
    return train_dataset, val_dataset