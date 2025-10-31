import abc
import utils
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Callable


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories.
    TrajectoryDataset[i] returns: (observations, actions, mask)
        observations: Tensor[T, ...], T frames of observations
        actions: Tensor[T, ...], T frames of actions
        mask: Tensor[T]: 0: invalid; 1: valid
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

class TrajectorySlicerDataset(TrajectoryDataset):
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window: int,
        action_window: int,
        vqbet_get_future_action_chunk: bool = True,
        future_conditional: bool = False,
        min_future_sep: int = 0,
        future_seq_len: Optional[int] = None,
        only_sample_tail: bool = False,
        transform: Optional[Callable] = None,
        use_libero_goal: bool = False,
        pad_seq_length: bool = True, # pad actions at end to ensure fixed length
    ):
        if future_conditional:
            assert future_seq_len is not None, "must specify a future_seq_len"
        self.dataset = dataset
        self.window = window
        self.action_window = action_window
        self.vqbet_get_future_action_chunk = vqbet_get_future_action_chunk
        self.future_conditional = future_conditional
        self.min_future_sep = min_future_sep
        self.future_seq_len = future_seq_len
        self.only_sample_tail = only_sample_tail
        self.transform = transform
        self.slices = []
        self.use_libero_goal = use_libero_goal
        self.pad_seq_length = pad_seq_length

        min_seq_length = np.inf
        min_window_required = window + action_window - 1
        for i in range(len(self.dataset)):  # type: ignore
            T = self.dataset.get_seq_length(i)  # avoid reading actual seq (slow)
            min_seq_length = min(T, min_seq_length)

            self.slices += [
                (i, 0, end + 1) for end in range(window - 1)
            ]  # slice indices follow convention [start, end)

            if self.pad_seq_length:
                if T - self.window >= 0:
                    self.slices += [
                        (i, start, start + self.window) for start in range(T - self.window + 1)
                    ]
                else:
                    pass
            else:
                if T - min_window_required < 0:
                    print(f"Ignored short sequence #{i}: len={T}, window={min_window_required}")
                else:
                    self.slices += [
                        (i, start, start + self.window) for start in range(T - min_window_required + 1)
                    ]

        if (not self.pad_seq_length) and (min_seq_length < min_window_required):
            print(
                f"Ignored short sequences. To include all, set window <= {min_seq_length}."
            )

    def get_seq_length(self, idx: int) -> int:
        if self.future_conditional:
            return self.future_seq_len + self.window
        else:
            return self.window

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        i, start, end = self.slices[idx]
        obs, act, *others = self.dataset[i]
        T = obs.shape[0]
        if end - start < self.window:
            obs_win = utils.inference.repeat_start_to_length(
                obs[start:end], self.window, dim=0
            )
            repeated_others = [
                utils.inference.repeat_start_to_length(other[start:end], self.window, dim=0)
                for other in others
            ]
        else:
            obs_win = obs[start:end]
            repeated_others = [other[start:end] for other in others]

        if self.vqbet_get_future_action_chunk:
            expected_len = self.action_window
            a_start = min(end - 1, T - 1)
        else:
            expected_len = self.window + self.action_window - 1
            a_start = start

        a_stop = min(a_start + expected_len, T)
        act_chunk = act[a_start:a_stop]

        if act_chunk.shape[0] < expected_len:
            if self.pad_seq_length:
                act_chunk = utils.inference.repeat_end_to_length(
                    act_chunk, self.action_window, dim=0
                )
            else:
                raise ValueError(
                    f"Action chunk too short: {act_chunk.shape[0]} < {self.action_window}, but pad_seq_length is False"
                )

        values = [obs_win, act_chunk, *repeated_others]

        # optionally apply transform
        if self.transform is not None:
            values = self.transform(values)
        return tuple(values)












