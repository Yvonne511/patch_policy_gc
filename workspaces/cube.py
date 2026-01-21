import utils
import hydra
import torch
import einops
import numpy as np
from workspaces import base
from accelerate import Accelerator
from utils import get_split_idx

OBS_ELEMENT_INDICES = {
    "joint_pos": np.arange(0, 6),
    "joint_vel": np.arange(6, 12),
    "effector_pos": np.arange(12, 15),
    "effector_yaw_cos_sin": np.array([15, 16]),
    "gripper_opening": np.array([17]),
    "gripper_contact": np.array([18]),
}

accelerator = Accelerator()


def calc_state_dist(a, b, obs_element_indices):
    result = {}
    for k, v in obs_element_indices.items():
        # import ipdb; ipdb.set_trace()
        idx = torch.Tensor(v).long()
        result[k] = ((a[idx] - b[idx]) ** 2).mean()
    result["total"] = ((a - b) ** 2).mean()
    
    return result

def mean_dicts(dicts):
    result = {}
    for k in dicts[0].keys():
        result[k] = np.mean([x[k] for x in dicts])
    return result


class CubeWorkspace(base.Workspace):
    def __init__(self, cfg, work_dir):
        super().__init__(cfg, work_dir)
        self.cfg = cfg
        self.obs_element_indices = {
            k: v.copy() for k, v in OBS_ELEMENT_INDICES.items()
        }
        if self.cfg.env.dataset.env_type == 'single':
            # self.obs_element_indices.update({
            #     "cube_pos": np.arange(19, 22),
            #     "cube_quaternion": np.arange(22, 26),
            #     "cube_yaw_cos_sin": np.array([26, 27]),
            # })
            self.obs_element_indices.update({
                "cube_0_state": np.arange(19, 28),
            })
            self.state_subset_idx = np.arange(0, 28)
        elif self.cfg.env.dataset.env_type == 'double':
            # self.obs_element_indices.update({
            #     "cube_0_pos": np.arange(19, 22),
            #     "cube_0_quaternion": np.arange(22, 26),
            #     "cube_0_yaw_cos_sin": np.array([26, 27]),
            #     "cube_1_pos": np.arange(28, 31),
            #     "cube_1_quaternion": np.arange(31, 35),
            #     "cube_1_yaw_cos_sin": np.array([35, 36]),
            # })
            self.obs_element_indices.update({
                "cube_0_state": np.arange(19, 28),
                "cube_1_state": np.arange(28, 37),
            })
            self.state_subset_idx = np.arange(0, 37)
        elif self.cfg.env.dataset.env_type == 'triple':
            # self.obs_element_indices.update({
            #     "cube_0_pos": np.arange(19, 22),
            #     "cube_0_quaternion": np.arange(22, 26),
            #     "cube_0_yaw_cos_sin": np.array([26, 27]),
            #     "cube_1_pos": np.arange(28, 31),
            #     "cube_1_quaternion": np.arange(31, 35),
            #     "cube_1_yaw_cos_sin": np.array([35, 36]),
            #     "cube_2_pos": np.arange(37, 40),
            #     "cube_2_quaternion": np.arange(40, 44),
            #     "cube_2_yaw_cos_sin": np.array([44, 45]),
            # })
            self.obs_element_indices.update({
                "cube_0_state": np.arange(19, 28),
                "cube_1_state": np.arange(28, 37),
                "cube_2_state": np.arange(37, 46),
            })
            self.state_subset_idx = np.arange(0, 46)
        elif self.cfg.env.dataset.env_type == 'quadruple':
            # self.obs_element_indices.update({
            #     "cube_0_pos": np.arange(19, 22),
            #     "cube_0_quaternion": np.arange(22, 26),
            #     "cube_0_yaw_cos_sin": np.array([26, 27]),
            #     "cube_1_pos": np.arange(28, 31),
            #     "cube_1_quaternion": np.arange(31, 35),
            #     "cube_1_yaw_cos_sin": np.array([35, 36]),
            #     "cube_2_pos": np.arange(37, 40),
            #     "cube_2_quaternion": np.arange(40, 44),
            #     "cube_2_yaw_cos_sin": np.array([44, 45]),
            #     "cube_3_pos": np.arange(46, 49),
            #     "cube_3_quaternion": np.arange(49, 53),
            #     "cube_3_yaw_cos_sin": np.array([53, 54]),
            # })
            self.obs_element_indices.update({
                "cube_0_state": np.arange(19, 28),
                "cube_1_state": np.arange(28, 37),
                "cube_2_state": np.arange(37, 46),
                "cube_3_state": np.arange(46, 55),
            })
            self.state_subset_idx = np.arange(0, 55)
        else:
            raise ValueError(f"Unknown env_type: {self.cfg.env.env_type}")


    def _report_result_upon_completion(self, goal_idx=None):
        return {
            "entered": self.env.entered,
            "moved": self.env.moved,
        }

    def run_offline_eval(self):
        train_idx, val_idx = get_split_idx(
            len(self.dataset),
            self.cfg.seed,
            train_fraction=self.cfg.train_fraction,
        )
        embeddings = utils.inference.embed_trajectory_dataset(
            self.encoder, self.dataset
        )
        embeddings = [
            einops.rearrange(x, "T V E -> T (V E)") for x in embeddings
        ]  # flatten views

        states = []


        if self.accelerator.is_main_process:
            states = []
            actions = []
            for i in range(len(self.dataset)):
                T = self.dataset.get_seq_length(i)
                state = self.dataset.states[i, :T]
                state = state[:, self.state_subset_idx]
                states.append(state)
                actions.append(self.dataset.actions[i, :T])
            embd_state_linear_probe_results = (
                utils.inference.linear_probe_with_trajectory_split(
                    embeddings,
                    states,
                    train_idx,
                    val_idx,
                )
            )
            # add prefix to keys
            embd_state_linear_probe_results = {
                f"embd_state_{k}": v for k, v in embd_state_linear_probe_results.items()
            }
            embd_action_linear_probe_results = (
                utils.inference.linear_probe_with_trajectory_split(
                    embeddings,
                    actions,
                    train_idx,
                    val_idx,
                )
            )
            embd_action_linear_probe_results = {
                f"embd_action_{k}": v
                for k, v in embd_action_linear_probe_results.items()
            }

            state_dists = []
            N = 200
            rng = np.random.default_rng(self.cfg.seed)
            for i in range(N):
                query_traj_idx = rng.choice(len(self.dataset))
                query_frame_idx = rng.choice(
                    range(10, self.dataset.get_seq_length(query_traj_idx))
                )
                query_embedding = embeddings[query_traj_idx][query_frame_idx]
                query_frame_state = self.dataset.states[
                    query_traj_idx, query_frame_idx, self.state_subset_idx
                ]

                pool_embeddings = torch.cat(
                    [x for i, x in enumerate(embeddings) if i != query_traj_idx]
                )
                pool_states = torch.cat(
                    [x for i, x in enumerate(states) if i != query_traj_idx]
                )
                _, nn_idx = utils.inference.batch_knn(
                    query_embedding.unsqueeze(0),
                    pool_embeddings,
                    metric=utils.inference.mse,
                    k=1,
                    batch_size=1,
                )
                closest_frame_state = pool_states[nn_idx[0, 0]]
                state_dist = calc_state_dist(query_frame_state, closest_frame_state, self.obs_element_indices)
                state_dists.append(state_dist)
            mean_state_dist = mean_dicts(state_dists)
            return {
                **embd_state_linear_probe_results,
                **embd_action_linear_probe_results,
                **mean_state_dist,
            }
        else:
            return None
