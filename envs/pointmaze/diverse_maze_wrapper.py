import numpy as np
import torch
import gym
import gym.spaces
from env.pointmaze.maze_model import MazeEnv
from utils import aggregate_dct

DEFAULT_MAPS_PATH = (
    "/scratch/yw4142/wm/HWM_PLDM/pldm_envs/diverse_maze/datasets"
    "/maze2d_large_diverse_25maps/train_maps.pt"
)

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])


class DiversePointMazeWrapper(gym.Env):
    """
    Pool of MazeEnv instances, one per map layout.
    State is 5D: (x, y, vx, vy, map_idx).
    prepare() routes to the correct sub-env based on init_state[4].
    """

    def __init__(
        self,
        maps_path: str = DEFAULT_MAPS_PATH,
        state_based: bool = False,
        reward_type: str = "sparse",
        reset_target: bool = False,
        ref_min_score: float = 23.85,
        ref_max_score: float = 161.86,
        dataset_url: str = "http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-umaze-sparse-v1.hdf5",
        **kwargs,
    ):
        super().__init__()
        maps = torch.load(maps_path)
        self.maps = {int(k): v for k, v in maps.items()}
        self.n_maps = len(self.maps)
        self.state_based = state_based
        return_value = "state" if state_based else "obs"

        self.sub_envs: dict = {}
        for map_idx, map_key in self.maps.items():
            self.sub_envs[map_idx] = MazeEnv(
                maze_spec=map_key,
                return_value=return_value,
                reward_type=reward_type,
                reset_target=reset_target,
                ref_min_score=ref_min_score,
                ref_max_score=ref_max_score,
                dataset_url=dataset_url,
                **kwargs,
            )

        self.active_map_idx: int = 0
        self.active_env: MazeEnv = self.sub_envs[0]
        self._render_ready: set = set()

        self.action_space = self.active_env.action_space
        self.action_dim = self.action_space.shape[0]
        obs_dim = 5  # (x, y, vx, vy, map_idx)
        self.observation_space = (
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            if state_based
            else gym.spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8)
        )

    # ------------------------------------------------------------------ gym API
    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        state_5d = np.append(info["state"], float(self.active_map_idx)).astype(np.float32)
        info["state"] = state_5d
        if self.state_based:
            obs["visual"] = state_5d
            obs["proprio"] = state_5d
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs, state_4d = self.active_env.reset()
        state_5d = np.append(state_4d, float(self.active_map_idx)).astype(np.float32)
        if self.state_based:
            obs["visual"] = state_5d
            obs["proprio"] = state_5d
        return obs, state_5d

    # ------------------------------------------------------------------ interface
    def sample_random_init_goal_states(self, seed, fix_goal=False):
        """
        Return two random states: one as the initial state and one as the goal state.
        """
        # TODO: fix_goal not used
        rs = np.random.RandomState(seed)
        map_idx = rs.randint(0, self.n_maps)
        self.active_map_idx = map_idx
        self.active_env = self.sub_envs[map_idx]

        def generate_state():
            if self.spec.name in ('point_maze_medium_diverse', 'point_maze_large_diverse', 'point_maze_giant_diverse'):
                x, y = self.active_env.sample_random_xy(rs)
                state = np.array([
                    x,
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                    float(map_idx),
                ])
            else: # U_MAZE
                valid = False
                while not valid:
                    x, y = self.active_env.sample_random_xy(rs)
                    valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))\
                            or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
                state = np.array([
                    x,
                    y,
                    rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
                    rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
                    float(map_idx),
                ])
            return state

        return generate_state(), generate_state()

    def update_env(self, env_info):
        pass

    def set_task_goal(self, goal_state):
        self.active_env.set_target(np.array([goal_state[0], goal_state[1]]))

    def eval_state(self, goal_state, cur_state):
        success = np.linalg.norm(goal_state[:2] - cur_state[:2]) < 0.5
        state_dist = np.linalg.norm(goal_state - cur_state)
        return {
            'success': success,
            'state_dist': state_dist,
        }

    def prepare(self, seed, init_state, stabilize=False):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        map_idx = int(round(float(init_state[4])))
        self.active_map_idx = map_idx
        self.active_env = self.sub_envs[map_idx]
        if map_idx not in self._render_ready:
            if not self.state_based:
                self.active_env.prepare_for_render()
            self._render_ready.add(map_idx)
        self.active_env.seed(seed)
        self.active_env.set_init_state(init_state[:4])
        obs, state_4d = self.active_env.reset(stabilize=stabilize)
        state_5d = np.append(state_4d, float(map_idx)).astype(np.float32)
        if self.state_based:
            obs["visual"] = state_5d
            obs["proprio"] = state_5d
        return obs, state_5d

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states, infos
