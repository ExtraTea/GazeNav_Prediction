import numpy as np
import gym
from gym.spaces.box import Box
import torch

from baselines import bench
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.vec_normalize import \
    VecNormalize as VecNormalize_

from rl.networks.shmem_vec_env import ShmemVecEnv
from rl.networks.dummy_vec_env import DummyVecEnv
from rl.vec_env.vec_pretext_normalize import VecPretextNormalize

def make_env(env_id, seed, rank, log_dir, allow_early_resets, config=None, envNum=1, ax=None, test_case=-1):
    def _thunk():
        env = gym.make(env_id)
        env.configure(config)
        envSeed = seed + rank if seed is not None else None
        env.thisSeed = envSeed
        env.nenv = envNum
        if envNum > 1:
            env.phase = 'train'
        else:
            env.phase = 'test'
        
        if ax:
            env.render_axix = ax
            if test_case>=0:
                env.test_case = test_case
        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)
        
        env = bench.Monitor(env, None, allow_early_resets=allow_early_resets)
        print(env)

        return env
    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  config=None,
                  ax = None, test_case = -1, wrap_pytorch=True, pretext_wrapper=False):
    envs = [make_env(env_name, seed, i, log_dir, allow_early_resets, config=config, envNum = num_processes, ax = ax, test_case = test_case)
            for i in range(num_processes)]
    
    test = False if len(envs) > 1 else True
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)
    if wrap_pytorch:
        if isinstance(envs.observation_space, Box):
            if len(envs.observation_space.shape) == 1:
                if gamma is None:
                    envs = VecNormalize(envs, ret=False, ob=False)
                else:
                    envs = VecNormalize(envs, gamma=gamma, ob=False, ret=False)

        envs = VecPyTorch(envs, device)
    if pretext_wrapper:
        if gamma is None:
            envs = VecPretextNormalize(envs, ret=False, ob=False, config=config, test=test)
        else:
            envs = VecPretextNormalize(envs, gamma=gamma, ob=False, ret=False, config=config, test=test)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif isinstance(envs.observation_space, Box):
        if len(envs.observation_space.shape) == 3:
            envs = VecPyTorchFrameStack(envs, 4, device)

    return envs
    
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, env, done, info = self.env.step(action)
        if done and self.env._max_episode_steps==self.env._elapsed_steps:
            info['bad_transition'] = True
        return obs, env, done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, dict):
            for key in obs:
                obs[key]=torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        if isinstance(obs, dict):
            for key in obs:
                obs[key] = torch.from_numpy(obs[key]).to(self.device)
        else:
            obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info

    def render_traj(self, path, episode_num):
        if self.venv.num_envs == 1:
            return self.venv.envs[0].env.render_traj(path, episode_num)
        else:
            for i, curr_env in enumerate(self.venv.envs):
                curr_env.env.render_traj(path, str(episode_num) + '.' + str(i))

class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) /
                          np.sqrt(self.ob_rms.var + self.epsilon),
                          -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs, ) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()