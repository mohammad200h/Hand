import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import  DummyVecEnv, SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import numpy as np

from fingers_multiprocessing.envs.fingerGymEnv import Workspace_Util as  FingerWorkspace_Util
from thumb_multiprocessing.envs.thumbGymEnv import Workspace_Util as  ThumbWorkspace_Util


class HandGymEnvOrchestrator(gymnasium.Env):
  def __init__(self, thumb_agent,fingers_agent,
               num_envs,hand_env_config,
               log_dir,
               success_threshold = 0.01):
    self._thumb_agent = thumb_agent
    self._fingers_agent = fingers_agent

    self._success_threshold = success_threshold

    print(f"creating {num_envs} vector envs...")
    env_id = "hand_multiprocessing-v0"
    env = make_vec_env(env_id, n_envs=num_envs,env_kwargs=hand_env_config,
                       monitor_dir=log_dir,vec_env_cls=SubprocVecEnv
    )
    self._env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    print(f"done {num_envs} vector envs.")

    # This is the final goal the agent should reach
    # self.synergy_goal

    ############# ws utils ################
    self.th_ws = ThumbWorkspace_Util()
    self.fingers_ws = FingerWorkspace_Util()
    ###########setting up state space###########
    self.delta = 0.05
    ff_ws_max,ff_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("ff")
    mf_ws_max,mf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("mf")
    rf_ws_max,rf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("rf")
    th_ws_max,th_ws_min = self.th_ws.get_max_min_xyz()

    self.state_limit = {
      "ff_goal":{
        "high":ff_ws_max,
        "low":ff_ws_min
      },
      "ff_fingertip":{
        "high":ff_ws_max,
        "low":ff_ws_min
      },
      "mf_goal":{
        "high":mf_ws_max,
        "low":mf_ws_min
      },
      "mf_fingertip":{
        "high":mf_ws_max,
        "low":mf_ws_min
      },
      "rf_goal":{
        "high":rf_ws_max,
        "low":rf_ws_min
      },
      "rf_fingertip":{
        "high":rf_ws_max,
        "low":rf_ws_min
      },
      "th_goal":{
        "high":th_ws_max,
        "low":th_ws_min
      },
      "th_fingertip":{
        "high":th_ws_max,
        "low":th_ws_min
      },
      "history":{
        "high":None,
        "low":None
      }
    }
    #last_act last_last_act
    history_high = ff_ws_max+mf_ws_max+rf_ws_max+th_ws_max
    history_high = history_high + history_high

    history_low = ff_ws_min+mf_ws_min+rf_ws_min+th_ws_min
    history_low = history_low + history_low

    self.state_limit["history"]["high"] = history_high
    self.state_limit["history"]["low"] = history_low


    self.obs_high = np.array(
      self.state_limit["ff_goal"]["high"] +  self.state_limit["mf_goal"]["high"] +
      self.state_limit["rf_goal"]["high"] +  self.state_limit["th_goal"]["high"] +
      self.state_limit["history"]["high"]
      ,dtype=np.float32)
    self.obs_low = np.array(
      self.state_limit["ff_goal"]["low"] +  self.state_limit["mf_goal"]["low"] +
      self.state_limit["rf_goal"]["low"] +  self.state_limit["th_goal"]["low"] +
      self.state_limit["history"]["low"]
      ,dtype=np.float32)
    self.observation_space =  spaces.Box(self.obs_low, self.obs_high)

    ###########setting up action space###########
    self.action_space = spaces.Box(np.array([-self.delta]*12,dtype=np.float32),
                                   np.array([self.delta]*12,dtype=np.float32)
    )
    self.seed = None
  def reset(self,seed=None, options=None):
    if seed is not None:
          self.seed = seed
          # If you use any random numbers, seed them here, e.g.
          import random
          import numpy as np
          random.seed(seed)
          np.random.seed(seed)

    obs = self._env.reset()
    return obs, {}

  def step(self,action):
    # every step the orchestrator produces and intermidiate goal so
    # that the hand reaches the final goal through intermidiate goals
    ff_gaol = action[:4]
    mf_goal = action[4:8]
    rf_goal = action[8:12]
    th_goal = action[12:]
    goals =  {
        "FF":ff_gaol,
        "MF":mf_goal,
        "RF":rf_goal,
        "TH":th_goal
      }
    # set_intermidiate_goal
    obs = self._env.reset()

    # wait 200 steps for the agent to reach the goal
    done = False
    while not done:
      actions = []
      for finger in ["FF","MF","RF"]:
        finger_obs =None
        finger_action = self._fingers_agent(finger_obs)
        actions += finger_action

      thumb_obs = None
      th_action = self._thumb_agent(thumb_obs)
      actions += th_action
      obs = self._env.step(action)

    # calculate reward by working out distance between current intermidiate goal and final goal


  def get_reward(self):
    pass

  def get_termination(self):
    pass




