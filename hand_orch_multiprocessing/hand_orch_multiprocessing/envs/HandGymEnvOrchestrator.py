import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import  DummyVecEnv, SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import numpy as np

from fingers_multiprocessing.envs.fingerGymEnv import Workspace_Util as  FingerWorkspace_Util
from thumb_multiprocessing.envs.thumbGymEnv import Workspace_Util as  ThumbWorkspace_Util


class HandGymEnvOrchestrator(gymnasium.Env):
  def __init__(self,thumb_agent,fingers_agent,
               num_envs,hand_env_config,
               log_dir,
               success_threshold = 0.01):
    self._thumb_agent = thumb_agent
    self._fingers_agent = fingers_agent

    self._success_threshold = success_threshold

    print(f"creating {num_envs} vector envs...")
    self._num_envs = num_envs
    env_id = "hand_multiprocessing-v0"
    env = make_vec_env(env_id, n_envs=num_envs,env_kwargs=hand_env_config,
                       monitor_dir=log_dir,vec_env_cls=SubprocVecEnv
    )
    self._env = env
    # self._env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    print(f"done {num_envs} vector envs.")

    # This is the final goal the agent should reach
    # self.synergy_goal

    self.max_episode_step = 10
    self.current_step =  np.full((2, 1), 0)
    ############# ws utils ################
    self.th_ws = ThumbWorkspace_Util()
    self.fingers_ws = FingerWorkspace_Util()
    ###########setting up state space###########
    self.delta = 0.05
    ff_ws_max,ff_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("ff")
    mf_ws_max,mf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("mf")
    rf_ws_max,rf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("rf")
    th_ws_max,th_ws_min = self.th_ws.get_max_min_xyz()

    # TODO: actually sample workspace
    self._goals = {
      "FF":np.array([0]*3),
      "MF":np.array([0]*3),
      "RF":np.array([0]*3),
      "TH":np.array([0]*3)
    }

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
          random.seed(seed)
          np.random.seed(seed)
    #TODO: choose a random main goal
    self.current_step =  np.full((2, 1), 0)
    minions_state = self._env.reset()
    state, state_dict = self.get_observation(minions_state)
    return state, {}

  def step(self,action):
    self.current_step[:] +=1
    # every step the orchestrator produces and intermediate goal so
    # that the hand reaches the final goal through intermediate goals

    # Set intermediate goal based on the action
    for i in range(self._num_envs):
      self._env.env_method("set_goal_location",action[i,:],indices=i)
    minions_state = self._env.reset()


    # wait 200 steps for the agent to reach the goal
    minions_done = np.array([False]* self._num_envs)
    while not minions_done.all():
      actions = []
      obs_dict = {
       "FF":minions_state[:,:20],
       "MF":minions_state[:,20:40],
       "RF":minions_state[:,40:60],
       "TH":minions_state[:,60:]
      }

      
      for finger in ["FF","MF","RF"]:
        finger_action = self._fingers_agent(obs_dict[finger])
        print(f"HandGymEnvOrchestrator::finger_action::{finger_action}")
        actions += finger_action

      th_action = self._thumb_agent(obs_dict["TH"])
      print(f"HandGymEnvOrchestrator::type::{type(th_action)}")
      print(f"HandGymEnvOrchestrator::th_action::shape::{th_action.shape}")
      print(f"HandGymEnvOrchestrator::th_action::{th_action}")

      actions += th_action

      # print(f"HandGymEnvOrchestrator::step::actions::{actions}")
      vec_action = np.vstack([actions, actions.copy()])
      # print(f"HandGymEnvOrchestrator::step::vec_action::{vec_action}")

      minions_state,minions_reward, minions_done, minions_truncated = self._env.step(vec_action)
      # print(f"minions_done::{minions_done}")
      # print(f"minions_done::type::{type(minions_done)}")


  

    state, state_dict = self.get_observation(minions_state)
    reward = self.get_reward(state_dict)
    done = self.get_termination(state_dict)
    info = {}

    # print(f"reward::shape::{reward.shape}")

    truncated = (self.current_step  > self.max_episode_step).all(axis=1, keepdims=True)

    # print(f"minions_state::{minions_state}")
    # print(f"minions_state::shape::{minions_state.shape}")
    return obs, reward, done, truncated, info

    # calculate reward by working out distance between current intermediate goal and final goal

  def get_reward(self,state_dict):
    reward = -1* np.sum(state_dict["dist_to_main_goal"],axis=1)
    return reward

  def get_termination(self,state_dict):

    # do this with numpy
    dists = state_dict["dist_from_current_goal_to_main_goal"]

    termination_flags = (dists < self._success_threshold).all(axis=1, keepdims=True)
    reached_max_episode = (self.current_step  > self.max_episode_step).all(axis=1, keepdims=True)

    return termination_flags | reached_max_episode

  def get_observation(self,minions_state):
    FF_obs = minions_state[:,:20]
    MF_obs = minions_state[:,20:40]
    RF_obs = minions_state[:,40:60]
    TH_obs = minions_state[:,60:]

    FF_finger_pos = FF_obs[:,12:15]
    MF_finger_pos = MF_obs[:,12:15]
    RF_finger_pos = RF_obs[:,12:15]
    TH_finger_pos = RF_obs[:,12:15]

    FF_current_goal = FF_obs[:,15:18]
    MF_current_goal = MF_obs[:,15:18]
    RF_current_goal = RF_obs[:,15:18]
    TH_current_goal = TH_obs[:,15:18]

    # num_env = 2
    # calculate distance between FF_finger_pos and FF_current_goal I am looking for shape  ( num_env, dist )
    dist_to_current_goal_FF = np.linalg.norm(FF_finger_pos - FF_current_goal, axis=1)
    dist_to_current_goal_MF = np.linalg.norm(MF_finger_pos - MF_current_goal, axis=1)
    dist_to_current_goal_RF = np.linalg.norm(RF_finger_pos - RF_current_goal, axis=1)
    dist_to_current_goal_TH = np.linalg.norm(TH_finger_pos - TH_current_goal, axis=1)


    dist_to_main_goal_FF = np.linalg.norm(FF_finger_pos - self._goals["FF"] , axis=1)
    dist_to_main_goal_MF = np.linalg.norm(MF_finger_pos - self._goals["MF"] , axis=1)
    dist_to_main_goal_RF = np.linalg.norm(RF_finger_pos - self._goals["RF"] , axis=1)
    dist_to_main_goal_TH = np.linalg.norm(TH_finger_pos - self._goals["TH"] , axis=1)

    dist_from_current_goal_to_main_goal_FF =  np.linalg.norm(FF_current_goal - self._goals["FF"] , axis=1)
    dist_from_current_goal_to_main_goal_MF =  np.linalg.norm(MF_current_goal - self._goals["MF"] , axis=1)
    dist_from_current_goal_to_main_goal_RF =  np.linalg.norm(RF_current_goal - self._goals["RF"] , axis=1)
    dist_from_current_goal_to_main_goal_TH =  np.linalg.norm(TH_current_goal - self._goals["TH"] , axis=1)


    # print(f"")
    # print(f"FF_finger_pos::shape::{FF_finger_pos.shape}")
    # print(f"FF_current_goal::shape::{FF_current_goal.shape}")
    # print(f"dist_to_current_goal_FF::shape::{dist_to_current_goal_FF.shape}")
    # print(f"dist_to_main_goal_FF::shape::{dist_to_main_goal_FF.shape}")


    goals = np.concatenate((
      self._goals["FF"],
      self._goals["MF"],
      self._goals["RF"],
      self._goals["TH"]
    ))
    main_goals = np.tile(goals, (self._num_envs, 1))



    state_dict = {
      # hand_goals: Main goal the hand wants to achieve
      "hand_goals": main_goals, # dim : (x,y,z) * num_fingers
      "current_goal":np.concatenate((FF_current_goal,
                                     MF_current_goal,
                                     RF_current_goal,
                                     TH_current_goal), axis=1),
      "fingertip_pos": np.concatenate((FF_finger_pos,
                                       MF_finger_pos,
                                       RF_finger_pos,
                                       TH_finger_pos), axis=1),
      "dist_from_current_goal_to_main_goal":np.stack((dist_from_current_goal_to_main_goal_FF,
                                                            dist_from_current_goal_to_main_goal_MF,
                                                            dist_from_current_goal_to_main_goal_RF,
                                                            dist_from_current_goal_to_main_goal_TH), axis=1),
      "dist_from_fingertip_to_current_goal": np.stack((dist_to_current_goal_FF,
                                                             dist_to_current_goal_MF,
                                                             dist_to_current_goal_RF,
                                                             dist_to_current_goal_TH), axis=1),
      "dist_to_main_goal":np.stack((dist_to_main_goal_FF,
                                          dist_to_main_goal_MF,
                                          dist_to_main_goal_RF,
                                          dist_to_main_goal_TH), axis=1),

    }

    state = np.concatenate((
      state_dict["hand_goals"],
      state_dict["current_goal"],
      state_dict["fingertip_pos"],
      state_dict["dist_from_current_goal_to_main_goal"],
      state_dict["dist_from_fingertip_to_current_goal"]
    ),axis=1)

    return state, state_dict
    # Obs dim: (combined state_dict dim) * num_env type: numpy




