import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import  DummyVecEnv, SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import numpy as np

from fingers_multiprocessing.envs.fingerGymEnv import Workspace_Util as  FingerWorkspace_Util
from thumb_multiprocessing.envs.thumbGymEnv import Workspace_Util as  ThumbWorkspace_Util

from fingers_multiprocessing.envs.fingerGymEnv import BasicGoalGenerator as  FingerBasicGoalGenerator
from thumb_multiprocessing.envs.thumbGymEnv import BasicGoalGenerator as  ThumbBasicGoalGenerator


class HandGymEnvOrchestrator(gymnasium.Env):
  def __init__(self,thumb_agent,fingers_agent,
               hand_env_config,
               log_dir,
               success_threshold = 0.01):
    self._thumb_agent = thumb_agent
    self._fingers_agent = fingers_agent

    self._success_threshold = success_threshold
    env_id = "hand_multiprocessing-v0"
    
    self._env = gymnasium.make(env_id,
        **hand_env_config
    )

    # This is the final goal the agent should reach
    # self.synergy_goal

    self.max_episode_step = 10
    self.current_step =  0
    ############# ws utils ################
    self.th_ws = ThumbWorkspace_Util()
    self.fingers_ws = FingerWorkspace_Util()
    ###########setting up state space###########
    self.delta = 0.05
    ff_ws_max,ff_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("ff")
    mf_ws_max,mf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("mf")
    rf_ws_max,rf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("rf")
    th_ws_max,th_ws_min = self.th_ws.get_max_min_xyz()

    # TODO: actually sample workspace#
    self._fingers_goal_generator = FingerBasicGoalGenerator()
    self._thumb_goal_generator = ThumbBasicGoalGenerator()
    self._goals = {
      "FF":self._fingers_goal_generator.get_goal("FF") ,
      "MF":self._fingers_goal_generator.get_goal("MF") ,
      "RF":self._fingers_goal_generator.get_goal("RF") ,
      "TH":self._thumb_goal_generator.get_goal() 
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
    self.current_step = 0
    minion_state,info = self._env.reset()
    state, state_dict = self.get_observation(minion_state)
    return state, {}

  def step(self,action):
    self.current_step[:] +=1
    # every step the orchestrator produces and intermediate goal so
    # that the hand reaches the final goal through intermediate goals

    # Set intermediate goal based on the action
    for i in range(self._num_envs):
      self._env.env_method("set_goal_location",action[i,:],indices=i)
    minions_state,info = self._env.reset()


    # wait 200 steps for the agent to reach the goal
    minions_done = False
    while not minions_done:
      actions = []
      obs_dict = {
       "FF":minions_state[:20],
       "MF":minions_state[20:40],
       "RF":minions_state[40:60],
       "TH":minions_state[60:]
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
      # print(f"HandGymEnvOrchestrator::step::vec_action::{vec_action}")

      minions_state,minions_reward, minions_done, minions_truncated = self._env.step(actions)
      # print(f"minions_done::{minions_done}")
      # print(f"minions_done::type::{type(minions_done)}")


  

    state, state_dict = self.get_observation(minions_state)
    reward = self.get_reward(state_dict)
    done = self.get_termination(state_dict)
    info = {}

    # print(f"reward::shape::{reward.shape}")

    truncated = self.current_step  > self.max_episode_step

    # print(f"minions_state::{minions_state}")
    # print(f"minions_state::shape::{minions_state.shape}")
    return obs, reward, done, truncated, info

    # calculate reward by working out distance between current intermediate goal and final goal

  def get_reward(self,state_dict):
    reward = -1* np.sum(state_dict["dist_to_main_goal"])
    return reward

  def get_termination(self,state_dict):

    # do this with numpy
    dists = state_dict["dist_from_current_goal_to_main_goal"]

    termination_flags = dists < self._success_threshold
    reached_max_episode = self.current_step  > self.max_episode_step

    return termination_flags | reached_max_episode

  def get_observation(self,minions_state):
    print(f"HandGymEnvOrchestrator::get_observation::minions_state::type::{type(minions_state)}")
    print(f"HandGymEnvOrchestrator::get_observation::minions_state::{minions_state}")
    FF_obs = minions_state[:20]
    MF_obs = minions_state[20:40]
    RF_obs = minions_state[40:60]
    TH_obs = minions_state[60:]

    FF_finger_pos = FF_obs[12:15]
    MF_finger_pos = MF_obs[12:15]
    RF_finger_pos = RF_obs[12:15]
    TH_finger_pos = RF_obs[12:15]

    FF_current_goal = FF_obs[15:18]
    MF_current_goal = MF_obs[15:18]
    RF_current_goal = RF_obs[15:18]
    TH_current_goal = TH_obs[15:18]
    print(f"HandGymEnvOrchestrator::get_observation::FF_finger_pos::{FF_finger_pos}")
    print(f"HandGymEnvOrchestrator::get_observation::FF_finger_pos::{FF_current_goal}")

    dist_to_current_goal_FF = np.linalg.norm(FF_finger_pos - FF_current_goal)
    dist_to_current_goal_MF = np.linalg.norm(MF_finger_pos - MF_current_goal)
    dist_to_current_goal_RF = np.linalg.norm(RF_finger_pos - RF_current_goal)
    dist_to_current_goal_TH = np.linalg.norm(TH_finger_pos - TH_current_goal)


    dist_to_main_goal_FF = np.linalg.norm(FF_finger_pos - self._goals["FF"])
    dist_to_main_goal_MF = np.linalg.norm(MF_finger_pos - self._goals["MF"])
    dist_to_main_goal_RF = np.linalg.norm(RF_finger_pos - self._goals["RF"])
    dist_to_main_goal_TH = np.linalg.norm(TH_finger_pos - self._goals["TH"])

    dist_from_current_goal_to_main_goal_FF =  np.linalg.norm(FF_current_goal - self._goals["FF"])
    dist_from_current_goal_to_main_goal_MF =  np.linalg.norm(MF_current_goal - self._goals["MF"])
    dist_from_current_goal_to_main_goal_RF =  np.linalg.norm(RF_current_goal - self._goals["RF"])
    dist_from_current_goal_to_main_goal_TH =  np.linalg.norm(TH_current_goal - self._goals["TH"])


    print(f"dist_from_current_goal_to_main_goal_FF::{dist_from_current_goal_to_main_goal_FF}")
    print(f"dist_from_current_goal_to_main_goal_MF::{dist_from_current_goal_to_main_goal_MF}")
    print(f"dist_from_current_goal_to_main_goal_RF::{dist_from_current_goal_to_main_goal_RF}")
    print(f"dist_from_current_goal_to_main_goal_TH::{dist_from_current_goal_to_main_goal_TH}")


    state_dict = {
      # hand_goals: Main goal the hand wants to achieve
      "hand_goals": self._goals["FF"] + self._goals["MF"] +self._goals["RF"] + self._goals["TH"],
      "current_goal":np.concatenate((FF_current_goal, 
                                     MF_current_goal,
                                     RF_current_goal,
                                     TH_current_goal)).tolist(),
      "fingertip_pos":np.concatenate((FF_finger_pos, 
                                      MF_finger_pos, 
                                      RF_finger_pos, 
                                      TH_finger_pos)).tolist(),
      "dist_from_current_goal_to_main_goal":[dist_from_current_goal_to_main_goal_FF,
                                                            dist_from_current_goal_to_main_goal_MF, 
                                                            dist_from_current_goal_to_main_goal_RF,
                                                            dist_from_current_goal_to_main_goal_TH],
      "dist_from_fingertip_to_current_goal":[dist_to_current_goal_FF,
                                                            dist_to_current_goal_MF,
                                                            dist_to_current_goal_RF,
                                                            dist_to_current_goal_TH],
      "dist_to_main_goal":[dist_to_main_goal_FF,
                                          dist_to_main_goal_MF, 
                                          dist_to_main_goal_RF,
                                          dist_to_main_goal_TH]
    }

    for key, value in state_dict.items():
      print(f"{key}::{type(value)}\n{value}")

    state = state_dict["hand_goals"] + state_dict["current_goal"] + state_dict["fingertip_pos"] + state_dict["dist_from_current_goal_to_main_goal"] + state_dict["dist_from_fingertip_to_current_goal"]
 

    return state, state_dict
    # Obs dim: (combined state_dict dim) * num_env type: numpy




