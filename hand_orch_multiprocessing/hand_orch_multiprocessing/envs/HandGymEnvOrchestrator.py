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
    self.ff_ws_max,ff_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("ff")
    self.mf_ws_max,mf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("mf")
    self.rf_ws_max,rf_ws_min = self.fingers_ws.get_max_min_xyz_for_finger("rf")
    self.th_ws_max,th_ws_min = self.th_ws.get_max_min_xyz()

    self.current_goal_pos = np.array(self.ff_ws_max + self.mf_ws_max + self.rf_ws_max + self.th_ws_max)
    self.ws_max =  np.array(self.ff_ws_max + self.mf_ws_max + self.rf_ws_max + self.th_ws_max)
    self.ws_min =  np.array(ff_ws_min + mf_ws_min + rf_ws_min + th_ws_min)



    print(f"HandGymEnvOrchestrator::current_goal_pos::{self.current_goal_pos}" )
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
        "high":self.ff_ws_max,
        "low":ff_ws_min
      },
      "ff_fingertip":{
        "high":self.ff_ws_max,
        "low":ff_ws_min
      },
      "mf_goal":{
        "high":self.mf_ws_max,
        "low":mf_ws_min
      },
      "mf_fingertip":{
        "high":self.mf_ws_max,
        "low":mf_ws_min
      },
      "rf_goal":{
        "high":self.rf_ws_max,
        "low":rf_ws_min
      },
      "rf_fingertip":{
        "high":self.rf_ws_max,
        "low":rf_ws_min
      },
      "th_goal":{
        "high":self.th_ws_max,
        "low":th_ws_min
      },
      "th_fingertip":{
        "high":self.th_ws_max,
        "low":th_ws_min
      },
      "dist":{
        "high":[0.2]*4,
        "low": [0]*4
      },
      "history":{
        "high":None,
        "low":None
      }
    }
    #last_act last_last_act
    history_high = self.ff_ws_max+self.mf_ws_max+self.rf_ws_max+self.th_ws_max
    history_high = history_high + history_high

    history_low = ff_ws_min+mf_ws_min+rf_ws_min+th_ws_min
    history_low = history_low + history_low

    self.state_limit["history"]["high"] = history_high
    self.state_limit["history"]["low"] = history_low


    self.obs_high = np.array(
      # Main goal
      self.state_limit["ff_goal"]["high"] +  self.state_limit["mf_goal"]["high"] +
      self.state_limit["rf_goal"]["high"] +  self.state_limit["th_goal"]["high"] +
      # Current goal
      self.state_limit["ff_goal"]["high"] +  self.state_limit["mf_goal"]["high"] +
      self.state_limit["rf_goal"]["high"] +  self.state_limit["th_goal"]["high"] +
      # Fingertip  pose
      self.state_limit["ff_fingertip"]["high"] +  self.state_limit["mf_fingertip"]["high"] +
      self.state_limit["rf_fingertip"]["high"] +  self.state_limit["th_fingertip"]["high"] +
      # Distance from current goal to main goal
      self.state_limit["dist"]["high"] +
      # Distance from fingertip to  current goal
      self.state_limit["dist"]["high"] +
      # history of goals for last and last_last_act
      self.state_limit["history"]["high"]
      ,dtype=np.float32)

    self.obs_low = np.array(
      # Main goal
      self.state_limit["ff_goal"]["low"] +  self.state_limit["mf_goal"]["low"] +
      self.state_limit["rf_goal"]["low"] +  self.state_limit["th_goal"]["low"] +
      # Current goal
      self.state_limit["ff_goal"]["low"] +  self.state_limit["mf_goal"]["low"] +
      self.state_limit["rf_goal"]["low"] +  self.state_limit["th_goal"]["low"] +
      # Fingertip  pose
      self.state_limit["ff_fingertip"]["low"] +  self.state_limit["mf_fingertip"]["low"] +
      self.state_limit["rf_fingertip"]["low"] +  self.state_limit["th_fingertip"]["low"] +
      # Distance from current goal to main goal
      self.state_limit["dist"]["low"] +
      # Distance from fingertip to  current goal
      self.state_limit["dist"]["low"] +
      # history of goals for last and last_last_act
      self.state_limit["history"]["low"]
      ,dtype=np.float32)
    self.observation_space =  spaces.Box(self.obs_low, self.obs_high)

    ###########setting up action space###########
    self.action_space = spaces.Box(np.array([-self.delta]*12,dtype=np.float32),
                                   np.array([self.delta]*12,dtype=np.float32)
    )
    ########## History ############
    self._history = {
      "last_act":[0]*12,
      "last_last_act":[0]*12
    }

    self.seed = None

  def reset(self,seed=None, options=None):
    if seed is not None:
          self.seed = seed
          # If you use any random numbers, seed them here, e.g.
          import random
          random.seed(seed)
          np.random.seed(seed)

    self.current_goal_pos = np.array(self.ff_ws_max + self.mf_ws_max + self.rf_ws_max + self.th_ws_max)
    self._goals = {
      "FF":self._fingers_goal_generator.get_goal("FF") ,
      "MF":self._fingers_goal_generator.get_goal("MF") ,
      "RF":self._fingers_goal_generator.get_goal("RF") ,
      "TH":self._thumb_goal_generator.get_goal()
    }


    self._history = {
      "last_act":[0]*12,
      "last_last_act":[0]*12
    }
    self.current_step = 0
    self._env.unwrapped.set_goal_location(self.current_goal_pos)
    minion_state,info = self._env.reset()
    state, state_dict = self.get_observation(minion_state, self._history)

    return state, {}

  def step(self,action):
    self.current_step +=1
    # every step the orchestrator produces and intermediate goal so
    # that the hand reaches the final goal through intermediate goals
    self.current_goal_pos += action
    clipped_action = np.clip(self.current_goal_pos, self.ws_min, self.ws_max)
    self.current_goal_pos = clipped_action

    # Set intermediate goal based on the action
    self._env.unwrapped.set_goal_location(clipped_action)
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
        actions += finger_action.tolist()

      th_action = self._thumb_agent(obs_dict["TH"])

      actions += th_action.tolist()

      minions_state,minions_reward, minions_done, _, _ = self._env.step(actions)

    self._history["last_last_act"] = self._history["last_act"]
    self._history["last_act"] = clipped_action.tolist()
  
    state, state_dict = self.get_observation(minions_state, self._history)
    reward = self.get_reward(state_dict)
    done = self.get_termination(state_dict)
    info = {}

    truncated = self.current_step  > self.max_episode_step

    return state, reward, done, truncated, info

    # calculate reward by working out distance between current intermediate goal and final goal

  def get_reward(self,state_dict):
    reward = -1* np.sum(state_dict["dist_to_main_goal"])
    return reward

  def get_termination(self,state_dict):

    # do this with numpy
    dists = state_dict["dist_from_current_goal_to_main_goal"]

    termination_flags = np.array(dists) < self._success_threshold
    reached_max_episode = self.current_step  > self.max_episode_step

    return termination_flags.all() | reached_max_episode

  def get_observation(self,minions_state, history):
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

    state_dict = {
      # hand_goals: Main goal the hand wants to achieve
      "hand_goals": self._goals["FF"] + self._goals["MF"] +self._goals["RF"] + self._goals["TH"], #12
      "current_goal":np.concatenate((FF_current_goal, 
                                     MF_current_goal,
                                     RF_current_goal,
                                     TH_current_goal)).tolist(), #12
      "fingertip_pos":np.concatenate((FF_finger_pos, 
                                      MF_finger_pos, 
                                      RF_finger_pos, 
                                      TH_finger_pos)).tolist(), #12
      "dist_from_current_goal_to_main_goal":[dist_from_current_goal_to_main_goal_FF,
                                             dist_from_current_goal_to_main_goal_MF,
                                             dist_from_current_goal_to_main_goal_RF,
                                             dist_from_current_goal_to_main_goal_TH], #4
      "dist_from_fingertip_to_current_goal":[dist_to_current_goal_FF,
                                             dist_to_current_goal_MF,
                                             dist_to_current_goal_RF,
                                             dist_to_current_goal_TH], #4
      "dist_to_main_goal":[dist_to_main_goal_FF,
                           dist_to_main_goal_MF,
                           dist_to_main_goal_RF,
                           dist_to_main_goal_TH] #4
    }

    # TODO: add history of previous goals chosen by the agent

    for key, value in state_dict.items():
      print("\n")
      print(f"{key}::{type(value)}\n{value}")
      print("\n")


    state = state_dict["hand_goals"] \
      + state_dict["current_goal"] \
      + state_dict["fingertip_pos"] \
      + state_dict["dist_from_current_goal_to_main_goal"] \
      + state_dict["dist_from_fingertip_to_current_goal"] \
      + history["last_last_act"] \
      + history["last_act"]


    return state, state_dict
    # Obs dim: (combined state_dict dim) * num_env type: numpy




