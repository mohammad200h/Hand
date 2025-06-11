import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import sys

from pkg_resources import resource_string,resource_filename
import time
import random
import math
import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np

import pybullet as p

from controller import Hand

import random
import pybullet_data
from pkg_resources import parse_version
from mamad_util import JointInfo

from collections import OrderedDict

import warnings 
import multiprocessing
from datetime import datetime
import yaml
import io

import matplotlib.pyplot as plt
import matplotlib

from tf_independednt_of_parameter import CoordinateFrameTrasform


from fingers_multiprocessing.envs.fingerGymEnv import Workspace_Util as  FingerWorkspace_Util
from fingers_multiprocessing.envs.fingerGymEnv import Observation as  FingersObservationBase
from fingers_multiprocessing.envs.fingerGymEnv import Action as  FingersActionBase
from fingers_multiprocessing.envs.fingerGymEnv import BasicGoalGenerator as  FingersBasicGoalGenerator
from fingers_multiprocessing.envs.fingerGymEnv import AdaptiveTaskParameter as  FingersAdaptiveTaskParameter
from fingers_multiprocessing.envs.fingerGymEnv import RandomStart as  FingersRandomStart
   


from thumb_multiprocessing.envs.thumbGymEnv import Workspace_Util as  ThumbWorkspace_Util
from thumb_multiprocessing.envs.thumbGymEnv import Observation as  ThumbObservationBase
from thumb_multiprocessing.envs.thumbGymEnv import Action as  ThumbActionBase
from thumb_multiprocessing.envs.thumbGymEnv import BasicGoalGenerator as  ThumbBasicGoalGenerator
from thumb_multiprocessing.envs.thumbGymEnv import AdaptiveTaskParameter as  ThumbAdaptiveTaskParameter
from thumb_multiprocessing.envs.thumbGymEnv import RandomStart as  ThumbRandomStart


class FingersObservation(FingersObservationBase):
  def __init__(self,physic_engine,finger_obj,workspace_util,obs_mode ="finger_joints_and_distnace"):
    super().__init__(physic_engine, finger_obj, workspace_util, obs_mode)

  def get_joint_values(self):
    # print("get_joint_values::self.finger_name:: ",self.finger_name)
    return  self.controller.get_Observation_finger(self.finger_name)

  def get_distance_from_fingertip_to_goal(self):

        goal_pos = self.get_goal_pos()
        finger_tip_pos =  self.controller.get_observation_finger_tip(self.finger_name)

        x_dist = goal_pos[0] - finger_tip_pos[0]
        y_dist = goal_pos[1] - finger_tip_pos[1]
        z_dist = goal_pos[2] - finger_tip_pos[2]
        dist =  math.sqrt(x_dist**2+y_dist**2+z_dist**2)

        return dist
  def get_finger_tip_pos_in_world_frame(self):
      finger_tip_pos =  self.controller.get_observation_finger_tip(self.finger_name)
      # print(f"FingersObservation::finger_name::{self.finger_name}::finger_tip_pos::{finger_tip_pos}")
      return finger_tip_pos

class ThumbObservation(ThumbObservationBase):
  def __init__(self,physic_engine,finger_obj,workspace_util,obs_mode ="finger_joint_and_xyz"):
    super().__init__(physic_engine, finger_obj, workspace_util, obs_mode)

  def get_joint_values(self):
        return self.controller.get_Observation_thumb()
  def get_distance_from_fingertip_to_goal(self):

        goal_pos = self.get_goal_pos()
        finger_tip_pos,_ = self.controller.get_complete_obs_thumb_tip()

        # print("get_distance_from_fingertip_to_goal::goal_pos:: ",goal_pos)
        # print("get_distance_from_fingertip_to_goal::finger_tip_pos:: ",finger_tip_pos)

        x_dist = goal_pos[0] - finger_tip_pos[0]
        y_dist = goal_pos[1] - finger_tip_pos[1]
        z_dist = goal_pos[2] - finger_tip_pos[2]
        dist =  math.sqrt(x_dist**2+y_dist**2+z_dist**2)

        return dist

  def get_finger_tip_pos_in_world_frame(self):
      finger_tip_pos =  self.controller.get_observation_finger_tip("TH")
      return finger_tip_pos

class FingersAction(FingersActionBase):
  def __init__(self,action_mode,symitric_action,controller_obj,workspace_util):
    super().__init__(action_mode,symitric_action,controller_obj,workspace_util)

  def get_current_state_of_joints(self):
    return self.controller_obj.getObservation()
class ThumbAction(ThumbActionBase):
  def __init__(self,action_mode,symitric_action,controller_obj):
    super().__init__(action_mode,symitric_action,controller_obj)
  def get_current_state_of_joints(self):
    return self.controller_obj.getObservation()

class Reward():
  def __init__(self,reward_mode="dense_distance"):
    self.reward_mode = reward_mode 
    self.reward_modes = ["dense_distance","dense_distance_and_goal","sparse"]

  def get_reward(self,distance,goal_achived_flag):
    reward = None
    if self.reward_mode=="dense_distance":
      reward = self.dense_distance(distance)

    elif self.reward_mode=="dense_distance_and_goal":
      reward = self.dense_distance_and_goal(distance,goal_achived_flag)

    if self.reward_mode=="sparse":
      reward = self.sparse()

    return reward
  
  def dense_distance(self,distance):
    goal_reward = 0 
    dist_penalty = -1*distance
  
    reward = goal_reward+dist_penalty

    return reward

  def dense_distance_and_goal(self,distance,goal_achived_flag):
    goal_reward = 0
    
    if goal_achived_flag:
      goal_reward = 10

    
    dist_penalty = -1*distance
  
    reward = goal_reward+dist_penalty

    return reward
    

  def sparse(self):
    goal_reward =-1
    
    
    reward = goal_reward

    return reward 

class HandGymEnv(gymnasium.Env):
    
    def __init__(self,renders=True,
                 render_mode = None,
                 timeStep=2000,random_robot_start=False,
                 record_performance=False,
                 obs_mode={"fingers":"finger_joint_and_xyz","thumb":"finger_joint_and_xyz"},
                 action_mode ="delta_jointControl",reward_mode="dense_distance",
                 adaptive_task_parameter_flag=False,atp_neighbour_radius=0.01,
                 atp_num_success_required = 2,
                 atp_use_lower_limit=False,
                 atp_sphare_thinkness=0.005,
                 symitric_action = False,
                 orchestrator_mode = False,
                 debug=False
                ):
        self._p = p 
        self._render = renders
        self._timeStep = 1/timeStep
        self.debug = debug

        self._orchestrator_mode = orchestrator_mode

        self.action_mode = action_mode
        self.random_robot_start = random_robot_start

       
        self.controller = None 

        self.finger_names = ["FF","MF","RF"]

        ###### finger goals #### 
        self.goals ={
          "ids":{
            "current":{
              "FF":None,
              "MF":None,
              "RF":None,
              "TH":None
            },
            "previous":{
              "FF":None,
              "MF":None,
              "RF":None,
              "TH":None
            }
            
          },
          "locations":{
            "current":{
              "FF":None,
              "MF":None,
              "RF":None,
              "TH":None
            },
            "previous":{
              "FF":None,
              "MF":None,
              "RF":None,
              "TH":None
            }
          }
        }
       
        ##### connecting to a physic server #####
        if self._render:
          cid = self._p.connect(self._p.SHARED_MEMORY)
          if (cid<0):
             id = self._p.connect(self._p.GUI)
          self._p.resetDebugVisualizerCamera(1.,50,-41,[0.1,-0.2,-0.1])
        else:
          self._p.connect(self._p.DIRECT)
        
        ##### loading the secne #####
        self.models = {
          "robot":{
            "position":[0,0,0],
            "orientation": self._p.getQuaternionFromEuler([0,0,0])
          },
          "object":{
            # TODO
          },
          "weees":{
            "id":{
              "FF":None,
              "MF":None,
              "RF":None,
              "TH":None
            }
          }
      }
        self.load_scene()
        ##### setting up random start ####
        print("adaptive_task_parameter_flag:: ",adaptive_task_parameter_flag)
        self.adaptive_task_parameter_flag = adaptive_task_parameter_flag
        self.finger_random_start = FingersRandomStart(finger_obj=self.controller,
                                        adaptive_task_parameter_flag=self.adaptive_task_parameter_flag,
                                        use_lower_limit=atp_use_lower_limit,
                                        neighbour_radius=atp_neighbour_radius,
                                        atp_num_success_required = atp_num_success_required,
                                        atp_sphare_thinkness = atp_sphare_thinkness
                                        
                                       )  
        #debug transformation 
        if self.debug:
          from tf2_debug_ros import DebugTransformation
          self.debugTF = DebugTransformation(self._p,self.controller)
        #setting sim parameters
       
        self._p.setTimeStep(self._timeStep)
        self._p.setRealTimeSimulation(0)
        self._p.setGravity(0,0,-10)

        self.current_step = 0
        self.max_episode_step = 2000 # an episode will terminate (end) if this number is reached

        self.threshold = 0.01 #the goal has been achievd if the distance between fingertip and goal is less than this

        self.control_delay = 10 # this term contorls how often agent gets to interact with the enviornment
        ###########Workspace_Util###########
        self.workspace_util_obj ={
            "fingers":FingerWorkspace_Util(),
            "thumb"  :ThumbWorkspace_Util()
        }
        ###########setting up state space###########
        
        self.obs_obj = {
            "fingers":FingersObservation(self._p,self.controller,workspace_util=self.workspace_util_obj["fingers"],obs_mode=obs_mode["fingers"]),
            "thumb"  :ThumbObservation  (self._p,self.controller,workspace_util=self.workspace_util_obj["thumb"],obs_mode=obs_mode["thumb"])
        }
        os_figners = self.obs_obj["fingers"].set_configuration()
        os_th = self.obs_obj["thumb"].set_configuration()
      
        self.observation_low  = (os_figners.low.tolist() )*3+os_th.low.tolist()
        self.observation_high = (os_figners.high.tolist())*3+os_th.high.tolist()
       
        self.observation_space = spaces.Box(np.array(self.observation_low), np.array(self.observation_high))

        print("self.observation_space.shape:: ",self.observation_space.shape)
        # sys.exit()
        ###########setting up action space###########
        
        print("Intializing action")
        self._action_obj = {
            "fingers":FingersAction(action_mode = action_mode,symitric_action = symitric_action,controller_obj = self.controller, workspace_util=self.workspace_util_obj["fingers"]),
            "thumb"  :ThumbAction  (action_mode = action_mode,symitric_action = symitric_action,controller_obj = self.controller)
        }
        as_fingers = self._action_obj["fingers"].set_configuration()
        as_th = self._action_obj["thumb"].set_configuration()
      
        self.action_low  = (as_fingers.low.tolist()) *3 +as_th.low.tolist()
        self.action_high = (as_fingers.high.tolist())*3 +as_th.high.tolist()
       
        self.action_space = spaces.Box(np.array(self.action_low), np.array(self.action_high))

        print("HandGymEnv::self.action_space:: ",self.action_space )
        
        ############# random start ############
        self.random_start_obj = {
          "fingers": FingersRandomStart(self.controller,
                                        adaptive_task_parameter_flag = self.adaptive_task_parameter_flag,
                                        use_lower_limit=atp_use_lower_limit,
                                        neighbour_radius=atp_neighbour_radius,
                                        atp_num_success_required = atp_num_success_required,
                                        atp_sphare_thinkness = atp_sphare_thinkness
                                        ),
          "thumb"  : ThumbRandomStart(self.controller,
                                        adaptive_task_parameter_flag = self.adaptive_task_parameter_flag,
                                        use_lower_limit=atp_use_lower_limit,
                                        neighbour_radius=atp_neighbour_radius,
                                        atp_num_success_required = atp_num_success_required,
                                        atp_sphare_thinkness = atp_sphare_thinkness
                                        )
        }

        ############ Perofrmance Metric #######
        # TODO :write performance metric

        ###########setting up Reward###########
        self.reward_obj = Reward(reward_mode) 
        ########### History ################
        self._history = {
          "last_act":{
            "FF":   [0]*4,
            "MF":   [0]*4,
            "RF":   [0]*4,
            "TH":   [0]*4,
          },

          "last_last_act":{
            "FF":   [0]*4,
            "MF":   [0]*4,
            "RF":   [0]*4,
            "TH":   [0]*4,
          }
        }
        
    def reset(self,seed=None,options=None):
        if seed is not None:
          # If you use any random numbers, seed them here, e.g.
          import random
          import numpy as np
          random.seed(seed)
          np.random.seed(seed)
        #resetting number of steps in an episode
        self.current_step = 0
        ###########getting random parameters for this episode###########
        joint_values=[]
        
        ####fingers
        for finger_name in ["FF","MF","RF"]:
          if self.random_robot_start:
            joint_values += self.random_start_obj["fingers"].get_joint_values()
          else:
            joint_values += [0]*4

          if not self._orchestrator_mode:
            goal = self.random_start_obj["fingers"].get_goal(finger_name)
            self.goals["locations"]["current"][finger_name] = goal
        ##### thumb
        if self.random_robot_start:
          joint_values += self.random_start_obj["thumb"].get_joint_values()
        else:
          joint_values += [0]*4

        if not self._orchestrator_mode:
          goal = self.random_start_obj["thumb"].get_goal()
          self.goals["locations"]["current"]["TH"] = goal
        ##############loading new and previous goal##################
        if not self._orchestrator_mode:
          for finger_name in ["FF","MF","RF","TH"]:
            self.change_goal_location(finger_name)

            goal_has_chaned = self.goals["locations"]["current"][finger_name] != self.goals["locations"]["previous"][finger_name]

            if goal_has_chaned:
              if self.goals["locations"]["previous"][finger_name] ==None:
                self.goals["locations"]["previous"][finger_name] = self.goals["locations"]["current"][finger_name]

              self.change_goal_location(finger_name,True)
              self.goals["locations"]["previous"][finger_name] = self.goals["locations"]["current"][finger_name]

             
        ############resetting the robot at the begining of each episode#######
        self.controller.reset(joint_values)
        self._p.stepSimulation()
        ############getting robot state##############
        initla_state = self.getObservation()
   
        return initla_state, {}

    def step(self,action):
      action = list(action)

      # print(f"HandGymEnv::action::type::{type(action)}")
      # print(f"HandGymEnv::action::{action}")
      # print(f"HandGymEnv::action::shape::{len(action)}")

      # print("step::action::len:: ",len(action))
      # print("step::action:: ",action)

      #### applying action #####
      action_dic = {
        "fingers":{
          "FF":action[:4],
          "MF":action[4:8],
          "RF":action[8:12]
        },
        "thumb":action[12:]
      }

      # print("step::action_dic:: ",action_dic)

      fingers = ["FF","MF","RF"]
      for i in range(self.control_delay):
        fingers_command = []
        for finger in fingers:
          finger_command = self._action_obj["fingers"].process_action(action_dic["fingers"][finger],finger)
          # print("\n\n")
          # print("finger_command::type:: ",type(finger_command))
          # print("\n\n")
          fingers_command += finger_command
        
        # print("\n\n")
        # print("step::fingers_command::len ",len(fingers_command))
        # print("step::fingers_command:: ",fingers_command)
        # print("\n\n")

        self.controller.applyActionToFingers(fingers_command)
        self.controller.applyActionToThumb(self._action_obj["thumb"].process_action(action_dic["thumb"]))

        self._p.stepSimulation()

      self.current_step +=1
      distance_from_fingertip_to_goal = {
        "FF":None,
        "MF":None,
        "RF":None,
        "TH":None,
      }

      ###########################
      for finger_name in ["FF","MF","RF","TH"]:
        if finger_name == "TH":
          self.obs_obj["thumb"].update_goal_and_finger_name(self.goals["ids"]["current"]["TH"])
          distance_from_fingertip_to_goal[finger_name] = self.obs_obj["thumb"].get_distance_from_fingertip_to_goal()
        
        self.obs_obj["fingers"].update_goal_and_finger_name(finger_name,self.goals["ids"]["current"][finger_name])  
        distance_from_fingertip_to_goal[finger_name] = self.obs_obj["fingers"].get_distance_from_fingertip_to_goal()
        
      goal_is_achived = self.is_goal_achived(distance_from_fingertip_to_goal)

      ##### observation #####
      obs = self.getObservation()
      ##### termination ####
      done = self.termination(goal_is_achived)
      truncated = self.current_step > self.max_episode_step and not done
      #### reward #####
      reward,_ = self.reward(distance_from_fingertip_to_goal,goal_is_achived)


      ############## history ###########

      for finger_name in ["FF","MF","RF","TH"]:
        if finger_name == "TH":
          self.obs_obj["thumb"].update_goal_and_finger_name(self.goals["ids"]["current"]["TH"])
          self._history["last_act"][finger_name] =  self.obs_obj["thumb"].get_joint_values()
          self._history["last_last_act"][finger_name] = self._history["last_act"][finger_name]
          continue

        self.obs_obj["fingers"].update_goal_and_finger_name(finger_name,self.goals["ids"]["current"][finger_name])
        self._history["last_act"][finger_name] =  self.obs_obj["fingers"].get_joint_values()
        self._history["last_last_act"][finger_name] = self._history["last_act"][finger_name]

      info = {"action":action}

      return obs, reward, done, truncated, info

    def render(self):
      pass
    
    def getObservation(self):
      obs_fingers = {
      "FF":None,
      "MF":None,
      "RF":None
      }
      obs_thumb = None
    
      for finger_name in ["FF","MF","RF"]:
        goalId = self.goals["ids"]["current"][finger_name]
        self.obs_obj["fingers"].update_goal_and_finger_name(finger_name,goalId)  
        history = {
          "last_act": self._history["last_act"][finger_name],
          "last_last_act":self._history["last_last_act"][finger_name]
        }
        obs_fingers[finger_name] =self.obs_obj["fingers"].get_state(history)
      
      history = {
          "last_act": self._history["last_act"]["TH"],
          "last_last_act":self._history["last_last_act"]["TH"]
      }
      goalId = self.goals["ids"]["current"]["TH"]
      self.obs_obj["thumb"].update_goal_and_finger_name(goalId)
      obs_thumb = self.obs_obj["thumb"].get_state(history)
    
      obs_dic = {
        "fingers":obs_fingers,
        "thumb"  :obs_thumb
      }

      # print("\n\n")
      # print("getObservation::obs_dic[fingers][FF]::type:: ",type(obs_dic["fingers"]["FF"]))
      # print("getObservation::obs_dic[thumb]::type:: ",type(obs_dic["thumb"]))
      # print("\n\n")


      # print(f"obs_dic[fingers][FF]::{obs_dic['fingers']['FF']}")
      # print(f"obs_dic[fingers][FF]::tolist{obs_dic['fingers']['FF'].tolist()}")
      # print(f"obs_dic[fingers][MF]::{obs_dic['fingers']['MF']}")


      obs = obs_dic["fingers"]["FF"].tolist()+obs_dic["fingers"]["MF"].tolist()+obs_dic["fingers"]["RF"].tolist()+obs_dic["thumb"].tolist()

      obs = np.array(obs)
      # print("\n\n")
      # print("getObservation::oba::shape",obs.shape)
      # print("\n\n")
      
      # print("\n\n")
      # print("obs:: ",obs)
      # print("\n\n")
      
      return obs 

    def reward(self,distance_from_fingertip_to_goal,goal_is_achived):
      total_reward =0
      reward = {
        "FF":None,
        "MF":None,
        "RF":None,
        "TH":None,
      }
      for finger_name in reward.keys():
        reward[finger_name] = self.reward_obj.get_reward(distance_from_fingertip_to_goal[finger_name],goal_is_achived[finger_name])
        total_reward +=reward[finger_name] 
      return total_reward,reward
    
    def termination(self,goal_is_achived):
      ###########recording performance#################
      # TODO: finish performance metric


      ################check if goal is achived #######
      
      goals_are_achived = True
      for finger_name in goal_is_achived.keys():
        if goal_is_achived[finger_name] == False:
          goals_are_achived = False
          break

      if self.current_step > self.max_episode_step or goals_are_achived : #episdoe will end
        ###############Adaptive task parameter###########
        
        for finger_name in ["FF","MF","RF"]:
          if goal_is_achived[finger_name]:
            self.random_start_obj["fingers"].increment_success(finger_name)

        if  goal_is_achived["TH"]:
          self.random_start_obj["thumb"].increment_success()  
          
        
        return True

      
        #################################################
      
      return False 
    
    ########utility function#########
    def load_scene(self):
      ####### load floor #########
      urdfRoot=pybullet_data.getDataPath()
      self.plane_id = self._p.loadURDF(os.path.join(urdfRoot,"plane.urdf"),[0,0,0])
      ####### load robot #########
      # TODO: after wrting controller finish this
      self.controller = Hand(self._p,self.models["robot"])
      ####### load goals #########
      goal_path = resource_filename(__name__,"/goal/goal.sdf")
      previous_goal_path = resource_filename(__name__,"/goal/previous_goal .sdf")
      for finger_name in ["FF","MF","RF","TH"]:
        self.goals["ids"]["current"][finger_name] = self._p.loadSDF(goal_path)[0]
        self.goals["ids"]["previous"][finger_name] = self._p.loadSDF(previous_goal_path)[0]


    def change_goal_location(self,finger_name,previous=False):
     
      euler_angle = [0,0,0]
      quaternion_angle = self._p.getQuaternionFromEuler(euler_angle)

      if previous:
        goalId =  self.goals["ids"]["previous"][finger_name]
        goal_loc = self.goals["locations"]["previous"][finger_name]
        self._p.resetBasePositionAndOrientation(goalId, goal_loc,quaternion_angle)
      else:
        goalId =  self.goals["ids"]["current"][finger_name]
        goal_loc = self.goals["locations"]["current"][finger_name]
        self._p.resetBasePositionAndOrientation(goalId, goal_loc,quaternion_angle)  

    def set_goal_location(self, goals:np.array):
      # print(f"HandGymEnv::set_goal_location::type::{type(goals)}")
      # print(f"HandGymEnv::set_goal_location::shape::{goals.shape}")
      # print(f"HandGymEnv::set_goal_location::goals::{goals}")
      goals_dic = {
        "FF":goals[:3],
        "MF":goals[3:6],
        "RF":goals[6:9],
        "TH":goals[9:]
      }
      euler_angle = [0,0,0]
      quaternion_angle = self._p.getQuaternionFromEuler(euler_angle)
      for finger_name in ["FF","MF","RF","TH"]:
        goalId =  self.goals["ids"]["current"][finger_name]
        goal_loc = goals_dic[finger_name]
        self._p.resetBasePositionAndOrientation(goalId, goal_loc,quaternion_angle)



    def is_goal_achived(self,distance_from_fingertip_to_goal):
      flags = {
        "FF":None,
        "MF":None,
        "RF":None,
        "TH":None,
      }
      
      for finger_name in ["FF","MF","RF","TH"]:
        dist_to_goal = distance_from_fingertip_to_goal[finger_name]
        if dist_to_goal <0.001:
          flags[finger_name] = True
        else:
          flags[finger_name] = False

      return flags


