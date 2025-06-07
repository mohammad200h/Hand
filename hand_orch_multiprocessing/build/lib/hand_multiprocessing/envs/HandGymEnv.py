import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import sys

from pkg_resources import resource_string,resource_filename
import time
import random
import math
import gym
from gym import spaces
from gym.utils import seeding
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



class FingerWorkspace_Util():
  def __init__(self):
    self.point_cloud = None
    path = resource_filename(__name__,"/model/FF.yml")
    with open(path, "r") as stream:
      try:
         self.point_cloud = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc)
   
    points  = np.array(self.point_cloud["vertix"])
  
    self.x = points[:,0]
    self.y = points[:,1]
    self.z = points[:,2]

    self.ws_pointcould = {
      "ff":self.get_points_for_finger("ff"),
      "mf":self.get_points_for_finger("mf"),
      "rf":self.get_points_for_finger("rf") 
    }
   
  def get_max_min_xyz(self):
    max = [np.max(self.x),np.max(self.y),np.max(self.z)]
    min = [np.min(self.x),np.min(self.y),np.min(self.z)]

    return max,min

  def get_max_min_xyz_for_finger(self,finger_name):
    x,y,z = self.get_points_for_finger(finger_name)
    
    max = [np.max(x),np.max(y),np.max(z)]
    min = [np.min(x),np.min(y),np.min(z)]

    return max,min
  
  def get_points_for_finger(self,finger_name):
    point_cloud = None
    path = None
    if finger_name =="ff":
      path = resource_filename(__name__,"/model/FF.yml")
    elif finger_name =="mf":
      path = resource_filename(__name__,"/model/MF.yml")
    elif finger_name =="rf":
      path = resource_filename(__name__,"/model/RF.yml")
    else:
      # TODO:Raise an error
      print("wrong finger!")

    with open(path, "r") as stream:
      try:
         point_cloud = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc)
   
    points  = np.array(point_cloud["vertix"])
  
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    return x,y,z

  def get_ws_pointcould(self):
    return self.ws_pointcould

class FingersObservation():
    def __init__(self,physic_engine,finger_obj,workspace_util,obs_mode ="finger_joint_and_xyz"):
   
        self._p = physic_engine
        self.controller = finger_obj
        self.finger_name =None
        self.goalId = None
        
        self.workspace_util =workspace_util
        self.ws_max,self.ws_min = self.workspace_util.get_max_min_xyz()
        self.TF = CoordinateFrameTrasform(self._p,self.controller)


        self.obs_modes = ["finger_joint_and_xyz"]
        self.obs_mode =obs_mode
        self.observation_space = None
        

        self.state_limit={
           "joint":{
            #[knuckle,proximal,middle,distal]
            "high":[0.349066,1.5708,1.5708,1.5708],
            "low":[-0.349066,0     ,0     ,0     ]
           },
           "xyz":{
            "high":self.ws_max,
            "low":self.ws_min
           },
           "finger_index":{
              # 0,1,2
               "high":[2],
               "low":[0]
           }
        }

    def update_goal_and_finger_name(self,name,goalId):
      self.finger_name =name
      self.goalId = goalId

    def get_state(self):
      state =None
      
      if self.obs_mode == "finger_joint_and_xyz":
        state = self.finger_joint_and_xyz(self.goalId)

      return state
    
    def set_obs_mode(self,mode):
      
      if self.obs_mode == "finger_joint_and_xyz":
        self.state_high = np.array(self.state_limit["joint"]["high"]+
                                   self.state_limit["xyz"]["high"]+
                                   self.state_limit["finger_index"]["high"]
                                  ,dtype=np.float32
                                  )
        self.state_low = np.array(self.state_limit["joint"]["low"]+
                                   self.state_limit["xyz"]["low"]+
                                   self.state_limit["finger_index"]["low"]
                                  ,dtype=np.float32
                                  )
      else:
        # TODO: rasie an error 
        print("not supported!") 

      self.observation_space = spaces.Box(self.state_low,self.state_high)
      print("obs_mode:: ",self.obs_mode)
      print("observation_space:: ",self.observation_space)

    def set_configuration(self):
      self.set_obs_mode(self.obs_mode)
      return self.observation_space
    
    # utility functions 
    def get_state_for_perfomance_metric(self):
      _,state_dic = self.finger_joints_and_distnace()

      return state_dic
    
    def finger_joints_and_distnace(self):
        state_dic ={
            # 4 joints 1 dist 1 finger index => dim= 6 
            "joints":self.get_joint_values(),
            "distance":self.get_distance_from_fingertip_to_goal(),
            "finger_index":self.get_finger_index(self.finger_name)
        }
        # print("finger_joints_and_distnace::state_dic:: ",state_dic)
        state = np.array(state_dic["joints"]+[state_dic["distance"]]+[state_dic["finger_index"]],dtype=np.float32)
        # print("finger_joints_and_distnace::state:: ",state)
        return state,state_dic
    
    def finger_joint_and_dist_xyz(self,goalId):
      pose = self.TF.get_in_local_finger_frame(goalId,self.finger_name)
      pos  = [abs(pose.x),abs(pose.y),abs(pose.z)]

      state_dic = {
            # 4 joints 3 dist 1 finger index => dim = 8
            "joints":self.get_joint_values(),
            "dist_xyz":pos,
            "finger_index":self.get_finger_index(self.finger_name)
      }
      # print("finger_joint_and_dist_xyz::state_dic:: ",state_dic)
      state = np.array(state_dic["joints"]+state_dic["dist_xyz"]+[state_dic["finger_index"]],dtype=np.float32)
      # print("finger_joint_and_dist_xyz::state:: ",state)
      return state
    
    def finger_joint_and_xyz(self,goalId):
      pose = self.TF.get_in_local_finger_frame(goalId,self.finger_name)
      pos  = [pose.x,pose.y,pose.z]

      state_dic = {
            # 4 joints 3 xyz 1 finger index => dim = 8
            "joints":self.get_joint_values(),
            "xyz":pos,
            "finger_index":self.get_finger_index(self.finger_name)
      }
      
      # print("finger_joint_and_xyz::state_dic[joints]:: ",state_dic["joints"])
      # print("finger_joint_and_xyz::state_dic[xyz]:: ",state_dic["xyz"])
      # print("finger_joint_and_xyz::state_dic[finger_index]:: ",state_dic["finger_index"])

      state = np.array(state_dic["joints"]+state_dic["xyz"]+[state_dic["finger_index"]],dtype=np.float32)
      # print("finger_joint_and_xyz::state:: ",state)
      return state

    def get_distance_from_fingertip_to_goal(self):
        
        goal_pos = self.get_goal_pos()
        finger_tip_pos =  self.controller.get_observation_finger_tip(self.finger_name)

        x_dist = goal_pos[0] - finger_tip_pos[0]
        y_dist = goal_pos[1] - finger_tip_pos[1]
        z_dist = goal_pos[2] - finger_tip_pos[2]
        dist =  math.sqrt(x_dist**2+y_dist**2+z_dist**2)

        return dist

    def get_joint_values(self):
        # print("get_joint_values::self.finger_name:: ",self.finger_name)
        return  self.controller.get_Observation_finger(self.finger_name)

    def get_finger_index(self,finger_name):
        finger_list=["FF","MF","RF"]
        
        return finger_list.index(finger_name)
    
    def get_goal_pos(self):
        goal_state =  p.getBasePositionAndOrientation(self.goalId)
        pos = goal_state[0]
        orn = goal_state[1]
        
        return pos
   
class FingersAction():
  """
  All the action processing will be done here in order 
  to clean up the code and simplify the controller
  """
  def __init__(self,action_mode,symitric_action,controller_obj):
   
    

    self.finger_offsets = {
      "ff":[0,0,0],
      "mf":[0.022,0.002536,0.003068],
      "rf":[0.044,0,0]
    }

    self.action_mode = action_mode
    self.symitric_action = symitric_action
    self.controller_obj = controller_obj
    self._action_modes = ["jointControl"]
    self._action_limit ={
      
      "joint":{
        "high":[0.349066,1.5708,1.5708,1.5708],
        "low" :[-0.349066,0     ,0     ,0    ]
      }
    }

   
    
  def process_action(self,command,finger_name):
    """
    will convert action to continious joint values
    The command will be processed differently according to action mode
    """
    processed_command = []
    if self.action_mode =="jointControl":
      processed_command = self.process_jointControl(command)
  

    return processed_command
  
  def set_configuration(self):
    """
    return  different setting for action all together
    """
    self.set_action_mode()
    self.set_To_symitric()

    return self.action_space

  # utility functions 
  def set_action_mode(self):
    action_mode = self.action_mode
    if action_mode == "jointControl": 
      self.action_high_non_symitric = np.array(self._action_limit["joint"]["high"],dtype=np.float32)
      self.action_low_non_symitric  = np.array(self._action_limit["joint"]["low"],dtype=np.float32)
    
  
    self.action_high = self.action_high_non_symitric
    self.action_low = self.action_low_non_symitric  
    self.action_space = spaces.Box(self.action_low, self.action_high)

  def set_To_symitric(self):

    if self.symitric_action == True:
        self.action_high = np.array(len(self.action_high_non_symitric) *[1],dtype=np.float32)
        self.action_low = np.array(len(self.action_low_non_symitric ) *[-1],dtype=np.float32)
    else:
        self.action_high = self.action_high_non_symitric
        self.action_low = self.action_low_non_symitric

    self.action_space = spaces.Box(self.action_low, self.action_high)

  def process_jointControl(self,command): 
      """
      We only need to check if incomming action is symiteric is yes then we need to convert it to
      Non symitric since robot commands are non symitric
      """
      processed_command = []

      if self.symitric_action ==False:
        processed_command = command.tolist()
      else:
        processed_command = self.convertTo_non_symitric_action(command)

      return processed_command
  
  def process_IK(self,command,finger_name): 
    # print("process_IK::finger_name:: ",finger_name)
    processed_command = []

    if self.symitric_action ==False:
      processed_command = command
    else:
      processed_command = self.convertTo_non_symitric_action(command)

    return self.proccess_IK_General(processed_command,finger_name)

  def process_delta_jointControl(self,delta_command):
    current_state_of_joints = self.get_current_state_of_joints()

    processed_command = current_state_of_joints  
    for index,dc in enumerate(delta_command):
      processed_command[index] +=dc

    if self.joint_commnd_is_within_upper_and_lower_joint_limit(processed_command):
      return processed_command
    else:
      processed_command = self.get_closest_viable_jointcommand(processed_command)
      return processed_command



  
  def process_delta_IK(self,delta_command,finger_name): 
    # print("process_delta_IK::finger_name:: ",finger_name)
    # print("process_delta_IK::ws_limit:: ",self._action_limit["ws"][finger_name])

    # print("process_delta_IK::delta_command:: ",delta_command)
    current_state_of_ee = self.get_current_state_of_ee()
    # print("process_delta_IK::current_state_of_ee:: ",current_state_of_ee)
    processed_command = list(current_state_of_ee)  # xyz
    for index,dc in enumerate(delta_command):
      processed_command[index] +=dc

    # print("process_delta_IK::processed_command:: ",processed_command)

    return self.proccess_IK_General(processed_command,finger_name)
  
  # utility functions 

  def proccess_IK_General(self,command,finger_name):
    # applying finger offset to command 
    processed_command = self.apply_command_offset(finger_name,command)

    # ee command to joint value command conversion 
    robot_id = self.controller_obj._robot_id
    index_of_ee = self.get_index_of_ee(finger_name)  


    return self.get_ik_values_for_finger(finger_name,robot_id,index_of_ee,processed_command)

  def apply_command_offset(self,finger_name,command):
    """
    is this nessessary?
    The agent operate based on workspace on off the fingers. To be exact ff. Therefor the points it procudes for the ik 
    will be base on that. If we are using a different finger then we need to shift the ik ee pose so that it is base on workspace
    for that finger
    """
    offset = self.finger_offsets[finger_name]
    command[0] += offset[0]
    command[1] += offset[1]
    command[2] += offset[2]

    # make sure the command is withing workspace limits
    if self.ee_command_is_within_ws_limits(command,finger_name):
      return command
    else:
      command = self.get_closet_viable_command(command,finger_name)
      return command
      
  def get_index_of_ee(self,finger_name):
    ee_name = None

    if finger_name == "ff":
      ee_name = "fingertip_FF"
    elif finger_name == "mf":
      ee_name = "fingertip_MF"
    elif finger_name == "rf":
      ee_name = "fingertip_RF"
    else:
      # TODO: raise an error 
      print("wrong finger")
    
    return self.controller_obj.get_endEffectorLinkIndex(ee_name)

  def get_ik_values_for_finger(self,finger_name,robot_id,index_of_ee,pos):

    joint_command_for_finger = None
    joint_commands_for_full_robot = p.calculateInverseKinematics(robot_id,index_of_ee,pos)

    if finger_name =="ff":
      joint_command_for_finger = joint_commands_for_full_robot[:4]
    elif finger_name =="mf": 
      joint_command_for_finger = joint_commands_for_full_robot[4:4*2]
    elif finger_name =="rf": 
      joint_command_for_finger = joint_commands_for_full_robot[4*2:]
    else:
      # TODO: raise an error 
      print("not the right finger")
    
    return joint_command_for_finger
    
  def convertTo_non_symitric_action(self,action):

      noneSymitric_action = len(self.action_high_non_symitric)*[0]
      for i in range(0, len(self.action_high)):
        noneSymitric_action[i] = ((action[i]-self.action_low[i])/(self.action_high[i]-self.action_low[i]))*(self.action_high_non_symitric[i]-self.action_low_non_symitric[i])+self.action_low_non_symitric[i]
      return noneSymitric_action

  def ee_command_is_within_ws_limits(self,command,finger_name):
   
    max,min = self._action_limit["ws"][finger_name]
    
    for index,com in enumerate(command):
      if not (com  >= min[index] and com <= max[index]): 
        # print("min:: ",min[index])
        # print("max:: ",max[index])
        # print("com:: ",com)
        return False

    return True

  def get_closet_viable_command(self,command,finger_name):
    finger_ws_pointcloud = self.ws_pointcoulds[finger_name]
    x_list,y_list,z_list  = finger_ws_pointcloud

    new_x = self.get_closet_number_in_the_numpy_list(x_list,command[0])
    new_y = self.get_closet_number_in_the_numpy_list(y_list,command[1])
    new_z = self.get_closet_number_in_the_numpy_list(z_list,command[2])

    return new_x,new_y,new_z

  def joint_commnd_is_within_upper_and_lower_joint_limit(self,jointcommand):
    upper_lmit = self._action_limit["joint"]["high"]
    lower_lmit = self._action_limit["joint"]["low"]

    for index,jc in enumerate(jointcommand):
      if not (jc >= lower_lmit[index] and jc <= upper_lmit[index]):
        return False

    return True
        
  def get_closest_viable_jointcommand(self,jointcommand):
    processed_command = [0]*4
    upper_lmit = self._action_limit["joint"]["high"]
    lower_lmit = self._action_limit["joint"]["low"]

    for index,jc in enumerate(jointcommand):
      if jc >= lower_lmit[index] and jc <= upper_lmit[index]:
        processed_command[index] = jc
      elif jc < lower_lmit[index]:
        processed_command[index] = lower_lmit[index]
      elif jc > upper_lmit[index]:
        processed_command[index] =jc > upper_lmit[index]
      else:
        # TODO: raise an error 
        print("this is not a valid joint command")

    return processed_command
  
  def get_closet_number_in_the_numpy_list(self,np_list,num):
    # https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
    # calculate the difference array
    difference_array = np.absolute(np_list-num)

    # find the index of minimum element from the array
    index = difference_array.argmin()

    new_num = np_list[index]

    return new_num

  def get_current_state_of_ee(self):
    return self.controller_obj.get_observation_finger_tip()
  
  def get_current_state_of_joints(self):
    return self.controller_obj.getObservation()

class FingersBasicGoalGenerator():
  def __init__(self):
    print("using BasicGoalGenerator")
    # sys.exit()
    self.finger_list = ["FF","MF","RF"]
    self.point_cloud ={
      "FF":self.load_all_goals("FF"),
      "MF":self.load_all_goals("MF"),
      "RF":self.load_all_goals("RF")
    }

  def get_goal(self,finger_name):
    goal  = random.choice(self.point_cloud[finger_name])
    return goal

  def load_all_goals(self,finger_name):

    point_cloud = None
        
    if finger_name not in self.finger_list:
        #Todo: raise an error
        print("wrong finger name")
    #load point cloud for finger
    path = resource_filename(__name__,"/model/"+finger_name+".yml")
    with open(path, "r") as stream:
        try:
           point_cloud = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return point_cloud["vertix"]
    
class FingersAdaptiveTaskParameter():
  def __init__(self,finger_obj,
               neighbour_radius,
               num_success_required=2,
               use_lower_limit=False,
               sphare_thinkness=0.1
              ):
    print("using AdaptiveTaskParameter")
    # sys.exit()
    self.controller = finger_obj
    self.finger_list = ["FF","MF","RF"]
    self.num_success_required = num_success_required
    self.use_lower_limit = use_lower_limit
    self.neighbour_radius = neighbour_radius
    self.neighbour_radius_lower_limit = self.neighbour_radius -sphare_thinkness
    self.intial_neighbour_radius = self.neighbour_radius

    # state 
    self.fingertips_intial_xyz = self.fingers_initial_state()
    self.starting = True
    self.success_counter = {
      "FF":0,
      "MF":0,
      "RF":0
    }


    self.point_cloud ={
      "FF":self.load_all_goals("FF"),
      "MF":self.load_all_goals("MF"),
      "RF":self.load_all_goals("RF")
    }
    self.point_cloud_copy = self.point_cloud.copy()


    self.current_goal = {
      "FF":self.choose_closest_goal_to_figner("FF"),
      "MF":self.choose_closest_goal_to_figner("MF"),
      "RF":self.choose_closest_goal_to_figner("RF")
    }
    
  def choose_closest_goal_to_figner(self,finger_name):
    """
    choose closet goal to the finger 
    """
    finger_pos_np = np.array(self.fingertips_intial_xyz[finger_name])
    dist_mat = self.calculate_dist_matrix(finger_name,finger_pos_np)
    
    index_min = np.argmin(dist_mat)

    print("choose_closest_goal_to_figner::index_min:: ",index_min)

    goal = self.point_cloud[finger_name][index_min]

    print("choose_closest_goal_to_figner::goal:: ",goal)

    return goal

  def get_goal(self,finger_name):
    
    if self.starting:
      self.starting = False
      return self.current_goal[finger_name]

    if self.success_counter[finger_name]>=self.num_success_required:
      self.update_goal_on_success(finger_name)
    
    return self.current_goal[finger_name]

  def increment_success(self,finger_name):
    # log.debug("AdaptiveTaskParameter::increment_success::finger_name:: {} ::success_counter:: {}".format(finger_name,self.success_counter[finger_name]))
    self.success_counter[finger_name] +=1

  def reset_counter_becasue_of_failiur(self,finger_name):
    # log.debug("AdaptiveTaskParameter::reset_counter_becasue_of_failiur::finger_name:: {}".format(finger_name))
    self.success_counter[finger_name] =0


  def update_goal_on_success(self,finger_name):
    self.neighbour_radius = self.intial_neighbour_radius 
    old_goal = self.current_goal[finger_name][:]
    
    if len(self.point_cloud[finger_name])>1:
      self.remove_goal(finger_name)
      if self.use_lower_limit:
        self.current_goal[finger_name] = self.get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit(finger_name)
      else:
        self.current_goal[finger_name] = self.get_a_goal_in_neighbourhood_of_current_goal(finger_name)
      if self.current_goal[finger_name] ==old_goal:
        print("neighbour_radius is too small")
      self.success_counter[finger_name] = 0
    else:
      self.sample_at_random_if_all_goals_achived(finger_name)

  # utility function
  def get_a_goal_in_neighbourhood_of_current_goal(self,finger_name):
    # find list of candidates
    
    indexes  = self.get_neighbourhood_indexs(finger_name)

    # print("get_a_goal_in_neighbourhood_of_current_goal::indexes:: ",indexes)
    # choose a candidate at random 
    candidate_index = random.choice(indexes)
    candidate       = self.point_cloud[finger_name][candidate_index] 

    return candidate
 
  def load_all_goals(self,finger_name):

    point_cloud = None
        
    if finger_name not in self.finger_list:
        #Todo: raise an error
        print("wrong finger name")
    #load point cloud for finger
    path = resource_filename(__name__,"/model/"+finger_name+".yml")
    with open(path, "r") as stream:
        try:
           point_cloud = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return point_cloud["vertix"]
    
  def calculate_dist_matrix(self,finger_name,target_xyz):
    
    dist_mat = self.point_cloud[finger_name] - target_xyz
    dist_mat = np.power(dist_mat,2)
    dist_mat = np.sum(dist_mat,axis=1) #sum along row
    dist_mat = np.power(dist_mat,1/2)

    return dist_mat

  def fingers_initial_state(self):
    postions = self.controller.get_Observation_fingertips()
    pos_dic = {
      "FF":postions["FF"],
      "MF":postions["MF"],
      "RF":postions["RF"]
    }

    return pos_dic

  def get_neighbourhood_indexs_with_band(self,finger_name):
    dist_mat = self.calculate_dist_matrix(finger_name,np.array(self.current_goal[finger_name]))
    ul_indexes  = np.where(dist_mat<self.neighbour_radius )[0].tolist()
    ll_indexes  = np.where(dist_mat>self.neighbour_radius_lower_limit )[0].tolist()
    indexes = self.common_member(ul_indexes,ll_indexes)
    # print("get_neighbourhood_indexs::indexes ",indexes)
    if len(indexes)==0:
      print("did not find a neighbure. Gowing  neighbour_radius inorder to find a new neighboure ")
      self.neighbour_radius *=2
      print("new search radious:: ", self.neighbour_radius)
      return self.get_neighbourhood_indexs_with_band(finger_name)

    return indexes

  def get_neighbourhood_indexs(self,finger_name):
    dist_mat = self.calculate_dist_matrix(finger_name,np.array(self.current_goal[finger_name]))
    indexes  = np.where(dist_mat<self.neighbour_radius)[0].tolist()
    # print("get_neighbourhood_indexs::indexes ",indexes)
    if len(indexes)==0:
      print("did not find a neighbure. Gowing  neighbour_radius inorder to find a new neighboure ")
      self.neighbour_radius *=2
      print("new search radious:: ", self.neighbour_radius)
      return self.get_neighbourhood_indexs(finger_name)

    return indexes

  def get_neighbourhood(self,finger_name):
    indexes  = self.get_neighbourhood_indexs(finger_name)
    goals =  np.array(self.point_cloud[finger_name])
    neighbourhood =goals[indexes] 
    # print("get_neighbourhood::neighbourhood:: ",neighbourhood)

    return neighbourhood

  def remove_goal(self,finger_name):
    # print("remove_goal::called")
    current_goal = self.current_goal[finger_name]
    goals = np.array(self.point_cloud[finger_name])
    indexs = np.where(np.all(goals==current_goal,axis=1))[0].tolist()
    # print("remove_goal::index:: ",indexs)
    goals = np.delete(goals ,indexs,axis =0)

    
    self.point_cloud[finger_name]= goals.tolist()
    # print("remove_goal::self.point_cloud[finger_name] ",self.point_cloud[finger_name])

  def sample_at_random_if_all_goals_achived(self,finger_name):
    
    goal = random.choice(self.point_cloud_copy[finger_name])
    return goal

  def get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit(self,finger_name):
     # find list of candidates
    
    indexes  = self.get_neighbourhood_indexs_with_band(finger_name)

    # print("get_a_goal_in_neighbourhood_of_current_goal::indexes:: ",indexes)
    # choose a candidate at random 
    candidate_index = random.choice(indexes)
    candidate       = self.point_cloud[finger_name][candidate_index] 

    return candidate
  
  def common_member(self,a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return list(a_set & b_set)
    else:
        return []
    
class FingersRandomStart():
    def __init__(self,finger_obj,adaptive_task_parameter_flag=False,
                 atp_num_success_required=2,
                 use_lower_limit=False,
                 neighbour_radius=0.05,
                 atp_sphare_thinkness=0.1
                 ):
        
        self.finger_list = ["FF","MF","RF"]
        self.last_finger_used = None 
     
        self._action_limit = {
            #[knuckle,proximal,middle,distal]
            "high":[0.349066,1.5708,1.5708,1.5708],
            "low":[-0.349066,0     ,0     ,0     ]
        }
        self.adaptive_task_parameter_flag = adaptive_task_parameter_flag

        if adaptive_task_parameter_flag:
          self.ATP = FingersAdaptiveTaskParameter(finger_obj,
                                           neighbour_radius,
                                           atp_num_success_required,
                                           use_lower_limit,
                                           atp_sphare_thinkness
                                          )
        else:
          self.BGG = FingersBasicGoalGenerator()
          
    def get_finger_name(self):
        """
        get a new finger position for this episode
        """
        #Pop last from end and add it to front. choose the last in the list 
        # print("get_finger_name::finger_list:: ",self.finger_list)
        finger_name = self.finger_list[-1]
        self.finger_list.pop()  #removing last element
        self.finger_list.insert(0,finger_name)
        # print("get_finger_name::finger_list:: ",self.finger_list)
       
        return finger_name

    def get_joint_values(self):
        """
        a random joint values is generated for new episode 
        """
        joint_values = []
        for i in range(4):
            joint_values.append(random.uniform(self._action_limit["low"][i],self._action_limit["high"][i]))

        return joint_values

    def get_goal(self,finger_name):
        """
        get a new goal position from point could in workspace
        """
        goal = None
        if self.adaptive_task_parameter_flag:
          goal = self.ATP.get_goal(finger_name)
        else:
          goal = self.BGG.get_goal(finger_name)

        return goal

    def increment_success(self,finger_name):
      # log.debug("RandomStart::increment_success::finger_name:: {}".format(finger_name))
      if self.adaptive_task_parameter_flag:
        self.ATP.increment_success(finger_name)

    def reset_counter_becasue_of_failiur(self,finger_name):
      # log.debug("RandomStart::reset_counter_becasue_of_failiur::finger_name:: {}".format(finger_name))
      if self.adaptive_task_parameter_flag:
        self.ATP.reset_counter_becasue_of_failiur(finger_name)


class ThumbWorkspace_Util():
  
  def __init__(self):
    self.point_cloud = None
    self.x ,self.y , self.z = self.get_points_for_finger()

    self.ws_pointcould = {
      "th":self.get_points_for_finger()
    }
   
  def get_max_min_xyz(self):
    max = [np.max(self.x),np.max(self.y),np.max(self.z)]
    min = [np.min(self.x),np.min(self.y),np.min(self.z)]

    return max,min

  def get_points_for_finger(self):
    point_cloud = None
    path = None
  
    path = resource_filename(__name__,"/model/TH.yml")

  

    with open(path, "r") as stream:
      try:
         point_cloud = yaml.safe_load(stream)
      except yaml.YAMLError as exc:
          print(exc)
   
    points  = np.array(point_cloud["vertix"])
  
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]

    return x,y,z

  def get_ws_pointcould(self):
    return self.ws_pointcould

class ThumbObservation():
    def __init__(self,physic_engine,finger_obj,workspace_util,obs_mode ="finger_joint_and_xyz"):
        
        self._p = physic_engine
        self.controller = finger_obj
        self.goalId = None

        self.workspace_util =workspace_util
        self.ws_max,self.ws_min = self.workspace_util.get_max_min_xyz()
        self.TF = CoordinateFrameTrasform(self._p,self.controller)


        self.obs_modes = ["finger_joint_and_xyz"]
        self.obs_mode =obs_mode
        self.observation_space = None
      

        self.state_limit={
           "joint":{
              "high":[1.0472, 1.22173, 0.698132, 1.5708],
              "low":[-1.0472, 0, -0.698132, 0]
           },
           "xyz":{
              "high":self.ws_max,
              "low":self.ws_min
           }
         
        }

    def update_goal_and_finger_name(self,goalId):
      self.goalId = goalId

    def get_state(self):
      state =None
   
      if self.obs_mode == "finger_joint_and_xyz":
        state = self.finger_joint_and_xyz(self.goalId)

      return state
   
    def set_obs_mode(self,mode):
      
      if self.obs_mode == "finger_joint_and_xyz":
        self.state_high = np.array(self.state_limit["joint"]["high"]+
                                   self.state_limit["xyz"]["high"]
                             
                                  ,dtype=np.float32
                                  )
        self.state_low = np.array(self.state_limit["joint"]["low"]+
                                   self.state_limit["xyz"]["low"]
                            
                                  ,dtype=np.float32
                                  )
      else:
        # TODO: rasie an error 
        print("not supported!") 

      self.observation_space = spaces.Box(self.state_low,self.state_high)
      print("obs_mode:: ",self.obs_mode)
      print("observation_space:: ",self.observation_space)

    def set_configuration(self):
      self.set_obs_mode(self.obs_mode)
      return self.observation_space
     
    # utility functions 
    def get_state_for_perfomance_metric(self):
      _,state_dic = self.finger_joints_and_distnace()

      return state_dic
    
    def finger_joints_and_distnace(self):
        state_dic ={
            # 4 joints 1 dist 1 finger index => dim= 6 
            "joints":self.get_joint_values(),
            "distance":self.get_distance_from_fingertip_to_goal()
         
        }
        state = np.array(state_dic["joints"]+[state_dic["distance"]])
        return state,state_dic
    
    def finger_joint_and_dist_xyz(self,goalId):
      pose = self.TF.get_in_local_finger_frame(goalId,"TH")
      pos  = [abs(pose.x),abs(pose.y),abs(pose.z)]

      state_dic = {
            # 4 joints 3 dist 1 finger index => dim = 8
            "joints":self.get_joint_values(),
            "dist_xyz":pos
          
      }
      # print("finger_joint_and_dist_xyz::state_dic:: ",state_dic)
      state = np.array(state_dic["joints"]+state_dic["dist_xyz"])
      # print("finger_joint_and_dist_xyz::state:: ",state)
      return state
    
    def finger_joint_and_xyz(self,goalId):
      pose = self.TF.get_in_local_finger_frame(goalId,"TH")
      pos  = [pose.x,pose.y,pose.z]

      state_dic = {
            # 4 joints 3 xyz 1 finger index => dim = 8
            "joints":self.get_joint_values(),
            "xyz":pos
      }
      # print("finger_joint_and_xyz::state_dic[joints]::",state_dic["joints"])
      # print("finger_joint_and_xyz::state_dic[xyz]::",state_dic["xyz"])
      state = np.array(state_dic["joints"]+state_dic["xyz"])
      # print("finger_joint_and_xyz::state:: ",state)
      return state
 
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

    def get_joint_values(self):
        return self.controller.get_Observation_thumb()
 
    def get_goal_pos(self):
        goal_state =  p.getBasePositionAndOrientation(self.goalId)
        pos = goal_state[0]
        orn = goal_state[1]
        
        return pos
    
class ThumbAction():
  """
  All the action processing will be done here in order 
  to clean up the code and simplify the controller
  """
  def __init__(self,action_mode,symitric_action,controller_obj):
   
  
    self.action_mode = action_mode
    self.symitric_action = symitric_action
    self.controller_obj = controller_obj
    self._action_modes = ["jointControl"]
    self._action_limit ={
      
      
      "joint":{
          "high":[1.0472, 1.22173, 0.698132, 1.5708],
          "low":[-1.0472, 0, -0.698132, 0]
      }
    }

   
    
  def process_action(self,command):
    """
    will convert action to continious joint values
    The command will be processed differently according to action mode
    """
    processed_command = []
    if self.action_mode =="jointControl":
      processed_command = self.process_jointControl(command)
 

    return processed_command
  
  def set_configuration(self):
    """
    return  different setting for action all together
    """
    self.set_action_mode()
    self.set_To_symitric()

    return self.action_space

  # utility functions 
  def set_action_mode(self):
    action_mode = self.action_mode
    if action_mode == "jointControl": 
      self.action_high_non_symitric = np.array(self._action_limit["joint"]["high"])
      self.action_low_non_symitric  = np.array(self._action_limit["joint"]["low"])

    
    self.action_high = self.action_high_non_symitric
    self.action_low = self.action_low_non_symitric  
    self.action_space = spaces.Box(self.action_low, self.action_high)

  def set_To_symitric(self):

    if self.symitric_action == True:
        self.action_high = np.array(len(self.action_high_non_symitric) *[ 1])
        self.action_low  = np.array(len(self.action_low_non_symitric ) *[-1])
    else:
        self.action_high = self.action_high_non_symitric
        self.action_low = self.action_low_non_symitric

    self.action_space = spaces.Box(self.action_low, self.action_high)

  def process_jointControl(self,command): 
      """
      We only need to check if incomming action is symiteric is yes then we need to convert it to
      Non symitric since robot commands are non symitric
      """
      processed_command = []

      if self.symitric_action ==False:
        processed_command = command
      else:
        processed_command = self.convertTo_non_symitric_action(command)

      return processed_command
  
  def process_IK(self,command): 
  
    processed_command = []

    if self.symitric_action ==False:
      processed_command = command
    else:
      processed_command = self.convertTo_non_symitric_action(command)

    return self.proccess_IK_General(processed_command)

  def process_delta_jointControl(self,delta_command):
    current_state_of_joints = self.get_current_state_of_joints()

    processed_command = current_state_of_joints  
    for index,dc in enumerate(delta_command):
      processed_command[index] +=dc

    if self.joint_commnd_is_within_upper_and_lower_joint_limit(processed_command):
      return processed_command
    else:
      processed_command = self.get_closest_viable_jointcommand(processed_command)
      return processed_command

  def process_delta_IK(self,delta_command): 

    current_state_of_ee = self.get_current_state_of_ee()
  
    processed_command = list(current_state_of_ee)  # xyz
    for index,dc in enumerate(delta_command):
      processed_command[index] +=dc

    # print("process_delta_IK::processed_command:: ",processed_command)

    return self.proccess_IK_General(processed_command)
  
  # utility functions 

  def proccess_IK_General(self,command):
  
    # ee command to joint value command conversion 
    robot_id = self.controller_obj._robot_id
    index_of_ee = self.get_index_of_ee()  

    return self.get_ik_values_for_finger(robot_id,index_of_ee,command)
      
  def get_index_of_ee(self): 
    ee_name = "fingertip_TH"
 
    
    return self.controller_obj.get_endEffectorLinkIndex(ee_name)

  def get_ik_values_for_finger(self,robot_id,index_of_ee,pos):

    joint_command_for_finger = p.calculateInverseKinematics(robot_id,index_of_ee,pos)

    return joint_command_for_finger
    
  def convertTo_non_symitric_action(self,action):

      noneSymitric_action = len(self.action_high_non_symitric)*[0]
      for i in range(0, len(self.action_high)):
        noneSymitric_action[i] = ((action[i]-self.action_low[i])/(self.action_high[i]-self.action_low[i]))*(self.action_high_non_symitric[i]-self.action_low_non_symitric[i])+self.action_low_non_symitric[i]
      return noneSymitric_action

  def ee_command_is_within_ws_limits(self,command):
   
    max,min = self._action_limit["ws"]["th"]
    
    for index,com in enumerate(command):
      if not (com  >= min[index] and com <= max[index]): 
        # print("min:: ",min[index])
        # print("max:: ",max[index])
        # print("com:: ",com)
        return False

    return True

  def get_closet_viable_command(self,command):
    finger_ws_pointcloud = self.ws_pointcoulds["th"]
    x_list,y_list,z_list  = finger_ws_pointcloud

    new_x = self.get_closet_number_in_the_numpy_list(x_list,command[0])
    new_y = self.get_closet_number_in_the_numpy_list(y_list,command[1])
    new_z = self.get_closet_number_in_the_numpy_list(z_list,command[2])

    return new_x,new_y,new_z

  def joint_commnd_is_within_upper_and_lower_joint_limit(self,jointcommand):
    upper_lmit = self._action_limit["joint"]["high"]
    lower_lmit = self._action_limit["joint"]["low"]

    for index,jc in enumerate(jointcommand):
      if not (jc >= lower_lmit[index] and jc <= upper_lmit[index]):
        return False

    return True
        
  def get_closest_viable_jointcommand(self,jointcommand):
    processed_command = [0]*4
    upper_lmit = self._action_limit["joint"]["high"]
    lower_lmit = self._action_limit["joint"]["low"]

    for index,jc in enumerate(jointcommand):
      if jc >= lower_lmit[index] and jc <= upper_lmit[index]:
        processed_command[index] = jc
      elif jc < lower_lmit[index]:
        processed_command[index] = lower_lmit[index]
      elif jc > upper_lmit[index]:
        processed_command[index] = upper_lmit[index]
      else:
        # TODO: raise an error 
        print("this is not a valid joint command")

    return processed_command
  
  def get_closet_number_in_the_numpy_list(self,np_list,num):
    # https://www.geeksforgeeks.org/find-the-nearest-value-and-the-index-of-numpy-array/
    # calculate the difference array
    difference_array = np.absolute(np_list-num)

    # find the index of minimum element from the array
    index = difference_array.argmin()

    new_num = np_list[index]

    return new_num

  def get_current_state_of_ee(self):
    return self.controller_obj.get_observation_finger_tip()
  
  def get_current_state_of_joints(self):
    return self.controller_obj.getObservation()

class ThumbBasicGoalGenerator():
  def __init__(self):
  
    self.point_cloud =self.load_all_goals()
   

  def get_goal(self):
    goal  = random.choice(self.point_cloud)
    return goal

  def load_all_goals(self):

    finger_name = "TH"
    point_cloud = None
        
    #load point cloud for finger
    path = resource_filename(__name__,"/model/"+finger_name+".yml")
    with open(path, "r") as stream:
        try:
           point_cloud = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return point_cloud["vertix"]
    
class ThumbAdaptiveTaskParameter():
  def __init__(self,finger_obj,
               neighbour_radius,
               num_success_required=2,
               use_lower_limit=False,
               sphare_thinkness=0.1
              ):
    self.controller = finger_obj
    self.num_success_required = num_success_required
    self.use_lower_limit = use_lower_limit
    self.neighbour_radius = neighbour_radius
    self.neighbour_radius_lower_limit = self.neighbour_radius -sphare_thinkness
    self.intial_neighbour_radius = self.neighbour_radius

    # state 
    self.fingertips_intial_xyz = self.fingers_initial_state()
    self.starting = True
    self.success_counter =0
  
    


    self.point_cloud = self.load_all_goals()
  
    self.point_cloud_copy = self.point_cloud.copy()


    self.current_goal = self.choose_closest_goal_to_figner()
   
    
    
  def choose_closest_goal_to_figner(self):
    """
    choose closet goal to the finger 
    """
    finger_pos_np = np.array(self.fingertips_intial_xyz)
    dist_mat = self.calculate_dist_matrix(finger_pos_np)
    
    index_min = np.argmin(dist_mat)

    print("choose_closest_goal_to_figner::index_min:: ",index_min)

    goal = self.point_cloud[index_min]

    print("choose_closest_goal_to_figner::goal:: ",goal)

    return goal

  def get_goal(self):
    
    if self.starting:
      self.starting = False
      return self.current_goal

    if self.success_counter>=self.num_success_required:
      self.update_goal_on_success()
    
    return self.current_goal

  def increment_success(self):
    self.success_counter +=1

  def reset_counter_becasue_of_failiur(self):
    self.success_counter =0


  def update_goal_on_success(self):
    self.neighbour_radius = self.intial_neighbour_radius 
    old_goal = self.current_goal[:]
    
    if len(self.point_cloud)>1:
      self.remove_goal()
      if self.use_lower_limit:
        self.current_goal = self.get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit()
      else:
        self.current_goal = self.get_a_goal_in_neighbourhood_of_current_goal()
      
      if self.current_goal ==old_goal:
        print("neighbour_radius is too small")
      self.success_counter = 0
    else:
      self.sample_at_random_if_all_goals_achived()

  # utility function
  def get_a_goal_in_neighbourhood_of_current_goal(self):
    # find list of candidates
    
    indexes  = self.get_neighbourhood_indexs()

    # print("get_a_goal_in_neighbourhood_of_current_goal::indexes:: ",indexes)
    # choose a candidate at random 
    candidate_index = random.choice(indexes)
    candidate       = self.point_cloud[candidate_index] 

    return candidate
  

  def get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit(self):
     # find list of candidates
    
    indexes  = self.get_neighbourhood_indexs_with_band()

    # print("get_a_goal_in_neighbourhood_of_current_goal::indexes:: ",indexes)
    # choose a candidate at random 
    candidate_index = random.choice(indexes)
    candidate       = self.point_cloud[candidate_index] 

    return candidate
  
  def get_neighbourhood_indexs_with_band(self):
    dist_mat = self.calculate_dist_matrix(np.array(self.current_goal))
    ul_indexes  = np.where(dist_mat<self.neighbour_radius )[0].tolist()
    ll_indexes  = np.where(dist_mat>self.neighbour_radius_lower_limit )[0].tolist()
    indexes = self.common_member(ul_indexes,ll_indexes)
    # print("get_neighbourhood_indexs::indexes ",indexes)
    if len(indexes)==0:
      print("did not find a neighbure. Gowing  neighbour_radius inorder to find a new neighboure ")
      self.neighbour_radius *=2
      print("new search radious:: ", self.neighbour_radius)
      return self.get_neighbourhood_indexs_with_band()

    return indexes
  
  def common_member(self,a, b):
    a_set = set(a)
    b_set = set(b)
 
    if (a_set & b_set):
        return list(a_set & b_set)
    else:
        return []
 
  def load_all_goals(self):

    point_cloud = None
        
    finger_name = "TH"
    #load point cloud for finger
    path = resource_filename(__name__,"/model/"+finger_name+".yml")
    with open(path, "r") as stream:
        try:
           point_cloud = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return point_cloud["vertix"]
    
  def calculate_dist_matrix(self,target_xyz):
    
    dist_mat = self.point_cloud - target_xyz
    dist_mat = np.power(dist_mat,2)
    dist_mat = np.sum(dist_mat,axis=1) #sum along row
    dist_mat = np.power(dist_mat,1/2)

    return dist_mat

  def fingers_initial_state(self):
    postions = self.controller.get_observation_finger_tip()
 

    return postions

  def get_neighbourhood_indexs(self):
    dist_mat = self.calculate_dist_matrix(np.array(self.current_goal))
    indexes  = np.where(dist_mat<self.neighbour_radius)[0].tolist()
    # print("get_neighbourhood_indexs::indexes ",indexes)
    if len(indexes)==0:
      print("did not find a neighbure. Gowing  neighbour_radius inorder to find a new neighboure ")
      self.neighbour_radius *=2
      print("new search radious:: ", self.neighbour_radius)
      return self.get_neighbourhood_indexs()

    return indexes

  def get_neighbourhood(self):
    indexes  = self.get_neighbourhood_indexs()
    goals =  np.array(self.point_cloud)
    neighbourhood =goals[indexes] 
    # print("get_neighbourhood::neighbourhood:: ",neighbourhood)

    return neighbourhood

  def remove_goal(self):
    # print("remove_goal::called")
    current_goal = self.current_goal
    goals = np.array(self.point_cloud)
    indexs = np.where(np.all(goals==current_goal,axis=1))[0].tolist()
    # print("remove_goal::index:: ",indexs)
    goals = np.delete(goals ,indexs,axis =0)

    
    self.point_cloud= goals.tolist()
    # print("remove_goal::self.point_cloud ",self.point_cloud)

  def sample_at_random_if_all_goals_achived(self):
    
    goal = random.choice(self.point_cloud_copy)
    return goal

class ThumbRandomStart():
    def __init__(self,finger_obj,adaptive_task_parameter_flag=False,
                 atp_num_success_required=2,
                 use_lower_limit=False,
                 neighbour_radius=0.05,
                 atp_sphare_thinkness=0.1
                 ):
        
        self.finger_list = ["TH"]
        self.last_finger_used = None 
        self._action_limit = {
            #[knuckle,proximal,middle,distal]
            "high":[1.0472, 1.22173, 0.698132, 1.5708],
            "low":[-1.0472, 0, -0.698132, 0]

        }

        self.adaptive_task_parameter_flag = adaptive_task_parameter_flag

        if adaptive_task_parameter_flag:
          self.ATP = ThumbAdaptiveTaskParameter(finger_obj,
                                           neighbour_radius,
                                           atp_num_success_required,
                                           use_lower_limit,
                                           atp_sphare_thinkness
                                          )
        else:
          self.BGG = ThumbBasicGoalGenerator()

    def get_joint_values(self):
        """
        a random joint values is generated for new episode 
        """
        joint_values = []
        for i in range(4):
            joint_values.append(random.uniform(self._action_limit["low"][i],self._action_limit["high"][i]))

        return joint_values

    def get_goal(self):
        """
        get a new goal position from point could in workspace
        """
        goal = None
        if self.adaptive_task_parameter_flag:
          goal = self.ATP.get_goal()
        else:
          goal = self.BGG.get_goal()

        return goal

    def increment_success(self):
      if self.adaptive_task_parameter_flag:
        self.ATP.increment_success()

    def reset_counter_becasue_of_failiur(self):
      if self.adaptive_task_parameter_flag:
        self.ATP.reset_counter_becasue_of_failiur()



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


class HandGymEnv(gym.Env):
    
    def __init__(self,renders=True,timeStep=2000,random_robot_start=False,
                 record_performance=False,obs_mode={"fingers":"finger_joint_and_xyz","thumb":"finger_joint_and_xyz"},
                 action_mode ="delta_jointControl",reward_mode="dense_distance",
                 adaptive_task_parameter_flag=False,atp_neighbour_radius=0.01,
                 atp_num_success_required = 2,
                 atp_use_lower_limit=False,
                 atp_sphare_thinkness=0.005,
                 symitric_action = False,
                 debug=False
                ):
        self._p = p 
        self._render = renders
        self._timeStep = 1/timeStep
        self.debug = debug

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
            "fingers":FingersAction(action_mode = action_mode,symitric_action = symitric_action,controller_obj = self.controller),
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
        
    def reset(self):
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

          goal = self.random_start_obj["fingers"].get_goal(finger_name)
          self.goals["locations"]["current"][finger_name] = goal
        ##### thumb 
        if self.random_robot_start:
          joint_values += self.random_start_obj["thumb"].get_joint_values()
        else:
          joint_values += [0]*4

        goal = self.random_start_obj["thumb"].get_goal()
        self.goals["locations"]["current"]["TH"] = goal
        ##############loading new and previous goal##################
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
   
        return initla_state

    def step(self,action):

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
      state = self.getObservation()
      ##### termination ####
      done = self.termination(goal_is_achived)
      #### reward #####
      reward,_ = self.reward(distance_from_fingertip_to_goal,goal_is_achived)

      return state,reward,done,{}

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
        obs_fingers[finger_name] =self.obs_obj["fingers"].get_state()
      
      goalId = self.goals["ids"]["current"]["TH"]
      self.obs_obj["thumb"].update_goal_and_finger_name(goalId)
      obs_thumb = self.obs_obj["thumb"].get_state()
    
      obs_dic = {
        "fingers":obs_fingers,
        "thumb"  :obs_thumb
      }

      # print("\n\n")
      # print("getObservation::obs_dic[fingers][FF]::type:: ",type(obs_dic["fingers"]["FF"]))
      # print("getObservation::obs_dic[thumb]::type:: ",type(obs_dic["thumb"]))
      # print("\n\n")

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


