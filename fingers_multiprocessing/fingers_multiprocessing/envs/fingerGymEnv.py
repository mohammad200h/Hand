from operator import truediv
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print ("current_dir=" + currentdir)
os.sys.path.insert(0,currentdir)
import sys
#workaround to make ros work with python3 
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
# import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
from pkg_resources import resource_string,resource_filename
import time
import random
import math


import numpy as np

import pybullet as p
from finger_controller import Fingers

import random
import pybullet_data
from pkg_resources import parse_version
from mamad_util import JointInfo

from collections import OrderedDict

import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
import warnings 

from datetime import datetime
import yaml
import io


import matplotlib.pyplot as plt
import matplotlib


from tf_independednt_of_parameter import CoordinateFrameTrasform

import logging

logging.basicConfig(filename='fingerGymEnv.log',level=os.environ.get("LOGLEVEL", "DEBUG"))
log = logging.getLogger(__name__)


"""
Note:
the way we are going to design the action space for ik control
The action space will be based on point cloud workspace generated for ff.
if the finger is not ff, ff point cloud would be sampled but it will be shifted according to the biase.
so the agent will work with an action sapce designed according to ff, regardlss of finger. However the agent
action will be shifted according to the biase for each finger.
"""


class BasicGoalGenerator():
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
    
class AdaptiveTaskParameter():
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
    log.debug("AdaptiveTaskParameter::increment_success::finger_name:: {} ::success_counter:: {}".format(finger_name,self.success_counter[finger_name]))
    self.success_counter[finger_name] +=1

  def reset_counter_becasue_of_failiur(self,finger_name):
    log.debug("AdaptiveTaskParameter::reset_counter_becasue_of_failiur::finger_name:: {}".format(finger_name))
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

class RandomStart():
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
          self.ATP = AdaptiveTaskParameter(finger_obj,
                                           neighbour_radius,
                                           atp_num_success_required,
                                           use_lower_limit,
                                           atp_sphare_thinkness
                                          )
        else:
          self.BGG = BasicGoalGenerator()

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
      log.debug("RandomStart::increment_success::finger_name:: {}".format(finger_name))
      if self.adaptive_task_parameter_flag:
        self.ATP.increment_success(finger_name)

    def reset_counter_becasue_of_failiur(self,finger_name):
      log.debug("RandomStart::reset_counter_becasue_of_failiur::finger_name:: {}".format(finger_name))
      if self.adaptive_task_parameter_flag:
        self.ATP.reset_counter_becasue_of_failiur(finger_name)

class Workspace_Util():
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

class PerformanceMetricVisulization():
  def __init__(self):
    self.render = True
    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 40}

    # matplotlib.rc('font', **font)


  def plot_performance(self,performance,average_performance,title,path=None):
    print("\n\n")
    print("performance::",performance)
    print("\n\n")

    print("\n\n")
    print("average_performance::",average_performance)
    print("\n\n")

    print("\n\n")
    print("path::",path)
    print("\n\n")


    G_labels = [ 'FF', 'MF', 'RF']
    Best = [performance["ff"],performance["mf"],performance["rf"]]
    Average = [average_performance["ff"],average_performance["mf"],average_performance["rf"]]

    x = np.arange(len(G_labels))  # the label locations
    width = 0.35  # the width of the bars

    plt.style.use('seaborn')

    fig, ax = plt.subplots()
    rects1 = ax.bar(x-width/2 , Best, width,label ="Best" )
    rects2 = ax.bar(x+width/2 , Average, width,label ="Average")
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Distance from target',fontsize=20)
    ax.set_title(title,fontsize=20)

    # ax.legend(labels=['Best', 'Average'],loc='lower right',fontsize=20)
 
    ax.set_ylim(0, 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(G_labels)

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    if path==None:
      plt.savefig('play_performance.png')
    else:
    
      plt.savefig(path)
    if self.render ==True:
      plt.show()

  def plot_performance_during_episode(self,performance,average_performance,path=None):
    self.plot_performance(performance,average_performance,'Performance Durning an episode',path)

  def plot_performance_at_end_episode(self,performance,average_performance,path=None):
    self.plot_performance(performance,average_performance,'Performance at end of episode',path)

  def plot_finger_touch(self,percentag,path=None):
    G_labels = ['FF', 'MF', 'RF']

    percentage=[percentag["ff"],percentag["mf"],percentag["rf"]]

    x = np.arange(len(G_labels))   # the label locations
    width = 0.5  # the width of the bars

    plt.style.use('seaborn')

    fig, ax = plt.subplots()

    # fig.set_size_inches(5,6)
    rects1 = ax.bar(x , percentage, width)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage',fontsize=20)
    ax.set_title("Percentage of fingers touching the target during an episode",fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(G_labels)

    ax.legend()
    ax.set_ylim(0, 1)
    
    

    fig.tight_layout()

    if path==None:
      plt.savefig('play_performance.png')
    else:
      plt.savefig(path)

    if self.render ==True:
      plt.show()

  def plot_table_collision(self,collision,path=None):
 
    labels = ['Table']

    percentage=[collision]

    x = np.arange(len(labels))  # the label locations
    width = 1  # the width of the bars

    plt.style.use('seaborn')

    fig, ax = plt.subplots()
    fig.set_size_inches(2,6)
    rects1 = ax.bar(x, percentage, width)
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percentage',fontsize=20)
    ax.set_title("Collision",fontsize=20)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    
    ax.legend()
    ax.set_ylim(0, 0.3)

    fig.tight_layout()

    if path==None:
      plt.savefig('play_performance.png')
    else:
      plt.savefig(path)


    if self.render ==True:
      plt.show()

class PerformanceMetric():
  def __init__(self,record_performance=False):

    
    self.record_performance =record_performance

    self.visulization = PerformanceMetricVisulization()

    self.perofrmance_log = {
      "episdoes":{# list of episdoe_perofrmance_log
          "ff":[],
          "mf":[],
          "rf":[]
         
      },
      
      "best_performance":{# best episode 
          "ff":None,
          "mf":None,
          "rf":None,
        }, 
      "best_terminal_state_performance":{
          "ff":None,
          "mf":None,
          "rf":None,
        },
      "average_best_performance":{# average performance accross all episdoes
          "ff":None,
          "mf":None,
          "rf":None,
        },
      "average_termination_performance":{
          "ff":None,
          "mf":None,
          "rf":None,
        }
    }

    self.episdoe_perofrmance_log = {
      "best_performance_during_episode":{
          "finger":None,
          "dist":None,
          "step":0
      },
      "performance_at_end_of_episode":{
          "finger":None,
          "dist":None,
          "touch":None
      }
    }

    self.first_run = True
  
  def performance_during_episode(self,obs_obj,finger,step):
    
    if self.record_performance:

      performance_is_better = False
      state = obs_obj.get_state_for_perfomance_metric()

      if self.first_run:
        performance_is_better = True
        self.first_run = False
      else:
        performance_is_better = self.Is_Perofrmance_better_than_last_step(state["distance"])
      
      if performance_is_better:
        print("PerformanceMetric::performance_during_episode::state:: ",state)
        print("PerformanceMetric::performance_during_episode::state[distance]:: ",state["distance"])
        self.episdoe_perofrmance_log["best_performance_during_episode"]["dist"] = finger
        self.episdoe_perofrmance_log["best_performance_during_episode"]["dist"] = state["distance"]     
        self.episdoe_perofrmance_log["best_performance_during_episode"]["step"] = step

  def performance_at_end_of_episode(self,obs_obj,finger,touch):

    if self.record_performance and not self.first_run:

      state = obs_obj.get_state_for_perfomance_metric()
      self.episdoe_perofrmance_log["performance_at_end_of_episode"] = {
            "finger":finger,
            "dist":state["distance"] ,
            "touch":touch
        }

      self.perofrmance_log["episdoes"][finger].append(self.episdoe_perofrmance_log) 
     
      self.episdoe_perofrmance_log = {
          "best_performance_during_episode":{
              "finger":None,
              "dist":None,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":None,
              "dist":None,
              "touch":None
          }
      }
      self.first_run = True
  
  def calculate_performacne_accross_episdoes(self):
    """
    This fucntion calculate peformance using episdoes collected
    average for distances
    percentage for fingers touching
    percentage for collision with table
    """
    average_performance = {
      "best_performance":{
         "dist":{
          
            "ff":None,
            "mf":None,
            "rf":None,
            
          }
      },
       "performance_at_end_of_episode":{
          "dist":{
           
            "ff":None,
            "mf":None,
            "rf":None,
          
          },
          "touch":{ #percentage
            "ff":None,
            "mf":None,
            "rf":None,
           
          }
      }
    }

    average_performance["best_performance"]["dist"] = self.calculate_average_best_performance()
    average_performance["performance_at_end_of_episode"]["dist"] = self.calculate_average_terminal_state_performance()
    average_performance["performance_at_end_of_episode"]["touch"] = self.calculate_percentage_of_fingers_touching()
    
    
    
    self.perofrmance_log["best_performance"]                 =   self.find_best_performance_during_episode_among_all_episodes()
    self.perofrmance_log["best_terminal_state_performance"]  =   self.find_best_performance_at_end_episode_among_all_episodes()
    self.perofrmance_log["average_best_performance"]         =   average_performance["best_performance"]
    self.perofrmance_log["average_termination_performance"]  =   average_performance["performance_at_end_of_episode"]
 
    return self.perofrmance_log


  # utility functions  
  def Is_Perofrmance_better_than_last_step(self,state):
    """
    if the agent perform better across 3 out of 5 then the performance is considered better
    """
    if state <self.episdoe_perofrmance_log["best_performance_during_episode"]["dist"] and state >0:
      return True
    
    return False

  def calculate_ave_performance(self,performance_dic_key):
    ave = {
     "ff":0,
     "mf":0,
     "rf":0,
    }

    #sum
    for key in ave.keys():
      for e in self.perofrmance_log["episdoes"][key]:
    
       ave[key] += e[performance_dic_key]["dist"]

    # average
    for key in ave.keys():
      length = len(self.perofrmance_log["episdoes"][key])
      if length>0:
        ave[key] /=length
      else:
        ave[key]=0


    return ave
  
  def calculate_average_best_performance(self):

    performance_dic_key = "best_performance_during_episode"
    ave = self.calculate_ave_performance(performance_dic_key)

    return ave

  def calculate_average_terminal_state_performance(self):
    performance_dic_key = "performance_at_end_of_episode"
    ave = self.calculate_ave_performance(performance_dic_key)

    return ave

  def calculate_percentage_of_fingers_touching(self):
    percentage = {
      "ff":0,
      "mf":0,
      "rf":0,
  
    }

    #frequency
    for key in percentage.keys():
      for e in self.perofrmance_log["episdoes"][key]:
  
       if e["performance_at_end_of_episode"]["touch"]:
         percentage[key] +=1 
  
    # percentage
    for key in percentage.keys():
      length = len(self.perofrmance_log["episdoes"][key]) 
      if length>0:
        percentage[key] /= length
      else:
        percentage[key] = 0
      
      percentage[key] *=100

    return percentage


  def find_best_performance_during_episode_among_all_episodes(self):
    pereformance = {
          "dist":{
            
            "ff"  :float('inf'),
            "mf"  :float('inf'),
            "rf"  :float('inf'),
          }
    }

    
    for key in pereformance["dist"].keys():
      for e in self.perofrmance_log["episdoes"][key]:
      
        
        if pereformance["dist"][key] > e["best_performance_during_episode"]["dist"] and e["best_performance_during_episode"]["dist"]>0:
           pereformance["dist"][key] = e["best_performance_during_episode"]["dist"]
      
      
      
 

    return pereformance

  def find_best_performance_at_end_episode_among_all_episodes(self):
    
    pereformance =  {
      "dist":{
            
            "ff"  :float('inf'),
            "mf"  :float('inf'),
            "rf"  :float('inf')
          }, 
      "touch":{
            
            "ff"  :False,
            "mf"  :False,
            "rf"  :False
          }
         
    }
  
    for key in pereformance["dist"].keys():
      for e in self.perofrmance_log["episdoes"][key]:
      
          if (pereformance["dist"][key] > e["performance_at_end_of_episode"]["dist"]):
              pereformance["dist"][key] = e["performance_at_end_of_episode"]["dist"]
              pereformance["touch"][key] = e["performance_at_end_of_episode"]["touch"]


    return pereformance
  
class Observation():
    def __init__(self,physic_engine,finger_obj,workspace_util,obs_mode ="finger_joints_and_distnace"):

        self._p = physic_engine
        self.controller = finger_obj
        self.finger_name =None
        self.goalId = None

        self.workspace_util =workspace_util
        self.ws_max,self.ws_min = self.workspace_util.get_max_min_xyz()
        self.TF = CoordinateFrameTrasform(self._p,self.controller)


        self.obs_modes = ["finger_joints_and_distnace","finger_joint_and_dist_xyz",
                          "finger_joint_and_xyz",
                          "comprehensive"
                          ]
        self.obs_mode =obs_mode
        self.observation_space = None
        self.dist_dic_max = self.get_dist_max()

        self.goal_max , self.goal_min = self.get_max_min_goal_xyz()

        self.state_limit={
           "joint":{
            #[knuckle,proximal,middle,distal]
            "high":[0.349066,1.5708,1.5708,1.5708],
            "low":[-0.349066,0     ,0     ,0     ]
           },
           "distance":{
               "high":[self.dist_dic_max["dist"] ], 
               "low":[0]

           },
           "dist_xyz":{
            "high":[self.dist_dic_max["dist_xyz"]["x"],self.dist_dic_max["dist_xyz"]["y"],self.dist_dic_max["dist_xyz"]["z"]],
            "low":[0]*3

           },
           "xyz":{
            "high":self.ws_max,
            "low":self.ws_min
           },
           "finger_index":{
              # 0,1,2
               "high":[2],
               "low":[0]
           },
           "goal":{
              # THis is max x,y,z of the goal
              "high":self.goal_max,
              "low":self.goal_min
           },
           "fignertip":{
              # THis is max x,y,z of the goal
              "high":self.goal_max,
              "low":self.goal_min
           },
           "history":{
              # last_act + last_last_act : [knuckle,proximal,middle,distal]*2
              "high":[0.349066,1.5708,1.5708,1.5708]*2,
              "low":[-0.349066,0     ,0     ,0     ]*2
           }
        }

    def update_goal_and_finger_name(self,name,goalId):
      self.finger_name =name
      self.goalId = goalId

    def get_state(self,history:None):
      state =None
      if self.obs_mode == "finger_joints_and_distnace":
        state,_ = self.finger_joints_and_distnace()
      elif self.obs_mode =="finger_joint_and_dist_xyz":
        state = self.finger_joint_and_dist_xyz(self.goalId)
      elif self.obs_mode == "finger_joint_and_xyz":
        state = self.finger_joint_and_xyz(self.goalId)
      elif self.obs_mode == "comprehensive":
        state = self.finger_comprehensive(history)

      return state

    def set_obs_mode(self,mode):
      if self.obs_mode == "finger_joints_and_distnace":
        self.state_high = np.array(self.state_limit["joint"]["high"]+
                                   self.state_limit["distance"]["high"]+
                                   self.state_limit["finger_index"]["high"]
                                   ,dtype=np.float32
                                  )
        self.state_low = np.array(self.state_limit["joint"]["low"]+
                                   self.state_limit["distance"]["low"]+
                                   self.state_limit["finger_index"]["low"]
                                  ,dtype=np.float32
                                  ) 
      elif self.obs_mode == "finger_joint_and_dist_xyz":
        self.state_high = np.array(self.state_limit["joint"]["high"]+
                                   self.state_limit["dist_xyz"]["high"]+
                                   self.state_limit["finger_index"]["high"]
                                  ,dtype=np.float32
                                  )
        self.state_low = np.array(self.state_limit["joint"]["low"]+
                                   self.state_limit["dist_xyz"]["low"]+
                                   self.state_limit["finger_index"]["low"]
                                  ,dtype=np.float32
                                  )
      elif self.obs_mode == "finger_joint_and_xyz":
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
      elif self.obs_mode == "comprehensive":
        self.state_high = np.array(self.state_limit["joint"]["high"]+
                                   self.state_limit["distance"]["high"]+
                                   self.state_limit["finger_index"]["high"]+
                                   self.state_limit["goal"]["high"]+
                                   self.state_limit["fignertip"]["high"]+
                                   self.state_limit["history"]["high"]
                                   ,dtype=np.float32
                                  )
        self.state_low = np.array(self.state_limit["joint"]["low"]+
                                   self.state_limit["distance"]["low"]+
                                   self.state_limit["finger_index"]["low"]+
                                   self.state_limit["goal"]["low"]+
                                   self.state_limit["fignertip"]["low"]+
                                   self.state_limit["history"]["low"]
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
      pose = self.TF.get_in_local_finger_frame(goalId)
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
      pose = self.TF.get_in_local_finger_frame(goalId)
      pos  = [pose.x,pose.y,pose.z]

      state_dic = {
            # 4 joints 3 xyz 1 finger index => dim = 8
            "joints":self.get_joint_values(),
            "xyz":pos,
            "finger_index":self.get_finger_index(self.finger_name)
      }
      
      state = np.array(state_dic["joints"]+state_dic["xyz"]+[state_dic["finger_index"]],dtype=np.float32)
      # print("finger_joint_and_xyz::state:: ",state)
      return state

    def finger_comprehensive(self,history:dict):
      _,chunk = self.finger_joints_and_distnace()
      fingertip_pos = self.get_finger_tip_pos_in_world_frame()
      goal = self.get_goal_pos()

      state = np.array(chunk["joints"]+
                      history["last_act"]+
                      history["last_last_act"]+
                      list(fingertip_pos)+
                      list(goal)+
                      [chunk["distance"]]+
                      [chunk["finger_index"]],
                      dtype=np.float32
              )

      return state

    def get_finger_tip_pos_in_world_frame(self):
        finger_tip_pos =  self.controller.get_observation_finger_tip()

        return finger_tip_pos

    def get_distance_from_fingertip_to_goal(self):
        
        goal_pos = self.get_goal_pos()
        finger_tip_pos =  self.controller.get_observation_finger_tip()

        x_dist = goal_pos[0] - finger_tip_pos[0]
        y_dist = goal_pos[1] - finger_tip_pos[1]
        z_dist = goal_pos[2] - finger_tip_pos[2]
        dist =  math.sqrt(x_dist**2+y_dist**2+z_dist**2)

        return dist

    def get_finger_index(self,finger_name):
        finger_list=["FF","MF","RF"]

        return finger_list.index(finger_name)

    def get_goal_pos(self):
        goal_state =  p.getBasePositionAndOrientation(self.goalId)
        pos = goal_state[0]
        orn = goal_state[1]
        
        return pos

    def get_dist_max(self):
      dist_x = abs(self.ws_max[0]-self.ws_min[0])
      dist_y = abs(self.ws_max[1]-self.ws_min[1])
      dist_z = abs(self.ws_max[2]-self.ws_min[2])

      dist = max([dist_x,dist_y,dist_z])

      dist_dic_max = {
          "dist":dist,
          "dist_xyz":{
            "x":dist_x,
            "y":dist_y,
            "z":dist_z
          }
      }
      return dist_dic_max

    def get_max_min_goal_xyz(self):
      ff_ws_max, ff_ws_min = self.workspace_util.get_max_min_xyz_for_finger("ff")
      mf_ws_max, mf_ws_min = self.workspace_util.get_max_min_xyz_for_finger("mf")


      print(f"get_max_min_goal_xyz::ff_ws_min::{ff_ws_min}")
      print(f"get_max_min_goal_xyz::mf_ws_max::{mf_ws_max}")

      min_ws = np.minimum(ff_ws_min, mf_ws_min)
      max_ws = np.maximum(mf_ws_max, ff_ws_max)

      print(f"get_max_min_goal_xyz::min_ws::type{type(min_ws)}")
      print(f"get_max_min_goal_xyz::max_ws::type{type(max_ws)}")


      return max_ws.tolist(), min_ws.tolist()


class Action():
  """
  All the action processing will be done here in order 
  to clean up the code and simplify the controller
  """
  def __init__(self,action_mode,symitric_action,controller_obj,workspace_util):
   
    self.workspace_util = workspace_util
    self.ws_max,self.ws_min = self.workspace_util.get_max_min_xyz()

    print("self.ws_max,self.ws_min:: ",self.ws_max,self.ws_min)

    self.finger_offsets = {
      "ff":[0,0,0],
      "mf":[0.022,0.002536,0.003068],
      "rf":[0.044,0,0]
    }

    self.action_mode = action_mode
    self.symitric_action = symitric_action
    self.controller_obj = controller_obj
    self._action_modes = ["jointControl","IK","delta_jointControl","delta_IK"]
    self._action_limit ={
      
      "ee":{
        "high":self.ws_max,
        "low" :self.ws_min
      },

      "delta_ee":{
        "high":[ 0.01, 0.01, 0.01],
        "low" :[-0.01,-0.01,-0.01]
      },

      "delta_joint":{
        "high":[ 0.01, 0.01, 0.01 ,0.01],
        "low" :[-0.01,-0.01,-0.01 ,-0.01]
      },

      "joint":{
        "high":[0.349066,1.5708,1.5708,1.5708],
        "low" :[-0.349066,0     ,0     ,0    ]
      },
      "ws":{
        "ff":self.workspace_util.get_max_min_xyz_for_finger("ff"),
        "mf":self.workspace_util.get_max_min_xyz_for_finger("mf"),
        "rf":self.workspace_util.get_max_min_xyz_for_finger("rf")
      }
    }

    self.ws_pointcoulds = self.workspace_util.get_ws_pointcould()
    
  def process_action(self,command,finger_name):
    """
    will convert action to continious joint values
    The command will be processed differently according to action mode
    """
    processed_command = []
    if self.action_mode =="jointControl":
      processed_command = self.process_jointControl(command)
    elif  self.action_mode =="IK":
      processed_command = self.process_IK(command,finger_name)
    elif self.action_mode =="delta_IK":
      processed_command = self.process_delta_IK(command,finger_name)
    elif self.action_mode =="delta_jointControl":
      processed_command = self.process_delta_jointControl(command)

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

    elif action_mode == "IK": 
      self.action_high_non_symitric = np.array(self._action_limit["ee"]["high"],dtype=np.float32)
      self.action_low_non_symitric  = np.array(self._action_limit["ee"]["low"],dtype=np.float32)
    
    elif action_mode == "delta_IK": 
      self.action_high_non_symitric = np.array(self._action_limit["delta_ee"]["high"],dtype=np.float32)
      self.action_low_non_symitric  = np.array(self._action_limit["delta_ee"]["low"],dtype=np.float32)

    elif action_mode == "delta_jointControl": 
      self.action_high_non_symitric = np.array(self._action_limit["delta_joint"]["high"],dtype=np.float32)
      self.action_low_non_symitric  = np.array(self._action_limit["delta_joint"]["low"],dtype=np.float32)

    
  
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
        processed_command = command
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

class FingerGymEnv(gymnasium.Env):

    def __init__(self,renders=True,
                 render_mode = None,
                 timeStep=1000,
                 max_episode_step = 200,
                 goal_threshold = 0.01,
                 learning_algorithem_uses_her = False,
                 random_robot_start=False,
                 record_performance=False,obs_mode="finger_joints_and_distnace",
                 action_mode ="delta_jointControl",reward_mode="dense_distance",
                 adaptive_task_parameter_flag=True,
                 atp_neighbour_radius=0.01,
                 atp_num_success_required = 2,
                 atp_use_lower_limit=False,
                 atp_sphare_thinkness=0.005,
                 symitric_action = True,
                 debug=False ):

        self._p = p 
        self._render = renders
        self._timeStep = 1/timeStep
        self.debug = debug

        self.finger_mapping_for_performance_metric = {
          "FF":"ff",
          "MF":"mf",
          "RF":"rf"
        }

        self.action_mode = action_mode

        self.random_robot_start = random_robot_start

        #objects that need to be set
        self.controller = None
        self.random_start = None

        self.finger_name = None #current active finger
        self.goalId = None
        self.previous_goalId = None
        self.goal = {
           "FF":None,
           "MF":None,
           "RF":None
        }
        self.previous_goal  = {
           "FF":None,
           "MF":None,
           "RF":None
        }

        self.wsId = None

        #connecting to a physic server
        if self._render:
          cid = self._p.connect(self._p.SHARED_MEMORY)
          if (cid<0):
             id = self._p.connect(self._p.GUI)
          self._p.resetDebugVisualizerCamera(1.,50,-41,[0.1,-0.2,-0.1])
        else:
          self._p.connect(self._p.DIRECT)

        #loading the secne
        self.load_scene()
        #setting up random start
        print("adaptive_task_parameter_flag:: ",adaptive_task_parameter_flag)
        self.adaptive_task_parameter_flag = adaptive_task_parameter_flag
        self.random_start = RandomStart(finger_obj=self.controller,
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
        self.max_episode_step = max_episode_step # an episode will terminate (end) if this number is reached

        self.goal_threshold = goal_threshold #the goal has been achievd if the distance between fingertip and goal is less than this

        self.control_delay = 5 # this term contorls how often agent gets to interact with the enviornment
        ###########Workspace_Util###########
        self.workspace_util = Workspace_Util()
        ########### HER ################
        self.learning_algorithem_uses_her = learning_algorithem_uses_her
        ###########setting up state space###########

        self.obs_obj = Observation(self._p,self.controller,self.workspace_util,obs_mode=obs_mode)
        self.observation_space = self.obs_obj.set_configuration()

        if self.learning_algorithem_uses_her:
          self.observation_space = spaces.Dict({
              "observation": self.obs_obj.set_configuration(),
              "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
              "desired_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
          })

        print("self.observation_space.shape:: ",self.observation_space.shape)
        # sys.exit()
        ###########setting up action space###########

        print("Intializing action")
        self.action_obj = Action(action_mode = self.action_mode,symitric_action = symitric_action,controller_obj = self.controller,workspace_util=self.workspace_util)
        self.action_space = self.action_obj.set_configuration()

        self.perfromanceMeteric = PerformanceMetric(record_performance)

        ###########setting up Reward###########
        self.reward_obj = Reward(reward_mode) 

        ########### History ################
        self._hsitory = {
          "last_act":[0]*4,
          "last_last_act":[0]*4
        }

    def reset(self,seed=None, options=None):
        if seed is not None:
          # If you use any random numbers, seed them here, e.g.
          import random
          import numpy as np
          random.seed(seed)
          np.random.seed(seed)

        #resetting number of steps in an episode
        self.current_step = 0

        ###########getting random parameters for this episode###########
        self.finger_name = self.random_start.get_finger_name()
        joint_values=None
        if self.random_robot_start:
          joint_values = self.random_start.get_joint_values()
        else:
          joint_values = [0]*4
        goal = self.random_start.get_goal(self.finger_name)
        # print("reset::finger_name:::: ",self.finger_name)
        # print("reset::goal:::: ",goal)
        ##############setting new goal##################
        self.update_ws_for_finger()
        self.change_goal_location(goal)
        if self.goal[self.finger_name] != goal:
          self.previous_goal[self.finger_name] = self.goal[self.finger_name]
          if self.previous_goal[self.finger_name] ==None:
             self.previous_goal[self.finger_name] = goal


          self.goal[self.finger_name] = goal

        self.change_goal_location(self.previous_goal[self.finger_name],True)

        ############resetting the robot at the begining of each episode#######
        self.controller.reset(self.finger_name,joint_values)
        self._p.stepSimulation()

        ############getting robot state##############

        self.obs_obj.update_goal_and_finger_name(self.finger_name,self.goalId)
        initla_state = self.getObservation(self._hsitory)
        # print("initla_state::shape:: ",initla_state.shape)

        log.debug("FingerGymEnv::reset::fingername::{}".format(self.finger_name))

        if self.learning_algorithem_uses_her:
          obs = {
            "observation":initla_state,
            "desired_goal":self.goal[self.finger_name],
            "achieved_goal": self.obs_obj.get_finger_tip_pos_in_world_frame()
          }
        else:
          obs = initla_state

        return obs,{}
    def step(self,action):
        ############# Steping #################
        finger_name = self.finger_mapping_for_performance_metric[self.finger_name]
        command = self.action_obj.process_action(action,finger_name)
        for i in range(self.control_delay):
            self.controller.applyAction(command)
            self._p.stepSimulation()

        self.current_step +=1
        ###########recording performance#################
        finger_name = self.finger_mapping_for_performance_metric[self.finger_name]
        self.perfromanceMeteric.performance_during_episode(self.obs_obj,finger_name,self.current_step)
        #################################################


        distance_from_fingertip_to_goal = self.obs_obj.get_distance_from_fingertip_to_goal()
        goal_is_achived = self.is_goal_achived(distance_from_fingertip_to_goal)


        state = self.getObservation(self._hsitory)
        reward = self.reward(distance_from_fingertip_to_goal,goal_is_achived)
        done = self.termination(goal_is_achived)

        ############## history ###########
        self._hsitory["last_last_act"] = self._hsitory["last_act"]
        self._hsitory["last_act"] =self.obs_obj.get_joint_values()
        ###########Debug TF Using ROS RVIZ#################
        if self.debug:
          print("\n\n")
          pose_tiny_tf = self.obs_obj.TF.get_in_local_finger_frame(self.goalId)
          self.debugTF.publish_tf_chain(self.goalId,pose_tiny_tf)
          print("\n\n")
        ####################################
        truncated = self.current_step > self.max_episode_step and not done
        info = {"action":action}

        ####################################
        if self.learning_algorithem_uses_her:
          obs = {
            "observation":state,
            "desired_goal":self.goal[self.finger_name],
            "achieved_goal": self.obs_obj.get_finger_tip_pos_in_world_frame()
          }
        else:
          obs = state
        
        return obs, reward, done, truncated, info

    def render(self):
      pass

    def compute_reward(self, achieved_goal, desired_goal, info):
      return -np.linalg.norm(achieved_goal - desired_goal)

    def getObservation(self,history=None):
       return self.obs_obj.get_state(history)

    def reward(self,distance_from_fingertip_to_goal,goal_is_achived):

       reward = self.reward_obj.get_reward(distance_from_fingertip_to_goal,goal_is_achived)
    
       return reward

    def termination(self,goal_is_achived):

        ###########recording performance#################
        finger_name = self.finger_mapping_for_performance_metric[self.finger_name]
        self.perfromanceMeteric.performance_at_end_of_episode(self.obs_obj,finger_name,goal_is_achived)

        if self.current_step > self.max_episode_step or goal_is_achived : #episdoe will end
          ###############Adaptive task parameter###########
          if goal_is_achived:
            self.random_start.increment_success(self.finger_name)
          else:
            self.random_start.reset_counter_becasue_of_failiur(self.finger_name)
          #################################################
          # print("termination::goal_is_achived:: ",goal_is_achived)
          # print("termination::self.current_step > self.max_episode_step:: ",self.current_step > self.max_episode_step)
          log.debug("FingerGymEnv::reset::fingername:: {} ::goal_is_achived::{}".format(self.finger_name,goal_is_achived))
          return True

        return False 
    
    ########utility function#########
    def load_scene(self):
        #load robot
        self.controller = Fingers(self._p)
        #load floor 
        urdfRoot=pybullet_data.getDataPath()
        self.plane_id = self._p.loadURDF(os.path.join(urdfRoot,"plane.urdf"),[0,0,0])
        #loading goal
        goal_path = resource_filename(__name__,"/goal/goal.sdf")
        self.goalId = self._p.loadSDF(goal_path)[0]
        previous_goal_path = resource_filename(__name__,"/goal/previous_goal .sdf")
        self.previous_goalId = self._p.loadSDF(previous_goal_path)[0]
        # loading ws
        ws_path = resource_filename(__name__,"/finger_ws/WS_FF.sdf")
        self.wsId = self._p.loadSDF(ws_path)[0]

    def change_goal_location(self,goal,previous=False):
      euler_angle = [0,0,0]
      quaternion_angle = self._p.getQuaternionFromEuler(euler_angle)
      if previous:
        self._p.resetBasePositionAndOrientation(self.previous_goalId, goal,quaternion_angle)
      else:
        self._p.resetBasePositionAndOrientation(self.goalId, goal,quaternion_angle)  

    def update_ws_for_finger(self):
        FF_offset = [0,0,0]
        MF_offset = [0.022,0.002536,0.003068]
        RF_offset = [0.044,0,0]

        if self.finger_name =="FF":
            self._p.resetBasePositionAndOrientation(self.wsId, FF_offset,[0,0,0,1])
        elif self.finger_name =="MF":
            self._p.resetBasePositionAndOrientation(self.wsId, MF_offset,[0,0,0,1])
        elif self.finger_name =="RF":
            self._p.resetBasePositionAndOrientation(self.wsId, RF_offset,[0,0,0,1])

    def is_goal_achived(self,distance_from_fingertip_to_goal):

      dist_to_goal = distance_from_fingertip_to_goal
      # print("is_goal_achived::dist_to_goal:: ",dist_to_goal)
      if dist_to_goal < self.goal_threshold :
        return True
      
      return False




#  from gym.envs.registration import register

# register(
#     id='MyEnv-v0',
#     entry_point='my_env:MyEnv',
# )