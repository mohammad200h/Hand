
import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)
from pkg_resources import resource_string,resource_filename

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import time
import sys

from mamad_util import JointInfo



try:
  from model import ModelInfo
  from model import DomainRandomization
except:
  from model.modelInfo_util import ModelInfo
  from model.modelGenerator import DomainRandomization
from collections import OrderedDict


class Hand:

    def __init__(self,physic_engine,robot_pose):
        #loading the model
        self._p = physic_engine
        robot_path = resource_filename(__name__,"/model/model_full.sdf")
        self._robot = self._p.loadSDF(robot_path)
        self._robot_id = self._robot[0]
        self._p.resetBasePositionAndOrientation(self._robot_id,robot_pose["position"],robot_pose["orientation"])


        #jointInfo
        self.jointInfo = JointInfo()
        self.jointInfo.get_infoForAll_joints(self._robot)
        self.numJoints = self._p.getNumJoints(self._robot_id)
        self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
        self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
        self.active_joints_info = self.jointInfo.getActiveJointsInfo()

    def reset(self,joint_values=None):
        # print("Hand::reset::joint_values:: ",joint_values)
        # print("Hand::reset::self.num_Active_joint:: ",self.num_Active_joint)
        if  joint_values:
            for i in range (self.num_Active_joint):
                p.resetJointState(self._robot_id,self.indexOf_activeJoints[i],joint_values[i])   
        else:
            for i in range (self.num_Active_joint):
                p.resetJointState(self._robot_id,self.indexOf_activeJoints[i],0) 

    def applyActionToFingers(self,command):
        # print("\n\n")
        # print("applyActionToFingers::command::len ",len(command))
        # print("applyActionToFingers::command:: ",command)
        # print("\n\n")
        for i in range(0,12):
           # print("applyActionToFingers::self.active_joints_info[i+7]",self.active_joints_info[i+7])
           jointIndex =  self.active_joints_info[i]["jointIndex"]
           jointMaxForce =  self.active_joints_info[i]["jointMaxForce"]
           jointMaxVelocity =  self.active_joints_info[i]["jointMaxVelocity"]

           self._p.setJointMotorControl2(self._robot_id ,jointIndex,
                                     self._p.POSITION_CONTROL,command[i], 
                                     targetVelocity=0,force=jointMaxForce, 
                                     maxVelocity=jointMaxVelocity,positionGain=1,
                                     velocityGain=1)
          
    def applyActionToThumb(self,command):
        for i in range(4):
          jointIndex =  self.active_joints_info[i+12]["jointIndex"]
          jointMaxForce =  self.active_joints_info[i+12]["jointMaxForce"]
          jointMaxVelocity =  self.active_joints_info[i+12]["jointMaxVelocity"]

          self._p.setJointMotorControl2(self._robot_id,jointIndex,
                                      p.POSITION_CONTROL,command[i], 
                                      targetVelocity=0,force=jointMaxForce, 
                                      maxVelocity=jointMaxVelocity,positionGain=1,
                                      velocityGain=1)
    
          
    def getObservation_joint(self,format="list"):
    
        indexOfActiveJoints = self.jointInfo.getIndexOfActiveJoints()
        jointsInfo = self.jointInfo.getActiveJointsInfo()

        jointsStates = []
        joints_state = {} #key:joint ,value = joint state 

        for i in range(len(indexOfActiveJoints)):
          jointName  = jointsInfo[i]["jointName"]
          jointIndex = indexOfActiveJoints[i]
          jointState = p.getJointState(self._robot_id ,jointIndex)
          joints_state[jointName] = jointState[0]
          jointsStates.append(jointState[0])

        if format == "dictinary":
          return joints_state
        else:
          return jointsStates
    
    def get_observation_finger_tip(self,finger_name):
        finger_tip = "fingertip_"+finger_name
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id ,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos
      
    def get_complete_obs_finger_tip(self,finger_name):
        finger_tip = "fingertip_"+finger_name
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id ,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos,orn
    
    def get_complete_obs_thumb_tip(self):
        finger_name ="TH"
        finger_tip = "fingertip_"+finger_name
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id ,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos,orn
    
    
      
    def get_Observation_finger(self,finger_name):
        # print("get_Observation_finger::finger_name::",finger_name)
        finger_key = ["J"+str(i)+"_"+finger_name for i in range(1,5)]
        joint_values =[]
        joint_info = self.getObservation_joint(format="dictinary")
        # print("get_Observation_finger::joint_info::",joint_info)
        cleaned_dic ={}
        for key,value in joint_info.items():
          cleaned_dic[key.decode()]=value

        # print("get_Observation_finger::cleaned_dic::",cleaned_dic)

        # print("\n")
        # print("cleaned_dic::keys",cleaned_dic.keys())
        # print("\n")
        for key in finger_key:
          joint_values.append(cleaned_dic[key])


        return joint_values
     
    def get_Observation_thumb(self):

        finger_key = ["THJ"+str(i) for i in range(1,6)if i!=3]
        # print("finger_key:: ",finger_key)
        # sys.exit()
        joint_values =[]
        joint_info = self.getObservation_joint(format="dictinary")
        cleaned_dic ={}
        for key,value in joint_info.items():
          cleaned_dic[key.decode()]=value


        # print("\n")
        # print("cleaned_dic::keys",cleaned_dic.keys())
        # print("\n")
        for key in finger_key:
          joint_values.append(cleaned_dic[key])  

        return joint_values
     

    