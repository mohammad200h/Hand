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



class Thumb:

    def __init__(self,physic_engine):
      #loading the model
      self._p = physic_engine
      robot_path = resource_filename(__name__,"/model/model_TH.sdf")
      self._robot = self._p.loadSDF(robot_path)
      self._robot_id = self._robot[0]
      

      #jointInfo
      self.jointInfo = JointInfo()
      self.jointInfo.get_infoForAll_joints(self._robot)
      self.numJoints = self._p.getNumJoints(self._robot_id)
      self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
      self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
      self.active_joints_info = self.jointInfo.getActiveJointsInfo()

    def reset(self,joint_values=None):
      #reseting the robot
      if  joint_values:
          for i in range (4):
              p.resetJointState(self._robot_id,self.indexOf_activeJoints[i],joint_values[i])   
      else:
          for i in range (4):
              p.resetJointState(self._robot_id,self.indexOf_activeJoints[i],0) 

    def applyAction(self,command):
        # print("\n")
        # print("applyAction::command:: ",command)
        # print("\n")
        #getting information about active joints
        num_active_joints  = self.jointInfo.getNumberOfActiveJoints()
        active_joints_info = self.jointInfo.getActiveJointsInfo()

        # print("\n")
        # print("applyAction::num_active_joints:: ",num_active_joints)
        # print("\n")
        #applting command to joints
        for i in range(4):
            jointIndex = active_joints_info[i]["jointIndex"]

            p.setJointMotorControl2(self._robot_id,jointIndex,
                                    p.POSITION_CONTROL,command[i], 
                                    targetVelocity=0,force=active_joints_info[i]["jointMaxForce"], 
                                    maxVelocity=active_joints_info[i]["jointMaxVelocity"],positionGain=1,
                                    velocityGain=1)

    def getObservation(self):
        return self.get_Observation_finger()

    #utility functions
    def getObservation_joint(self,format="list"):
    
        indexOfActiveJoints = self.jointInfo.getIndexOfActiveJoints()
        jointsInfo = self.jointInfo.getActiveJointsInfo()

        jointsStates = []
        joints_state = {} #key:joint ,value = joint state 

        for i in range(len(indexOfActiveJoints)):
          jointName  = jointsInfo[i]["jointName"]
          jointIndex = indexOfActiveJoints[i]
          jointState = p.getJointState(self._robot_id,jointIndex)
          joints_state[jointName] = jointState[0]
          jointsStates.append(jointState[0])

        if format == "dictinary":
          return joints_state
        else:
          return jointsStates

    def get_Observation_finger(self):

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
    def get_complete_obs_finger_tip(self):
        finger_tip = "fingertip_"+"TH"
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos,orn
    
    def get_observation_finger_tip(self):
        finger_tip = "fingertip_"+"TH"
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos

    def get_endEffectorLinkIndex(self,EEName):
      name = EEName.encode(encoding='UTF-8',errors='strict') 
      info = self.jointInfo.searchBy(key="linkName",value = name)[0]
      j_index = info["jointIndex"]
      # print("info::",info)
      # print("j_index::",j_index)

      return j_index