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



class Fingers:

    def __init__(self,physic_engine):
      #loading the model
      self._p = physic_engine
      robot_path = resource_filename(__name__,"/model/model_fingers.sdf")
      self._robot = self._p.loadSDF(robot_path)
      self._robot_id = self._robot[0]


      self.finger_name = None

      #jointInfo
      self.jointInfo = JointInfo()
      self.jointInfo.get_infoForAll_joints(self._robot)
      self.numJoints = self._p.getNumJoints(self._robot_id)
      self.num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
      self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
      self.active_joints_info = self.jointInfo.getActiveJointsInfo()
      

      self.fingerTip_link_name = ["distal_FF","distal_MF","distal_RF"] #this are joints between final link and one before it
      self.pid = {
                  "hand":{
                    "MF":[
                            {
                             "p":1,
                             "d":1   
                            },
                            {
                             "p":1,
                             "d":1   
                            },
                            {
                             "p":1,
                             "d":1   
                            },
                            {
                             "p":1,
                             "d":1   
                            },
                          ] ,

                    }
      }
      self.fingers_list = ["FF","MF","RF"]
      self.joints_order = {
        "FF":[i for i in range(4)],
        "MF":[i for i in range(4,4*2)],
        "RF":[i for i in range(4*2,4*3)],
   
      }
        
    def reset(self,finger_name,joint_values=None):
      self.finger_name =finger_name
      self.disabled_fingers = self.fingers_list[:]
      self.disabled_fingers.remove(self.finger_name)


      self.reset_active_finger(joint_values)
      self.reset_disabled_fingers()

    def applyAction(self,command):
      self.apply_action_for_disable_fingers()
      self.apply_action_for_active_finger(command)

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
          jointState = p.getJointState(self._robot_id ,jointIndex)
          joints_state[jointName] = jointState[0]
          jointsStates.append(jointState[0])

        if format == "dictinary":
          return joints_state
        else:
          return jointsStates

    def get_Observation_finger(self):

        finger_key = ["J"+str(i)+"_"+self.finger_name for i in range(1,5)]
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

    def get_observation_finger_tip(self):
        finger_tip = "fingertip_"+self.finger_name
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id ,jointIndex)
        pos = link_state[0]
        orn = link_state[1]
        return pos

    def get_complete_obs_finger_tip(self):
      finger_tip = "fingertip_"+self.finger_name
      finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
      Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
      jointIndex = Info["jointIndex"]
      link_state = p.getLinkState(self._robot_id ,jointIndex)
      pos = link_state[0]
      orn = link_state[1]
      return pos,orn


    def get_Observation_fingertips(self):
      pos = {
        "FF":None,
        "MF":None,
        "RF":None
      }
      fingertips = ["fingertip_"+finger_name for finger_name in self.fingers_list]
      for idx,fingertip in enumerate(fingertips):
        finger_tip_encoded = fingertip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self._robot_id ,jointIndex)
        position = link_state[0]
        pos[self.fingers_list[idx]] = position

      return pos

    
    def reset_active_finger(self,joint_values=None):
        if  joint_values:
            for i in range (4):
                adjusted_index = self.get_index_for_finger(self.finger_name,i)

                
                self._p.resetJointState(self._robot_id ,self.indexOf_activeJoints[adjusted_index],joint_values[i])   
        else:
            for i in range (4):
                adjusted_index = self.get_index_for_finger(self.finger_name,i)
                self._p.resetJointState(self._robot_id ,self.indexOf_activeJoints[adjusted_index],0) 

    def reset_disabled_fingers(self):
  
      for finger in self.disabled_fingers:
        for i in range (4):
  
          adjusted_index = self.get_index_for_finger(finger,i)
       
          self._p.resetJointState(self._robot_id ,self.indexOf_activeJoints[adjusted_index],0) 
      
      
    def apply_action_for_active_finger(self,command):
      for i in range(4):
            adjusted_index = self.get_index_for_finger(self.finger_name,i) 
            jointIndex = self.active_joints_info[adjusted_index]["jointIndex"]

            self._p.setJointMotorControl2(self._robot_id ,jointIndex,
                                    self._p.POSITION_CONTROL,command[i], 
                                    targetVelocity=0,force=self.active_joints_info[adjusted_index]["jointMaxForce"], 
                                    maxVelocity=self.active_joints_info[adjusted_index]["jointMaxVelocity"],positionGain=1,
                                    velocityGain=1)

    def apply_action_for_disable_fingers(self):
      
      for finger in self.disabled_fingers:

        for i in range(4):
            adjusted_index = self.get_index_for_finger(finger,i) 
            jointIndex = self.active_joints_info[adjusted_index]["jointIndex"]

            self._p.setJointMotorControl2(self._robot_id ,jointIndex,
                                    p.POSITION_CONTROL,0, 
                                    targetVelocity=0,force=self.active_joints_info[adjusted_index]["jointMaxForce"], 
                                    maxVelocity=self.active_joints_info[adjusted_index]["jointMaxVelocity"],positionGain=1,
                                    velocityGain=1)
    
    def get_index_for_finger(self,finger_name,index):
      index_for_finger = self.joints_order[finger_name][index]

      return index_for_finger

    def get_endEffectorLinkIndex(self,EEName):
      name = EEName.encode(encoding='UTF-8',errors='strict') 
      info = self.jointInfo.searchBy(key="linkName",value = name)[0]
      j_index = info["jointIndex"]
      # print("info::",info)
      # print("j_index::",j_index)

      return j_index
  