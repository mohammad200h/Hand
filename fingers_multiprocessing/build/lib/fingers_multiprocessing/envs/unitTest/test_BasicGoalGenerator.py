import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("\n")
print ("test_AW::current_dir=" + currentdir)
print("\n")

import unittest
import gym
import fingers_multiprocessing

import pybullet as p

import os
import sys

import random
import math
import numpy as np



EXPIREMENT_FOLDER = "dummy_exp_folder"
EXPIREMENT_PATH ="./../Expirements/"+EXPIREMENT_FOLDER+"/PPO/"
AW_PROGRESS_PATH = EXPIREMENT_PATH +"AW_progress"
ENV_SETTING_PATH = EXPIREMENT_PATH+"gym_env_setting"



class BasicGoalGenerator_Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(BasicGoalGenerator_Test, self).__init__(*args, **kwargs)
        
        self.env_name = "fingers_multiprocessing-v0"
        self._render = False
        self._p = p
  
    def test_get_a_goal(self):
        
        env =  gym.make("fingers_multiprocessing-v0",
                         obs_mode="finger_joints_and_distnace",
					     renders=False	,
                         adaptive_task_parameter_flag=False 			
	    )
      
        # ************expected output***************
        for finger in ["FF","MF","RF"]:
            goal = env.random_start.BGG.get_goal(finger)
            print("goal:: ",goal)
            self.assertEqual(np.array(goal).shape,(3,))
           
    def test_goals_generated_are_random(self):
        env =  gym.make("fingers_multiprocessing-v0",
                         obs_mode="finger_joints_and_distnace",
					     renders=False	,
                         adaptive_task_parameter_flag=False 			
	    )
      
        # ************expected output***************
        finger = "FF"
        goals =[]
        for i in range(10):
            goal = env.random_start.BGG.get_goal(finger)
            goals.append(goal)
            # print("goal:: ",goal)
        
        for i in range(9):
            self.assertNotEqual(goals[i][0],goals[i+1][0])
            self.assertNotEqual(goals[i][1],goals[i+1][1])
            self.assertNotEqual(goals[i][2],goals[i+1][2])

    



if __name__ =='__main__':
    unittest.main()

"""
To run this  file:
1. be in same doirectory
2. python3 -m unittest test_BasicGoalGenerator
"""