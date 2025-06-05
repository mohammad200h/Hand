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



class AdaptiveTaskParameter_Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(AdaptiveTaskParameter_Test, self).__init__(*args, **kwargs)
        
        self.env_name = "fingers_multiprocessing-v0"
        self._render = False
        self._p = p
  
    def test_get_a_goal_in_neighbourhood_of_current_goal(self):
        
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
        num_trails = 1000
        for i in range(num_trails):
            # choose a goal at random change current goal manually
            random_goal_FF = random.choice(env.random_start.ATP.point_cloud["FF"])
            # print("random_goal_FF:: ",random_goal_FF)
            env.random_start.ATP.current_goal["FF"] = random_goal_FF
            # get neighbours of this point and make sure their distance is less than self.neighbour_radius
            neighbours = env.random_start.ATP.get_a_goal_in_neighbourhood_of_current_goal("FF")
            # print("neighbours:: ",neighbours)
            # print("neighbours::shape:: ",np.array(neighbours).shape)
            dist = np.array(neighbours) - np.array(random_goal_FF)
            dist = np.power(dist,2)
            dist = np.sum(dist)
            dist = np.power(dist,1/2)
            # print("dist:: ",dist)
            # ************expected output***************

            # Test one number of neighbours should be equal to one 
            self.assertEqual(np.array(neighbours).shape,(3,))
            # Distance of neighbours to random_goal_FF should be within thereshold
            self.assertTrue(dist<env.random_start.ATP.neighbour_radius)

    def tset_get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit(self):
        
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True,
                         atp_neighbour_radius=0.1,
                         atp_use_lower_limit= True,
                        atp_sphare_thinkness= 0.05

	    )
        num_trails = 1000
        for i in range(num_trails):
            # choose a goal at random change current goal manually
            random_goal_FF = random.choice(env.random_start.ATP.point_cloud["FF"])
            # print("random_goal_FF:: ",random_goal_FF)
            env.random_start.ATP.current_goal["FF"] = random_goal_FF
            # get neighbours of this point and make sure their distance is less than self.neighbour_radius
            neighbours = env.random_start.ATP.get_a_goal_in_neighbourhood_of_current_goal_wthin_a_upper_and_lower_limit("FF")
            # print("neighbours:: ",neighbours)
            # print("neighbours::shape:: ",np.array(neighbours).shape)
            dist = np.array(neighbours) - np.array(random_goal_FF)
            dist = np.power(dist,2)
            dist = np.sum(dist)
            dist = np.power(dist,1/2)
            # print("dist:: ",dist)
            # ************expected output***************

            # Test one number of neighbours should be equal to one 
            self.assertEqual(np.array(neighbours).shape,(3,))
            # Distance of neighbours to random_goal_FF should be within thereshold
            self.assertTrue(dist<env.random_start.ATP.neighbour_radius)
            self.assertTrue(dist>env.random_start.ATP.self.neighbour_radius_lower_limit)
    
    
    
    def test_choose_closest_goal_to_figner(self):
     
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )

        fingertips_xyz = env.controller.get_Observation_fingertips()
        initial_fingertips_goal = {
            "FF":env.random_start.ATP.get_goal("FF"),
            "MF":env.random_start.ATP.get_goal("MF"),
            "RF":env.random_start.ATP.get_goal("RF")
        }
        print("fingertips_xyz::FF:: ",fingertips_xyz["FF"])
        print("initial_fingertips_goal::FF:: ",initial_fingertips_goal["FF"])

        print("fingertips_xyz::MF:: ",fingertips_xyz["MF"])
        print("initial_fingertips_goal::MF:: ",initial_fingertips_goal["MF"])

        print("fingertips_xyz::RF:: ",fingertips_xyz["RF"])
        print("initial_fingertips_goal::RF:: ",initial_fingertips_goal["RF"])
        # changing neighbour radious to get more candidates 
        env.random_start.ATP.neighbour_radius = 0.1
        # ************expected output***************
        # the distance between goal generated and finger tip should be the smallest
        # we do this by getting neigubers of suggested goal, compute the ditance to fignertip
        # and make sure suggessted point is the closet poit
        for finger in ["FF","MF","RF"]:
            neighbourhood = env.random_start.ATP.get_neighbourhood(finger)

            # print("neighbourhood::len ",len(neighbourhood))

            dist = np.array(neighbourhood) - np.array(initial_fingertips_goal[finger])
            dist = np.power(dist,2)
            dist = np.sum(dist,axis=1)
            dist = np.power(dist,1/2)
            min_dist = np.min(dist)
            min_dist_index = np.argmin(dist)

            # print("dist:: ",dist)
            # print("min_dist:: ",min_dist)
            # print("min_dist_index:: ",min_dist_index)
            # print("dist::shape ",dist.shape)

            current_goal = env.random_start.ATP.current_goal[finger]
            # print("current_goal:: ",current_goal)
            # print("closest_goal:: ",neighbourhood[min_dist_index])

            threshold = np.abs(np.array(current_goal)-np.array(neighbourhood[min_dist_index]))
            threshold = np.sum(threshold)/3
            # print("threshold:: ",threshold)

            self.assertTrue(threshold<1e-6)
   
    def test_get_goal(self):
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
        # ************expected output***************
        # testing self.starting flag 

         # first time runing 
        env.random_start.ATP.get_goal("FF")
        self.assertEqual( env.random_start.ATP.starting , False)

        # making sure the goal has the right dimentions
        goal = env.random_start.ATP.get_goal("FF")
        self.assertEqual(np.array(goal).shape , (3,))
     
    def test_remove_goal(self):
        """
        1.Here we should make sure that goal is removed when remove_goal function is called
        2.After all the goals are visited size of self.pointcloud[finger] is zero
        """
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
       
        # ************expected output***************
        # Here we should make sure that goal is removed when remove_goal function is called
        # ******************************************
        current_goal = np.array(env.random_start.ATP.current_goal["FF"])

        # the goal should not be removed if success rate is not achived
        for i in range(env.random_start.ATP.num_success_required-1):
            env.random_start.increment_success("FF")
            env.random_start.get_goal("FF")
        goals  = np.array(env.random_start.ATP.point_cloud["FF"])
        indexs = np.where(np.all(goals==current_goal,axis=1))[0].tolist()
        self.assertEqual(len(indexs),1)
        
        # reseting success counte
        env.random_start.ATP.success_counter["FF"]=0
        # simulating when reset function is called (get goal is called once in reset this will not increase success)
        env.random_start.get_goal("FF")
        # the goal should be removemed when success rate is met
        current_success_counter = env.random_start.ATP.success_counter["FF"]
        for i in range(env.random_start.ATP.num_success_required):
            env.random_start.increment_success("FF")
    
            current_success_counter+=1
            print("test_remove_goal::current_success_counter:: ",current_success_counter)
            print("test_remove_goal::env.random_start.ATP.success_counter[FF]::",env.random_start.ATP.success_counter["FF"])
            self.assertEqual(current_success_counter, env.random_start.ATP.success_counter["FF"])
            
            env.random_start.get_goal("FF")
            
        
        goals = np.array(env.random_start.ATP.point_cloud["FF"])
        indexs = np.where(np.all(goals==current_goal,axis=1))[0].tolist()
        

        print("\n\n")
        print("test_remove_goal::indexs:: ",indexs)
        print("\n\n")
        self.assertEqual(len(indexs),0)

    


        # ************expected output***************
        # After all the goals are visited size of self.pointcloud[finger] is zero
        # ******************************************
        num_goals = len(env.random_start.ATP.point_cloud["FF"])
       
        while(num_goals>1):

            for i in range(env.random_start.ATP.num_success_required):
                env.random_start.increment_success("FF")
                env.random_start.get_goal("FF")            
            num_goals = len(env.random_start.ATP.point_cloud["FF"])
            print("test_remove_goal::num_goals:: ",num_goals)

        self.assertEqual(len(env.random_start.ATP.point_cloud["FF"]),1)


    """
    def test_update_goal_on_success(self):
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
        # ************expected output***************
        # here we check if the self.success_counter works
        for finger in ["FF","MF","RF"]:
            # checking if the incrementing works appropriatly
            # first time runing 
            env.random_start.ATP.get_goal(finger)
            self.assertEqual(env.random_start.ATP.success_counter[finger], 0)

            # normal operation Fail
            env.random_start.ATP.get_goal(finger)
            self.assertEqual(env.random_start.ATP.success_counter[finger], 0)
            # normal operation Success
            env.random_start.increment_success(finger)
            env.random_start.ATP.get_goal(finger)
            self.assertEqual(env.random_start.ATP.success_counter[finger], 1)

            # meeting required success rate 
            success_rate = env.random_start.ATP.num_success_required
            adjusted_sucess_rate = success_rate-1
            for i in range(1,adjusted_sucess_rate+1):
                env.random_start.increment_success(finger)
                env.random_start.ATP.get_goal(finger)
                

            self.assertEqual(env.random_start.ATP.success_counter[finger], 0)
    """
    def test_sample_at_random_if_all_goals_achived(self):
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
        
        num_goals = len(env.random_start.ATP.point_cloud["FF"])
       
        while(num_goals>1):

            for i in range(env.random_start.ATP.num_success_required):
                env.random_start.increment_success("FF")
                env.random_start.get_goal("FF") 

            num_goals = len(env.random_start.ATP.point_cloud["FF"])

        # ************expected output***************
        # we are still getting goals after all goals have been visited
        # ******************************************
        goal = None
        for i in range(env.random_start.ATP.num_success_required):
            env.random_start.increment_success("FF")
            goal = env.random_start.get_goal("FF") 
        
        self.assertTrue(goal) 

    def test_repeated_success(self):
        env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True 			
	    )
        # ************expected output***************
        # If we do not reach the reuqied success rate in a continious 
        # sequence this should cause the success coutner to reset 
        # otherwise it should reach success rate threshold
        # ******************************************
        for i in range(3):
            env.random_start.increment_success("FF")
        self.assertEqual(env.random_start.ATP.success_counter["FF"],3)
        env.random_start.reset_counter_becasue_of_failiur("FF")
        self.assertEqual(env.random_start.ATP.success_counter["FF"],0)
        
        env.random_start.get_goal("FF")
        for i in range(env.random_start.ATP.num_success_required):
            env.random_start.increment_success("FF")

        self.assertEqual(env.random_start.ATP.success_counter["FF"],env.random_start.ATP.num_success_required)
        env.random_start.get_goal("FF")
        self.assertEqual(env.random_start.ATP.success_counter["FF"],0)

        



         



if __name__ =='__main__':
    unittest.main()

"""
To run this  file:
1. be in same doirectory
2. python3 -m unittest test_AdaptiveTaskParameter

Run A single test:
python3  test_AdaptiveTaskParameter.py AdaptiveTaskParameter_Test.test_sample_at_random_if_all_goals_achived

"""