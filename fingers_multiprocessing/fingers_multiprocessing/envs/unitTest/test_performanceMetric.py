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

import math



EXPIREMENT_FOLDER = "dummy_exp_folder"
EXPIREMENT_PATH ="./../Expirements/"+EXPIREMENT_FOLDER+"/PPO/"
AW_PROGRESS_PATH = EXPIREMENT_PATH +"AW_progress"
ENV_SETTING_PATH = EXPIREMENT_PATH+"gym_env_setting"



class performmanceMetric_Test(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(performmanceMetric_Test, self).__init__(*args, **kwargs)
        
        self.env_name = "kuka_handlit_multiprocess-v0"
        self._render = False
        self._p = p
  
    def test_calculate_ave_performance(self):
      # *************input*************
      # ff 
      episdoe_perofrmance_log_one_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":2,
              "touch":False
             
          }
      }

      # mf 
      episdoe_perofrmance_log_one_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":2,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":3,
              "touch":False
             
          }
      }
     

      env =  gym.make("fingers_multiprocessing-v0",
					renders=False
						
	    )
      # ff 
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_one_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_two_ff)
      # mf 
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_one_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_two_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_three_mf)
      
      # ************eexpected output***************
      ëxpected_ave = {
   
        "ff":1.5,
        "mf":2,
        "rf":0,
       
      }

      out_ave_best_performance_during_episode = env.perfromanceMeteric.calculate_ave_performance("best_performance_during_episode")
      out_ave_performance_at_end_of_episode = env.perfromanceMeteric.calculate_ave_performance("performance_at_end_of_episode")
      
      for key in ëxpected_ave.keys():
        self.assertEqual(ëxpected_ave[key],out_ave_best_performance_during_episode[key])
      for key in ëxpected_ave.keys():
        self.assertEqual(ëxpected_ave[key],out_ave_performance_at_end_of_episode[key])

    def test_calculate_percentage_of_fingers_touching(self):
      # *************input*************
      
      # ff 
      episdoe_perofrmance_log_one_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":2,
              "touch":True
             
          }
      }

      # mf 
      episdoe_perofrmance_log_one_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":1,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":2,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_three_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":3,
              "touch":False
             
          }
      }
     
      env =  gym.make("fingers_multiprocessing-v0",
					renders=False
				
	    )

      # ff 
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_one_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_two_ff)
      # mf 
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_one_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_two_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_three_mf)
      
      # ************eexpected output***************
      ëxpected_percentage = {
  
        "ff":50.0,
        "mf":66.66666666666666,
        "rf":0,
       
      }

      finger_touch_percentage = env.perfromanceMeteric.calculate_percentage_of_fingers_touching()

      for key in ëxpected_percentage.keys():
        self.assertEqual(ëxpected_percentage[key],finger_touch_percentage[key]) 
     
    def test_Is_Perofrmance_better_than_last_step(self):
        
        env =  gym.make("fingers_multiprocessing-v0",
			  		renders=False
			  				
	      )

        # ff 
        episdoe_perofrmance_log_good_ff = {
            "best_performance_during_episode":{
                "finger":"ff",
                "dist":1,
                "step":0
            },
            "performance_at_end_of_episode":{
                "finger":"ff",
                "dist":1,
                "touch":False
               
            }
        }
        episdoe_perofrmance_log_bad_ff = {
            "best_performance_during_episode":{
                "finger":"ff",
                "dist":2,
                "step":1
            },
            "performance_at_end_of_episode":{
                "finger":"ff",
                "dist":2,
                "touch":False
               
            }
        }
  
        # ************False***************
        env.perfromanceMeteric.episdoe_perofrmance_log["best_performance_during_episode"]["dist"] = episdoe_perofrmance_log_good_ff["best_performance_during_episode"]["dist"]
    
        
       
        state = episdoe_perofrmance_log_bad_ff["best_performance_during_episode"]["dist"]
     

        expected_flag = False
        out_flag = env.perfromanceMeteric.Is_Perofrmance_better_than_last_step(state) 

        self.assertEqual(out_flag,expected_flag) 

        # ************True***************
        env.perfromanceMeteric.episdoe_perofrmance_log["best_performance_during_episode"]["dist"] = episdoe_perofrmance_log_bad_ff["best_performance_during_episode"]["dist"]
       
        
        state = episdoe_perofrmance_log_good_ff["best_performance_during_episode"]["dist"]
 

        expected_flag = True
        out_flag = env.perfromanceMeteric.Is_Perofrmance_better_than_last_step(state) 

        self.assertEqual(out_flag,expected_flag) 

    def test_find_best_performance_during_episode_among_all_episodes(self):
      
      env =  gym.make("fingers_multiprocessing-v0",
					renders=False
						
	    )
      
      # *************input*************
      # ff 
      episdoe_perofrmance_log_one_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":4,
              "touch":False
             
          }
      }

      # mf 
      episdoe_perofrmance_log_one_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":2,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":4,
              "touch":False
             
          }
      }
      # rf 
      episdoe_perofrmance_log_one_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":4,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":4,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_two_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":2,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":1,
              "touch":False
             
          }
      }
     
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_one_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_two_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_three_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_four_ff)


      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_one_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_two_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_three_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_four_mf)

      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_one_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_two_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_three_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_four_rf)
     # ************eexpected output***************
      expected ={
        "ff":episdoe_perofrmance_log_two_ff["best_performance_during_episode"]["dist"],
        "mf":episdoe_perofrmance_log_one_mf["best_performance_during_episode"]["dist"],
        "rf":episdoe_perofrmance_log_four_rf["best_performance_during_episode"]["dist"],
      }
      
      out = env.perfromanceMeteric.find_best_performance_during_episode_among_all_episodes()["dist"]

      self.assertEqual(expected,out)
 
    def test_find_best_performance_at_end_episode_touch_flag(self):
      
      env =  gym.make("fingers_multiprocessing-v0",
					renders=False
							
	    )
      
      # *************input*************


       # ff 
      episdoe_perofrmance_log_one_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":4,
              "touch":False
             
          }
      }

      # mf 
      episdoe_perofrmance_log_one_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":1,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":2,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_three_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":4,
              "touch":False
             
          }
      }
      # rf 
      episdoe_perofrmance_log_one_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":4,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":4,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":2,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":3,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_four_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":1,
              "touch":True
             
          }
      }
     
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_one_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_two_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_three_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_four_ff)


      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_one_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_two_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_three_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_four_mf)

      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_one_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_two_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_three_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_four_rf)
      # ************eexpected output***************
      """
      Fingers touching is more important than reducing the distance
      """
      expected ={
      "ff":25.0,
      "mf":50.0,
      "rf":75.0,
  
    }
      
      out = env.perfromanceMeteric.calculate_percentage_of_fingers_touching()

      self.assertEqual(expected,out)

    def test_find_best_performance_at_end_episode_among_all_episodes(self):
      
      env =  gym.make("fingers_multiprocessing-v0",
					renders=False
						
	    )
      
      # *************input*************
       # ff 
      episdoe_perofrmance_log_one_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":1,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_ff = {
          "best_performance_during_episode":{
              "finger":"ff",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"ff",
              "dist":4,
              "touch":False
             
          }
      }

      # mf 
      episdoe_perofrmance_log_one_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":1,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":1,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":2,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_three_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":3,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_four_mf = {
          "best_performance_during_episode":{
              "finger":"mf",
              "dist":4,
              "step":4
          },
          "performance_at_end_of_episode":{
              "finger":"mf",
              "dist":4,
              "touch":False
             
          }
      }
      # rf 
      episdoe_perofrmance_log_one_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":4,
              "step":0
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":4,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_two_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":2,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":2,
              "touch":False
             
          }
      }
      episdoe_perofrmance_log_three_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":3,
              "step":3
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":3,
              "touch":True
             
          }
      }
      episdoe_perofrmance_log_four_rf = {
          "best_performance_during_episode":{
              "finger":"rf",
              "dist":1,
              "step":1
          },
          "performance_at_end_of_episode":{
              "finger":"rf",
              "dist":1,
              "touch":True
             
          }
      }
     
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_one_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_two_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_three_ff)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["ff"].append(episdoe_perofrmance_log_four_ff)


      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_one_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_two_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_three_mf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["mf"].append(episdoe_perofrmance_log_four_mf)

      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_one_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_two_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_three_rf)
      env.perfromanceMeteric.perofrmance_log["episdoes"]["rf"].append(episdoe_perofrmance_log_four_rf)
      # ************eexpected output***************
      """
      checking distance
      """
      expected ={
      "dist":{
            
            "ff"  :episdoe_perofrmance_log_two_ff ["performance_at_end_of_episode"]["dist"],
            "mf"  :episdoe_perofrmance_log_one_mf ["performance_at_end_of_episode"]["dist"],
            "rf"  :episdoe_perofrmance_log_four_rf["performance_at_end_of_episode"]["dist"]
          }, 
      "touch":{
            
            "ff"  :episdoe_perofrmance_log_two_ff ["performance_at_end_of_episode"]["touch"],
            "mf"  :episdoe_perofrmance_log_one_mf ["performance_at_end_of_episode"]["touch"],
            "rf"  :episdoe_perofrmance_log_four_rf["performance_at_end_of_episode"]["touch"]
          }
         
    }
      
      out = env.perfromanceMeteric.find_best_performance_at_end_episode_among_all_episodes()

      self.assertEqual(expected,out)

if __name__ =='__main__':
    unittest.main()

"""
To run this  file:
1. be in same doirectory
2. python3 -m unittest test_performanceMetric
"""