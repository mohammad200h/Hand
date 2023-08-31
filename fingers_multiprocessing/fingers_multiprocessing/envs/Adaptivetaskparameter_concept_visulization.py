import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import gym
import fingers_multiprocessing
"""
Here i use the env to viulize how adaptive task parameter concept work.
I will manually force success to see how the goal moves after success.
"""

def main():

    env =  gym.make("fingers_multiprocessing-v0",
					     renders=False	,
                         adaptive_task_parameter_flag=True)

    env.reset()

    while(1):
        pass


