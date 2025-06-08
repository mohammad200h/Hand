import gymnasium as gym
from gymnasium.envs.registration import register

import hand_multiprocessing

from gymnasium.envs.registration import register

import numpy as np

register(
    id='hand_multiprocessing-v0',
    entry_point='hand_multiprocessing.envs:HandGymEnv',
)

register(
    id='hand_orch_multiprocessing-v0',
    entry_point='hand_orch_multiprocessing.envs.HandGymEnvOrchestrator:HandGymEnvOrchestrator',  # Replace with actual module and class path
)

import hand_orch_multiprocessing


def thumb_action(obs):
	# TODO: develop this
	return [0]*4

def finger_action(obs):
	# TODO: develop this
	return [0]*4

def random_agent(episodes=100):
	hand_env_config = {
		"obs_mode": {
      "fingers":"comprehensive",
      "thumb":"comprehensive"
    },
		"renders":False
  }
	env = gym.make("hand_orch_multiprocessing-v0",
								thumb_agent = thumb_action,
								fingers_agent = finger_action,
								num_envs = 2,
								hand_env_config=hand_env_config,
								log_dir ="."
								)
	done = False
	obs,info = env.reset()
	print(f"rest::obs::shape::{obs.shape}")
	print(f"rest::obs::{obs}")

	while(1):
		action = env.action_space.sample()
		print(f"action::type::{type(action)}")
		print(f"action::{action}")
		vec_action = np.vstack([action, action.copy()])
		print(f"vec_action::shape::{vec_action.shape}")
		print(f"vec_action::{vec_action}")

		env.step(vec_action)
		if done:
			env.reset()

	

if __name__ == "__main__":
    random_agent()