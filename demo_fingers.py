import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='fingers_multiprocessing-v0',
    entry_point='fingers_multiprocessing.envs.fingerGymEnv:FingerGymEnv',  # Replace with actual module and class path
)

import fingers_multiprocessing



def random_agent(episodes=100):
	env = gym.make("fingers_multiprocessing-v0",
	obs_mode="comprehensive"
	)
	done = False
	env.reset()
	while(1):
		
		env.step(env.action_space.sample())
		if done:
			env.reset()

if __name__ == "__main__":
    random_agent()
