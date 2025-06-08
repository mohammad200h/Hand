import gymnasium as gym
from gymnasium.envs.registration import register

register(
    id='thumb_multiprocessing-v0',
    entry_point='thumb_multiprocessing.envs.thumbGymEnv:ThumbGymEnv',  # Replace with actual module and class path
)
import thumb_multiprocessing



def random_agent(episodes=100):
	env = gym.make("thumb_multiprocessing-v0",
								obs_mode="comprehensive"
	)
	env.reset()
	done =False
	while(1):
		
		env.step(env.action_space.sample())
		if done:
			env.reset()

	

if __name__ == "__main__":
    random_agent()
