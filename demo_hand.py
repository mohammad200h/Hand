import gymnasium as gym
import hand_multiprocessing

from gymnasium.envs.registration import register


register(
    id='hand_multiprocessing-v0',
    entry_point='hand_multiprocessing.envs:HandGymEnv',
)

def random_agent(episodes=100):
	env = gym.make("hand_multiprocessing-v0",
	)
	env.reset()
	while(1):
		
		state,reward,done,_ = env.step(env.action_space.sample())
		if done:
			env.reset()

	

if __name__ == "__main__":
    random_agent()
