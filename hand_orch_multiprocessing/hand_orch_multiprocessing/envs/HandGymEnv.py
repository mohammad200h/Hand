import gymnasium
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import  DummyVecEnv, SubprocVecEnv,VecNormalize
from stable_baselines3.common.env_util import make_vec_env


class FingerGymEnv(gymnasium.Env):
  def __init__(self, thumb_agent,fingers_agent,
               num_envs,env_kwargs,
               log_dir,
               success_threshold = 0.01):
    self._thumb_agent = thumb_agent
    self._fingers_agent = fingers_agent

    self._success_threshold = success_threshold

    env_id = "hand_multiprocessing-v0"
    env = make_vec_env(env_id, n_envs=num_envs,env_kwargs=env_kwargs,
                       monitor_dir=log_dir,vec_env_cls=SubprocVecEnv
    )

    self._env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # This is the final goal the agent should reach
    self.synergy_goal

  def reset(self):
    self._env.reset()



  def step(self,action):
    # every step the orchestrator produces and intermidiate goal so
    # that the hand reaches the final goal through intermidiate goals
    ff_gaol = action[:4]
    mf_goal = action[4:8]
    rf_goal = action[8:12]
    th_goal = action[12:]

    # set_intermidiate_goal
    obs = self._env.reset(
      {
        "FF":ff_gaol,
        "MF":mf_goal,
        "RF":rf_goal,
        "TH":th_goal
      }
    )

    # wait 200 steps for the agent to reach the goal
    done = False
    while not done:
      actions = []
      for finger in ["FF","MF","RF"]:
        finger_obs =None
        finger_action = self._fingers_agent(finger_obs)
        actions += finger_action

      thumb_obs = None
      th_action = self._thumb_agent(thumb_obs)
      actions += th_action
      obs = self.env_step(actions)

    # calculate reward by working out distance between current intermidiate goal and final goal


  def get_reward(self):
    pass

  def get_termination(self):
    pass




