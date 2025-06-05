from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
register(
    id='thumb_multiprocessing-v0',
    entry_point='thumb_multiprocessing.envs:ThumbGymEnv',
    kwargs={
	    'renders' : False,
	    'obs_mode':"finger_joints_and_distnace",
	    'action_mode' :"jointControl",
            'adaptive_task_parameter_flag':False,
	    'symitric_action': False
	   
        
	},
)
