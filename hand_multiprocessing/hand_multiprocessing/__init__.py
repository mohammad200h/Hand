from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
register(
    id='hand_multiprocessing-v0',
    entry_point='hand_multiprocessing.envs:HandGymEnv',
    kwargs={
	    'renders' : True,
	    'obs_mode':{"fingers":"finger_joint_and_xyz","thumb":"finger_joint_and_xyz"},
	    'action_mode' :"jointControl",
        'adaptive_task_parameter_flag':False,
	    'symitric_action': False
	      
	},
)

