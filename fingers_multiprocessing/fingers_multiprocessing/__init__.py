from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
register(
    id='fingers_multiprocessing-v0',
    entry_point='fingers_multiprocessing.envs:FingerGymEnv',
    kwargs={'render_mode' : False,
            'obs_mode':"finger_joints_and_distnace",
	        'action_mode' :"jointControl",
            'adaptive_task_parameter_flag':False,
 	        'atp_neighbour_radius':0.1,
            'atp_use_lower_limit':False,
            'atp_sphare_thinkness':0.05,
	    'atp_num_success_required':2,
	    'symitric_action':True,

	},
)
