#!/usr/bin/env python

import pybullet as p


from modelGenerator import DomainRandomization

finger = "full"

dr = DomainRandomization(load_ws=False,load_ws_pcd = False)
dr.save_setting()
dr.generate_model_sdf(control_mode=finger)

if __name__ == '__main__':
	p.connect(p.GUI)
	robot = p.loadSDF("./model_"+finger+".sdf")
	# ws    = p.loadSDF("./meshes/ws/FF/model.sdf") 
	while(1):
		
		# shadowHand = robot[0]
		p.setRealTimeSimulation(1)