#!/usr/bin/env python

import pybullet as p


from modelGenerator import DomainRandomization
dr = DomainRandomization(load_ws=True,load_ws_pcd = False)
dr.save_setting()
dr.generate_model_sdf(control_mode="TH")

if __name__ == '__main__':
	p.connect(p.GUI)
	# objects = p.loadSDF("./model_full.sdf")
	objects = p.loadSDF("./model_TH.sdf")
	# ws = p.loadSDF("./meshes/ws/TH/model.sdf")

	while(1):
		

		p.setRealTimeSimulation(1)