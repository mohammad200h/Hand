#!/usr/bin/env python

import pybullet as p




finger = "FF"



if __name__ == '__main__':
	p.connect(p.GUI)
	robot = p.loadSDF("./WS_"+finger+".sdf")
	# ws    = p.loadSDF("./meshes/ws/FF/model.sdf") 
	while(1):
		
		# shadowHand = robot[0]
		p.setRealTimeSimulation(1)