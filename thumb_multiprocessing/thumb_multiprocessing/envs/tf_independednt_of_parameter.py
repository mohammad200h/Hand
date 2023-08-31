
from tiny_tf.tf import TFNode, TFTree, Transform
from pprint import pprint
import numpy as np



class CoordinateFrameTrasform():
    
    def __init__(self,physic_engine,finger_obj):
       
        self._p = physic_engine
        self.controller = finger_obj
   
        self.frames = ["world","finger","obj"]

        self.lookuptf_stateMachine = {
            "waiting":0,
            "recived":1
        }

        self.state = {
            "obj_relative_to_finger":{
                "pos":None,
                "orn":None
            },
            "finger":{
                "pos":None,
                "orn":None
            },
            "obj":{
                "pos":None,
                "orn":None
            }
            
        }
        self.tree = TFTree()

    def get_in_local_finger_frame(self,goalId):
        self.get_state(goalId)
        self.setup_tree()
        return self.get_state_in_local_frame()

    # utility function
    def get_state(self,goalId):
        finger_pos,finger_orn = self.controller.get_complete_obs_finger_tip()
        goal_state =  self._p.getBasePositionAndOrientation(goalId)
        obj_pos,obj_orn = goal_state[0],goal_state[1]  

        self.state["finger"]["pos"] =  finger_pos
        self.state["finger"]["orn"] =  finger_orn

        self.state["obj"]["pos"] =  obj_pos
        self.state["obj"]["orn"] =  obj_orn

    def setup_tree(self):
        finger_tf = Transform(x =self.state["finger"]["pos"][0],
                              y =self.state["finger"]["pos"][1],
                              z =self.state["finger"]["pos"][2],
                              qx=self.state["finger"]["orn"][0],
                              qy=self.state["finger"]["orn"][1],
                              qz=self.state["finger"]["orn"][2],
                              qw=self.state["finger"]["orn"][3]
                    )
        obj_tf    = Transform(x =self.state["obj"]["pos"][0],
                              y =self.state["obj"]["pos"][1],
                              z =self.state["obj"]["pos"][2],
                              qx=self.state["obj"]["orn"][0],
                              qy=self.state["obj"]["orn"][1],
                              qz=self.state["obj"]["orn"][2],
                              qw=self.state["obj"]["orn"][3]
                    )
    


        self.tree.add_transform("world","finger",finger_tf)
        self.tree.add_transform("world","obj"   ,obj_tf   )

    def get_state_in_local_frame(self):
        return self.look_up_tf()

    def look_up_tf(self):
        pose = self.tree.lookup_transform("obj","finger")
        # print("CoordinateFrameTrasform::pose:: ",pose.x,pose.y,pose.z)
        return pose
