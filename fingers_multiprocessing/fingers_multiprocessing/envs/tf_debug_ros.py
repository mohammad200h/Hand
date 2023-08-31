import rospy
import sys
import tf
import tf2_ros


import geometry_msgs.msg

import time


# https://www.youtube.com/watch?v=gnTlLzqFslU&t=2299s
# https://answers.ros.org/question/323075/transform-the-coordinate-frame-of-a-pose-from-one-fixed-frame-to-another/
class DebugTransformation():
    def __init__(self,physic_engine,finger_obj):
        self._p = physic_engine
        self.controller = finger_obj
   
        self.frames = ["world","finger","obj"]

        self.lookuptf_stateMachine = {
            "waiting":0,
            "recived":1
        }
        self.state = {
            "finger":{
                "pos":None,
                "orn":None
            },
            "obj":{
                "pos":None,
                "orn":None
            },
            "obj_relative_to_finger":{
                "pos":None,
                "orn":None
            },
        }

        rospy.init_node('DebugTransformationNode')
        """
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        """
        self.br = tf.TransformBroadcaster()

        self.listener = tf.TransformListener()

        self.turtle_name = rospy.get_param('obj','finger')
        print("turtle_name:: ",self.turtle_name)

     
     
        # sys.exit()

    def publish_tf(self,goalId):
        self.get_state(goalId)
        self.publish_obj()
        self.publish_finger()

    def publish_tf_chain(self,goalId):
        self.get_state(goalId)
        self.publish_obj()
        self.publish_finger()


        state = self.look_up_tf()
        if self.lookuptf_stateMachine[state]:
            self.publish_general(child_frame_name="finger",parent_child="world")
            self.publish_general(child_frame_name="obj_relative_to_finger",parent_child="finger")


    # utility functions 
    def get_state(self,goalId):
        finger_pos,finger_orn = self.controller.get_complete_obs_finger_tip()
        goal_state =  self._p.getBasePositionAndOrientation(goalId)
        obj_pos,obj_orn = goal_state[0],goal_state[1]   
        self.state = {
            "finger":{
                "pos":finger_pos,
                "orn":finger_orn
            },
            "obj":{
                "pos":obj_pos,
                "orn":obj_orn
            },
            "obj_relative_to_finger":
            {
                "pos":obj_pos,
                "orn":obj_orn
            }
        }
  
    def publish_finger(self):
        self.publish_general("finger")

    def publish_obj(self):
        self.publish_general("obj")

    def publish_general(self,child_frame_name,parent_child="world"):
        pose_msg = geometry_msgs.msg.Pose()

        pose_msg.position.x = self.state[child_frame_name]["pos"][0]
        pose_msg.position.y = self.state[child_frame_name]["pos"][1]
        pose_msg.position.z = self.state[child_frame_name]["pos"][2]
        
        pose_msg.orientation.x = self.state[child_frame_name]["orn"][0]
        pose_msg.orientation.y = self.state[child_frame_name]["orn"][1]
        pose_msg.orientation.z = self.state[child_frame_name]["orn"][2]
        pose_msg.orientation.w = self.state[child_frame_name]["orn"][3]

        self.br.sendTransform((pose_msg.position.x,pose_msg.position.y,pose_msg.position.z),
                            (pose_msg.orientation.x ,pose_msg.orientation.y ,pose_msg.orientation.z ,pose_msg.orientation.w ),
                            rospy.Time.now(),
                            "/"+child_frame_name,
                            "/"+parent_child
        )


    def look_up_tf(self):
        try:
            trans,rot = self.listener.lookupTransform("/finger", "/obj", rospy.Time(0))
            self.state["obj_relative_to_finger"]["pose"] = trans
            self.state["obj_relative_to_finger"]["orn"] = rot
            return "recived"
        except:
            return "waiting"