import rospy
import sys
import tf
import tf2_ros

import time

import geometry_msgs.msg

import tf2_geometry_msgs 
from geometry_msgs.msg import Pose


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

        rospy.init_node('DebugTransformationNode')
        self.br = tf2_ros.TransformBroadcaster()
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.turtle_name = rospy.get_param('obj','finger')
        # print("turtle_name:: ",self.turtle_name)

     
     
        # sys.exit()

    def publish_tf(self,goalId):
        self.get_state(goalId)
        self.publish_obj()
        self.publish_finger()

    def publish_tf_chain(self,goalId,tiny_tf_pose):
        self.get_state(goalId)
        self.publish_obj()
        self.publish_finger()


        state = self.look_up_tf()
        # print("state:: ",state)
        if self.lookuptf_stateMachine[state]:
            self.publish_general(child_frame_name="finger",parent_child="world")
            self.publish_general(child_frame_name="obj_relative_to_finger",parent_child="finger")

        self.publish_tiny_tf(tiny_tf_pose,child_frame_name="obj_relative_to_finger_tinytf",parent_child="finger")
        # self.transform_pose()

    # utility functions 
    def get_state(self,goalId):
        finger_pos,finger_orn = self.controller.get_complete_obs_finger_tip()
        goal_state =  self._p.getBasePositionAndOrientation(goalId)
        obj_pos,obj_orn = goal_state[0],goal_state[1]  

        self.state["finger"]["pos"] =  finger_pos
        self.state["finger"]["orn"] =  finger_orn

        self.state["obj"]["pos"] =  obj_pos
        self.state["obj"]["orn"] =  obj_orn

       
    def publish_finger(self):
        self.publish_general("finger")

    def publish_obj(self):
        self.publish_general("obj")

    def publish_general(self,child_frame_name,parent_child="world"):
  
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "/"+parent_child
        t.child_frame_id = "/"+child_frame_name
        t.transform.translation.x = self.state[child_frame_name]["pos"][0]
        t.transform.translation.y = self.state[child_frame_name]["pos"][1]
        t.transform.translation.z = self.state[child_frame_name]["pos"][2]
        
        t.transform.rotation.x = self.state[child_frame_name]["orn"][0]
        t.transform.rotation.y = self.state[child_frame_name]["orn"][1]
        t.transform.rotation.z = self.state[child_frame_name]["orn"][2]
        t.transform.rotation.w = self.state[child_frame_name]["orn"][3]
        
   
        self.br.sendTransform(t)

    def publish_tiny_tf(self,pose,child_frame_name,parent_child="world"):
        t = geometry_msgs.msg.TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "/"+parent_child
        t.child_frame_id = "/"+child_frame_name
        t.transform.translation.x = pose.x
        t.transform.translation.y = pose.y
        t.transform.translation.z = pose.z
        
        t.transform.rotation.x = pose.qx
        t.transform.rotation.y = pose.qy
        t.transform.rotation.z = pose.qz
        t.transform.rotation.w = pose.qw
        
   
        self.br.sendTransform(t)
    
    def transform_pose(self):
        child_frame_name = "obj"
        obj_pose = Pose()
        obj_pose.position.x = self.state[child_frame_name]["pos"][0]
        obj_pose.position.y = self.state[child_frame_name]["pos"][1]
        obj_pose.position.z = self.state[child_frame_name]["pos"][2]
        obj_pose.orientation.x = self.state[child_frame_name]["orn"][0]
        obj_pose.orientation.y = self.state[child_frame_name]["orn"][1]
        obj_pose.orientation.z = self.state[child_frame_name]["orn"][2]
        obj_pose.orientation.w = self.state[child_frame_name]["orn"][3]

        pose_stamped = tf2_geometry_msgs.PoseStamped()
        pose_stamped.pose = obj_pose
        pose_stamped.header.frame_id = "obj"
        pose_stamped.header.stamp = rospy.Time.now()
        try:
            output_pose_stamped = self.tf_buffer.transform(pose_stamped, "fingeer", rospy.Duration(1))
            # print("output_pose_stamped:: ",output_pose_stamped)
            return "recived"
        except:
            return "waiting"

    def look_up_tf(self):
        try:
       
            pose = geometry_msgs.msg.TransformStamped()
            pose = self.tfBuffer.lookup_transform("finger", "obj", rospy.Time(),rospy.Duration(1))
            print("DebugTransformation::pose::transform ",pose.transform.translation)
            
            pos = pose.transform.translation
            orn = pose.transform.rotation
           
            self.state["obj_relative_to_finger"]["pos"] = [pos.x,pos.y,pos.z]
            self.state["obj_relative_to_finger"]["orn"]  = [orn.x,orn.y,orn.z,orn.w]
           
            return "recived"
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print("error:: ",e)
            return "waiting"