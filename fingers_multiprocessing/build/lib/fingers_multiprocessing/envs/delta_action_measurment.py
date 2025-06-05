

import pybullet as p
from mamad_util import JointInfo
import yaml
import random
from pkg_resources import resource_string,resource_filename

from mamad_util import JointInfo

class caclulate_delta_xyz():

    def __init__(self):
        self.robot,self.robot_id = self.setUpWorld(timestep=1000)
        self.threshold= 0.001

        #jointInfo
        self.jointInfo = JointInfo()     
        self.jointInfo.get_infoForAll_joints(self.robot)
        numJoints = p.getNumJoints(self.robot_id)
        num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
        self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
        self.active_joints_info = self.jointInfo.getActiveJointsInfo()

        # ee 
         
        EEName = "fingertip_FF"        
        name = EEName.encode(encoding='UTF-8',errors='strict') 
        info = self.jointInfo.searchBy(key="linkName",value = name)[0]
        self.EE_index = info["jointIndex"]


        # state
        self.current_joint_command = None

    def calculate_joint_command_for_current_ee_command(self,goal_xyz):
        self.current_joint_command =  p.calculateInverseKinematics(self.robot_id,self.EE_index,goal_xyz)
        return self.current_joint_command

    def setUpWorld(self,timestep=1000):
        """
        Reset the simulation to the beginning and reload all models.

        Parameters
        ----------
        initialSimSteps : int

        Returns
        -------
        baxterId : int
        endEffectorId : int 
        """
        p.connect(p.GUI)
        p.resetSimulation()
        p.setTimeStep(1/timestep)
        p.setRealTimeSimulation(0)
        p.setGravity(0., 0., -10.)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # Load Baxter
        robot = p.loadSDF("./model/model_fingers.sdf")
        robot_id = robot[0]
    
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return robot,robot_id

    def get_goal(self,finger_name):
        """
        get a new goal position from point could in workspace
        """
        finger_list = ["FF","MF","RF"]
        point_cloud = None

        if finger_name not in finger_list:
            #Todo: raise an error
            print("wrong finger name")
    
        #load point cloud for finger
        path = resource_filename(__name__,"./model/"+finger_name+".yml")
    
        with open(path, "r") as stream:
            try:
                point_cloud = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        goal  = random.choice(point_cloud["vertix"])
        return goal

    def get_current_joint_values_for_finger(self):
        joint_values= self.getObservation_joint()
        # print("\n\n")
        # print("get_current_joint_values_for_finger::joint_values:: ",joint_values)
        # print("\n\n")

        ff_joint_values = joint_values[:4]
        return ff_joint_values

    def get_joint_command_for_current_ee_command(self):
        ff_joint_command = self.current_joint_command[:4]
        return ff_joint_command

    def check_the_goal_has_been_achived(self):
        goal_joint_command = self.get_joint_command_for_current_ee_command()
        current_joint_values = self.get_current_joint_values_for_finger()

        # print("\n\n")
        # print("goal_joint_command:: ",goal_joint_command)
        # print("current_joint_values:: ",current_joint_values)
        # print("\n\n")


        for i in range(len(goal_joint_command)):
            if abs(goal_joint_command[i]-current_joint_values[i])>self.threshold:
                return False

        return True
   
    def step(self,command):
        self.apply_action(command)
        p.stepSimulation()
        
        return self.check_the_goal_has_been_achived()
    
    def apply_action(self,jointcommand):
        i=0
        for info in self.active_joints_info:
            j_index = info["jointIndex"]
            p.setJointMotorControl2(self.robot_id ,j_index,
            p.POSITION_CONTROL,jointcommand[i], 
        	targetVelocity=0,
			force=2, 
        	maxVelocity=2,
			positionGain=1,
        	velocityGain=1)
            i+=1
    def reset(self):

        for jIndex in self.indexOf_activeJoints:
            p.resetJointState(self.robot_id ,jIndex,0) 
        p.stepSimulation()
        

    def sample(self):
        self.reset()
        goal_xyz = self.get_goal("FF")
        starting_position = self.get_ee_current_position()
        # print("goal_xyz:: ",goal_xyz)
        
        command = self.calculate_joint_command_for_current_ee_command(goal_xyz)
        done = False
        step_counter = 0
        while not done:
            step_counter +=1
            done = self.step(command)

        finishing_position = self.get_ee_current_position()
        
        diff = [0]*3
        for i in range(len(diff)):
            diff[i] = abs(finishing_position[i]-starting_position[i])


        dx = diff[0]/step_counter
        dy = diff[1]/step_counter
        dz = diff[2]/step_counter


        return dx,dy,dz

    def get_ee_current_position(self):
        finger_tip = "fingertip_FF"
        finger_tip_encoded = finger_tip.encode(encoding='UTF-8',errors='strict')
        Info = self.jointInfo.searchBy(key="linkName",value = finger_tip_encoded)[0]
        jointIndex = Info["jointIndex"]
        link_state = p.getLinkState(self.robot_id ,jointIndex)
        position = link_state[0]

        return position
    
    def getObservation_joint(self,format="list"):
    
        indexOfActiveJoints = self.jointInfo.getIndexOfActiveJoints()
        jointsInfo = self.jointInfo.getActiveJointsInfo()

        jointsStates = []
        joints_state = {} #key:joint ,value = joint state 

        for i in range(len(indexOfActiveJoints)):
          jointName  = jointsInfo[i]["jointName"]
          jointIndex = indexOfActiveJoints[i]
          jointState = p.getJointState(self.robot_id ,jointIndex)
          joints_state[jointName] = jointState[0]
          jointsStates.append(jointState[0])

        if format == "dictinary":
          return joints_state
        else:
          return jointsStates

    def get_maximum_dx_dy_dz(self):
        dx_list = []
        dy_list = []
        dz_list = []
        for i  in range(100):
            dx,dy,dz = self.sample()
            dx_list.append(dx)
            dy_list.append(dy)
            dz_list.append(dz)
        maximium = [max(dx_list),max(dy_list),max(dz_list)]
        return maximium,max(maximium)

class calculate_delta_joint():
    def __init__(self):
        self.robot,self.robot_id = self.setUpWorld(timestep=1000)
        self.threshold= 0.001

        self.joint_limits = {
            "high":[ 0.349066,1.5708,1.5708,1.5708],
            "low" :[-0.349066,0     ,0     ,0    ]
        }

        #jointInfo
        self.jointInfo = JointInfo()     
        self.jointInfo.get_infoForAll_joints(self.robot)
        numJoints = p.getNumJoints(self.robot_id)
        num_Active_joint = self.jointInfo.getNumberOfActiveJoints()
        self.indexOf_activeJoints  = self.jointInfo.getIndexOfActiveJoints()
        self.active_joints_info = self.jointInfo.getActiveJointsInfo()

    def setUpWorld(self,timestep=1000):
        """
        Reset the simulation to the beginning and reload all models.

        Parameters
        ----------
        initialSimSteps : int

        Returns
        -------
        baxterId : int
        endEffectorId : int 
        """
        p.connect(p.GUI)
        p.resetSimulation()
        p.setTimeStep(1/timestep)
        p.setRealTimeSimulation(0)
        p.setGravity(0., 0., -10.)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,0)
        # Load Baxter
        robot = p.loadSDF("./model/model_FF.sdf")
        robot_id = robot[0]
    
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING,1)

        return robot,robot_id

    def step(self,command):
        self.apply_action(command)
        p.stepSimulation()
        
        return self.check_the_goal_has_been_achived(command)
    
    def apply_action(self,jointcommand):
        i=0
        for info in self.active_joints_info:
            j_index = info["jointIndex"]
            p.setJointMotorControl2(self.robot_id ,j_index,
            p.POSITION_CONTROL,jointcommand[i], 
        	targetVelocity=0,
			force=2, 
        	maxVelocity=2,
			positionGain=1,
        	velocityGain=1)
            i+=1
    
    def reset(self):

        for jIndex in self.indexOf_activeJoints:
            p.resetJointState(self.robot_id ,jIndex,0) 
        p.stepSimulation()

    def sample(self):
  
        upper_limits = self.joint_limits["high"] 
        steps_for_joints = [0]*4
        diff = [0]*4
        delta_joint = [0]*4
    

        for i,jIndex in enumerate(self.indexOf_activeJoints):
            self.reset()
            starting_position = self.get_joint_current_position()
            jointcommand = [0]*4
            jointcommand[i] = upper_limits[i]
            
            done = False
            while not done:
                steps_for_joints[i]+=1
                done = self.step(jointcommand)
            
            finishing_position = self.get_joint_current_position()
            diff[i] = abs(finishing_position[i]-starting_position[i])
            delta_joint[i] = diff[i]/steps_for_joints[i]

        return delta_joint

  

    def check_the_goal_has_been_achived(self,jointcommand):
        current_joint_values = self.get_joint_current_position()

        for i,jc in enumerate(jointcommand):
            dist = abs(jc-current_joint_values[i]) 
            if dist>self.threshold:
                return False

        return True

    def get_joint_current_position(self):

        jointsStates = []
        
        for i,jointIndex in enumerate(self.indexOf_activeJoints ):
        
          jointState = p.getJointState(self.robot_id ,jointIndex)
        
          jointsStates.append(jointState[0])

        return jointsStates





########## xyz ###########
"""

cd_xyz = caclulate_delta_xyz()
maximum_elements, max_element = cd_xyz.get_maximum_dx_dy_dz()

print("dx,dy,dz:: ",maximum_elements[0],maximum_elements[1],maximum_elements[2])
print("max_element:: ",max_element)

"""

########## joint ############
cd_joint = calculate_delta_joint()

while(1):
    delta_joint = cd_joint.sample()

    print("delta_joint:: ",delta_joint)




        