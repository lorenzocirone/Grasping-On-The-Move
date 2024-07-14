import pinocchio
import os
from numpy.linalg import pinv

import numpy as np
import rospy
import control_msgs.msg
import trajectory_msgs.msg
import actionlib 
import time
import pickle as pkl

from gazebo_msgs.msg import ModelStates
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .ReducedModel import*
# from .ForwardKin import *
import matplotlib.pyplot as plt

# Management of control loop.
class ArmMotionControlManager:
    def __init__(self):
        rospy.init_node('tiago_arm', log_level=rospy.INFO)
        # model initialization
        model_xml = rospy.get_param('/robot_description')
        self.rmodel = getReducedModel(model_xml)
        self.rdata = self.rmodel.createData()
        self.q = pinocchio.neutral(self.rmodel)
        self.dq = np.zeros(self.rmodel.nv)

        # definition of initial and final end effector position
        self.xi_init = np.zeros(3)
        self.xi_final = np.zeros(3)

        # subscribers
        target_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.target_odom_callback)
        init_conf_sub = rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, self.init_conf_callback)

        #publishers
        self.arm_command_topic = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)
        self.pub_gripper_controller = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=10)
        self.rate = rospy.Rate(10)  # 10 Hz
        
        self.history = {
            "des_xi_x" : [],
            "des_xi_y" : [],
            "des_xi_z" : [],
            "des_dxi_x" : [],
            "des_dxi_y" : [],
            "des_dxi_z" : [],
            "des_ddxi_x" : [],
            "des_ddxi_y" : [],
            "des_ddxi_z" : [],
            "xi_x" : [],
            "xi_y" : [],
            "xi_z" : [],
            "cntrl_x" : [],
            "cntrl_y" : [],
            "cntrl_z" : [],
            "pos_err_x" : [],
            "pos_err_y" : [],
            "pos_err_z" : []
        }
        
    def arm_trajectory_goal_msg(self, joint_names, q_des, dq_des, duration):
        # Create the ROS message to be given back
        msg = trajectory_msgs.msg.JointTrajectory()
        msg.joint_names = joint_names
        # inside msg there is the additional message jointTrajectoryPoint
        msg_point = trajectory_msgs.msg.JointTrajectoryPoint()
        msg_point.positions = q_des
        msg_point.velocities = dq_des
        msg_point.time_from_start = rospy.Duration(duration)
        # append trajectory_msg to Control_msg
        msg.points.append(msg_point)
        return msg

    def target_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'unit_sphere':
                self.xi_final[0] = msg.pose[i].position.x
                self.xi_final[1] = msg.pose[i].position.y
                self.xi_final[2] = msg.pose[i].position.z
            elif msg.name[i] == 'tiago':
                self.q[0] = msg.pose[i].position.x
                self.q[1] = msg.pose[i].position.y
                self.q[2] = msg.pose[i].position.z
                self.q[3] = msg.pose[i].orientation.x
                self.q[4] = msg.pose[i].orientation.y
                self.q[5] = msg.pose[i].orientation.z
                self.q[6] = msg.pose[i].orientation.w

    def init_conf_callback(self, msg):
        for i in range(len(msg.actual.positions)):
            if self.rmodel.nq > 10:
                self.q[i+7] = msg.actual.positions[i]
                self.dq[i+6] = msg.actual.velocities[i]
            else:
                self.q[i] = msg.actual.positions[i]
                self.dq[i] = msg.actual.velocities[i]

    def close_gripper(self):
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        trajectory_points = JointTrajectoryPoint()
        trajectory_points.positions = [0.015, 0.015]
        trajectory_points.time_from_start = rospy.Duration(0.5)
        trajectory.points.append(trajectory_points)
        self.pub_gripper_controller.publish(trajectory)            
        rospy.loginfo("Close gripper command published")      
    
    def parameter_law(self, tau): 
        s = 6*tau**5 - 15*tau**4 + 10*tau**3           # s(tau)
        ds = 30*tau**4 - 60*tau**3 + 30*tau**2      # s_dot(tau)
        dds = 120*tau**3 - 180*tau**2 + 60*tau   # s_dot_dot(tau)
        return s, ds, dds
    
    def planning_trajectory(self, t, T, initial_position, final_position):
        # time vector for the number of samples
        tau = min(1, t/T)
        # parameters of the trajectory
        s, ds, dds = self.parameter_law(tau)
        # compute desired position, velocity and accelereation of the end effector
        pos = initial_position + s*(final_position - initial_position)
        vel = ds*(final_position - initial_position)
        acc = dds*(final_position - initial_position)
        return pos, vel, acc

    def start(self):
        rospy.sleep(11)
        
        start_time = time.time()
        # compute forward kinematics
        pinocchio.framesForwardKinematics(self.rmodel, self.rdata, self.q)
        
        # gripper grasping frame id
        ggf_id = self.rmodel.getFrameId("gripper_grasping_frame")
        
        # get the initial end effector position
        self.xi_init = self.rdata.oMf[ggf_id].translation
        #arm joint names
        joint_names = [
            'arm_1_joint',
            'arm_2_joint',
            'arm_3_joint',
            'arm_4_joint',
            'arm_5_joint',
            'arm_6_joint',
            'arm_7_joint'
        ]
        # frequency of the control loop
        frequency = 10
        dt = 1 / frequency
        rate = rospy.Rate(frequency) # Hz
        # time of the trajectory
        t = 0
        T = 5
        # PD parameters
        k = np.diag([20, 20, 40])
        
        print("*************************", time.time()-start_time)
        while not rospy.is_shutdown():
            
            # update frame placement
            pinocchio.framesForwardKinematics(self.rmodel, self.rdata, self.q)
            
            # compute jacobian and pseudoinverse
            J = pinocchio.computeFrameJacobian(self.rmodel, self.rdata, self.q, ggf_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            
            J = J[:3, :] # make a copy through slicing
            Jpinv = pinv(J)
            # compute jacobian derivative
            pinocchio.computeJointJacobiansTimeVariation(self.rmodel, self.rdata, self.q, self.dq)
            dJ = pinocchio.getFrameJacobianTimeVariation(self.rmodel, self.rdata, ggf_id, pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            dJ = dJ[:3, :] # make a copy thorugh slicing
            
            # compute actual position and velocity of the end effector 
            xi = self.rdata.oMf[ggf_id].translation

            # compute NEXT desired position, velocity and acceleration of the end effector
            xi_des, dxi_des, ddxi_des = self.planning_trajectory(t + dt, T, xi, self.xi_final)
            
            # PD + feedforward
            pos_err = self.xi_final - xi
            cntrl_input = dxi_des + np.matmul(k,pos_err)

            # compute joint velocity and position
            dq = np.matmul(Jpinv, cntrl_input)
            q  = self.q[-7:]  +  dq[-7:]*dt

            # compute message to be sent
            msg = self.arm_trajectory_goal_msg(joint_names, q[-7:], dq[-7:], dt)
            # send message
            self.arm_command_topic.publish(msg)            
            
            t += dt
            
            self.history["des_xi_x"].append(xi_des[0])
            self.history["des_xi_y"].append(xi_des[1])
            self.history["des_xi_z"].append(xi_des[2])
            self.history["des_dxi_x"].append(dxi_des[0])
            self.history["des_dxi_y"].append(dxi_des[1])
            self.history["des_dxi_z"].append(dxi_des[2])
            self.history["des_ddxi_x"].append(ddxi_des[0])
            self.history["des_ddxi_y"].append(ddxi_des[1])
            self.history["des_ddxi_z"].append(ddxi_des[2])
            self.history["xi_x"].append(xi[0])
            self.history["xi_y"].append(xi[1])
            self.history["xi_z"].append(xi[2])
            self.history["cntrl_x"].append(cntrl_input[0])
            self.history["cntrl_y"].append(cntrl_input[1])
            self.history["cntrl_z"].append(cntrl_input[2])
            self.history["pos_err_x"].append(pos_err[0])
            self.history["pos_err_y"].append(pos_err[1])
            self.history["pos_err_z"].append(pos_err[2])

            print(f"t : {round(t,2)}| Error: {round(sum(pos_err),2)}", end="\r")
            if(t > T and sum(pos_err) < 0.1):
                self.close_gripper()
                break
            
        print("*************************", time.time()-start_time)
        return self.history

def main():
    motion_control_manager = ArmMotionControlManager()
    history = motion_control_manager.start()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    savefile_path = "results/quintic_arm.pkl"
    output_path = os.path.join(script_dir, savefile_path)

    with open(output_path, "wb") as pkl_file:
        pkl.dump(history,pkl_file,protocol=pkl.HIGHEST_PROTOCOL)