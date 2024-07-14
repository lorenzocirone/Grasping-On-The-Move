#!/usr/bin/env python

import actionlib 
import control_msgs.msg
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl
import pinocchio
from qpsolvers import available_solvers, print_matrix_vector, solve_qp
import rospy 
import time 
import trajectory_msgs.msg
import pickle as pkl

from control_msgs.msg import JointTrajectoryControllerState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist 
from numpy.linalg import pinv
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .plots import *
from .quaternion import *
from .ReducedModel import*
from .tangent_point import *

class ArmBaseController:
    def __init__(self):
        rospy.init_node('tiago_grasping_on_the_move', log_level=rospy.INFO)
        # [ARM] Reduced model
        model_xml = rospy.get_param('/robot_description')
        self.rmodel = getReducedModel(model_xml)
        self.rdata = self.rmodel.createData()
        self.q = pinocchio.neutral(self.rmodel)
        self.dq = np.zeros(self.rmodel.nv)

        # [ARM] Initial and final end effector position
        self.xi_init = np.zeros(3)
        self.xi_final = np.zeros(3)

        # [ARM] Nodes
        init_conf_sub = rospy.Subscriber('/arm_controller/state', JointTrajectoryControllerState, self.init_conf_callback)
        self.arm_command_topic = rospy.Publisher('/arm_controller/command', JointTrajectory, queue_size=1)

        # [Base] Node settings
        self.velocity_publisher = rospy.Publisher('/mobile_base_controller/cmd_vel', Twist, queue_size=10)
        self.base_odom_subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, self.base_odom_callback)
        self.pub_gripper_controller = rospy.Publisher('/gripper_controller/command', JointTrajectory, queue_size=1)
        self.rate = rospy.Rate(10)  # 10 Hz

        # [Base] Goal flags
        self.is_desired_pose_reached = False
        self.is_desired_orientation_updated = False
        self.second_phase = False
        self.set_init_rho = False
        
        # [Base] Goal thresholds
        self.gripper_threshold = 1.65 # PRMstarkConfigDefault
        self.threshold = 1.75 # Grasping from the front
        self.gripper_threshold_n = 1.60
        self.threshold_n = 1.50 # Leave the ball from the frontS
        
        # [Base] Closest approach circle
        self.r_c = 0.6
        self.d = 1.0
        self.d_n = 1.0

        # [Base] TIAgo initial configuration
        self.eps_b_x = 0.0
        self.eps_b_y = 0.0
        self.theta = 0.0 
        self.orientation_quaternion = Quaternion()

        # [Base] Target position
        self.eps_t_x = 0.0
        self.eps_t_y = 0.0

        # [Base] Closest approach position
        self.eps_c_x = 0
        self.eps_c_y = 0
        self.eps_c_n_x = 0
        self.eps_c_n_y = 0

        # [Base] Next target position
        self.eps_n_x = 0.0
        self.eps_n_y = 0.0

        # [Base] Errors
        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 1.0
        self.rho_n = 1.0
        self.history = {
            "v" : [],
            "w" : [],
            "alpha" : [],
            "rho" : [],
            "beta": [],
            "rho_n": [],
            "time" : 0.0,
            "x" : [],
            "y" : [],
            "dq_1" : [],
            "dq_2" : [],
            "dq_3" : [],
            "dq_4" : [],
            "dq_5" : [],
            "dq_6" : [],
            "dq_7" : [],
            "q_1" : [],
            "q_2" : [],
            "q_3" : [],
            "q_4" : [],
            "q_5" : [],
            "q_6" : [],
            "q_7" : [],
            "delta_a1" : [],
            "delta_a2" : [],
            "delta_a3" : [],
            "delta_v" : [],
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

        # [Base] Controller
        self.v_b = 0.3
        self.w_b = 0.0
        self.k_alpha = 4
        self.k_beta = 2.2
        self.block = False

    def base_odom_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] == 'tiago':
                self.q[0] = msg.pose[i].position.x
                self.q[1] = msg.pose[i].position.y
                self.q[2] = msg.pose[i].position.z
                self.q[3] = msg.pose[i].orientation.x
                self.q[4] = msg.pose[i].orientation.y
                self.q[5] = msg.pose[i].orientation.z
                self.q[6] = msg.pose[i].orientation.w

                self.eps_b_x = msg.pose[i].position.x
                self.eps_b_y = msg.pose[i].position.y
                self.orientation_quaternion.set_coeffs(
                    w = msg.pose[i].orientation.w,
                    x = msg.pose[i].orientation.x, 
                    y = msg.pose[i].orientation.y,
                    z = msg.pose[i].orientation.z
                )
                rad, _ = self.orientation_quaternion.to_euler_angles()
                self.theta = rad[2]

                self.d = math.sqrt((self.eps_t_x - self.eps_b_x)**2 + (self.eps_t_y - self.eps_b_y)**2)
                self.d_n = math.sqrt((self.eps_n_x - self.eps_b_x)**2 + (self.eps_n_y - self.eps_b_y)**2)

                self.rho = 0 if self.r_c > self.d else math.sqrt(self.d**2 - self.r_c**2)
                self.alpha = math.atan2(self.eps_b_y - self.eps_c_y, self.eps_b_x - self.eps_c_x) + math.pi - self.theta

                self.beta = math.atan2(self.eps_b_y - self.eps_c_n_y, self.eps_b_x - self.eps_c_n_x) + math.pi - self.theta
                self.rho_n = 0 if self.r_c > self.d_n else math.sqrt(self.d_n**2 - self.r_c**2)
                
                self.history["alpha"].append(self.alpha)
                self.history["beta"].append(self.beta)
                self.history["rho"].append(self.rho)
                self.history["rho_n"].append(self.rho_n)
                self.history["x"].append(self.eps_b_x)
                self.history["y"].append(self.eps_b_y)
                
                self.eps_c_x,self.eps_c_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_t_x,self.eps_t_y,self.r_c)
                self.eps_c_n_x,self.eps_c_n_y = tangent_point(self.eps_b_x,self.eps_b_y,self.eps_n_x,self.eps_n_y,self.r_c)

            if msg.name[i] == 'unit_sphere':    
                self.eps_t_x = msg.pose[i].position.x
                self.eps_t_y = msg.pose[i].position.y
                self.xi_final[0] = msg.pose[i].position.x - 0.06
                self.xi_final[1] = msg.pose[i].position.y + 0.04
                self.xi_final[2] = msg.pose[i].position.z - 0.20
            if msg.name[i] == 'unit_box_1':    
                self.eps_n_x = msg.pose[i].position.x
                self.eps_n_y = msg.pose[i].position.y

                

    def close_gripper(self):
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        trajectory_points = JointTrajectoryPoint()
        trajectory_points.positions = [0.015, 0.015]
        trajectory_points.time_from_start = rospy.Duration(0.5)
        trajectory.points.append(trajectory_points)
        self.pub_gripper_controller.publish(trajectory)            

    def open_gripper(self):
        trajectory = JointTrajectory()
        trajectory.joint_names = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        trajectory_points = JointTrajectoryPoint()
        trajectory_points.positions = [0.044, 0.044]
        trajectory_points.time_from_start = rospy.Duration(0.5)
        trajectory.points.append(trajectory_points)
        self.pub_gripper_controller.publish(trajectory)                    

    def final_errors(self,history):
        plot_errors(history)

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

    def init_conf_callback(self, msg):
        for i in range(len(msg.actual.positions)):
            if self.rmodel.nq > 10:
                self.q[i+7] = msg.actual.positions[i]
                self.dq[i+6] = msg.actual.velocities[i]
            else:
                self.q[i] = msg.actual.positions[i]
                self.dq[i] = msg.actual.velocities[i]

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
        rospy.sleep(2.5)
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
        t_traj = 0
        # PD parameters
        k = np.diag([10, 10, 20])
        solver = "quadprog" #hpipm
        
        while not rospy.is_shutdown() and not self.is_desired_pose_reached:
            if(t > 9.7 and self.block == False):
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
                pos_err = xi_des - xi

                cntrl_input = dxi_des + np.matmul(k,pos_err)
                dq = np.matmul(Jpinv, cntrl_input)
                q  = self.q[-7:] + dq[-7:]*dt        

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
                
                # # QP solver
                v_r = self.v_b
                d_wheels = 0.445
                r_wheels = 0.1
                
                # Q = H
                k_a = 1e6
                error_b = np.linalg.norm(self.rho)
                lambda_q = np.hstack(
                    [
                        np.repeat(1/error_b,2),
                        np.repeat(k_a,7)
                    ]
                )

                error_ee = np.linalg.norm(pos_err)
                lambda_v = np.hstack(
                    [
                        np.repeat(1/error_ee,4)
                    ]
                )

                Q = np.diagflat(
                    [
                        np.hstack([lambda_q,lambda_v])
                    ]
                )

                # Aeq = Jcal
                aeq_w_block = np.vstack(
                    [
                        np.zeros((3,2)),
                        np.array([0.5,0.5])
                    ]
                )
                aeq_q_block = np.vstack(
                    [
                        J[:3,:7],
                        np.zeros((1,7))
                    ]
                )
                aeq_delta_a_block = np.vstack(
                    [
                        np.eye(3),
                        np.zeros((1,3))
                    ]
                )
                aeq_delta_v_block = np.vstack(
                    [
                        np.zeros((3,1)),
                        np.array([1])
                    ]
                )

                Aeq = np.hstack(
                    [
                        aeq_w_block,
                        aeq_q_block,
                        aeq_delta_a_block,
                        aeq_delta_v_block
                    ]
                )

                # Beq = v
                Beq = np.vstack(
                    [
                        cntrl_input.reshape(-1,1),
                        np.array([v_r])
                    ]
                )

                # Ain = A
                ain_w_block = np.zeros((14,2))
                ain_q_block = np.vstack([np.eye(7),np.eye(7)])
                ain_delta_a_block = np.zeros((14,3))
                ain_delta_v_block = np.zeros((14,1))
                Ain = np.hstack(
                    [
                        ain_w_block,
                        ain_q_block,
                        ain_delta_a_block,
                        ain_delta_v_block 
                    ]
                )

                # Bin = B

                q_lower = [0, -1.57, -3.53, -0.39, -2.09, -1.41, -2.09]
                q_upper = [2.75, 1.09, 1.57, 2.36, 2.09, 1.41, 2.09]

                max_q_vector = []
                min_q_vector = []

                for i in range(len(q)):
                    max_q_vector.append((q_upper[i]-q[i])/dt)
                    min_q_vector.append((q[i]-q_lower[i])/dt)

                max_q_vector = np.array(max_q_vector).reshape(-1,1)
                min_q_vector = np.array(min_q_vector).reshape(-1,1)

                Bin = np.vstack(
                    [
                        max_q_vector,
                        min_q_vector
                    ]
                )

                # Upper and lower bounds
                ub = np.array([7.7, 7.7, 2.70, 3.66, 4.58, 4.58, 1.95, 1.76, 1.76, np.inf, np.inf, np.inf, np.inf])
                lb = np.array([-7.7, -7.7,-2.70, -3.66, -4.58, -4.58, -1.95, -1.76, -1.76, 0, 0, 0, 0])
                c = np.zeros((13,1))
                
                sol = solve_qp(Q, c, Ain, Bin, Aeq, Beq, lb=lb, ub=ub, solver=solver)                
                self.v_b = r_wheels * (sol[0] + sol[1])/2 + 0.05
                dq = sol[2:9]
                
                self.history["dq_1"].append(dq[0])
                self.history["dq_2"].append(dq[1])
                self.history["dq_3"].append(dq[2])
                self.history["dq_4"].append(dq[3])
                self.history["dq_5"].append(dq[4])
                self.history["dq_6"].append(dq[5])
                self.history["dq_7"].append(dq[6])

                self.history["q_1"].append(q[0])
                self.history["q_2"].append(q[1])
                self.history["q_3"].append(q[2])
                self.history["q_4"].append(q[3])
                self.history["q_5"].append(q[4])
                self.history["q_6"].append(q[5])
                self.history["q_7"].append(q[6])
                
                self.history["delta_a1"].append(sol[9])
                self.history["delta_a2"].append(sol[10])
                self.history["delta_a3"].append(sol[11])
                self.history["delta_v"].append(sol[12])
                
                # print(sol)
            
                if ((t > T and sum(pos_err) < 0.20)):
                    self.close_gripper()
                    self.v_b = 0.2
                    print("**********************", round(time.time() - start_time,2))
                    self.block = True
                
                msg = self.arm_trajectory_goal_msg(joint_names, q[-7:], dq[-7:], dt)
                self.arm_command_topic.publish(msg)
                t_traj += dt
                print(f"Time: {round(t_traj,2)} | Error: {round(sum(pos_err),2)}", end="\r")

            
            if self.rho > self.threshold:
                vel_msg = Twist()
                vel_msg.linear.x = self.v_b
                vel_msg.angular.z = (self.k_alpha * self.alpha) * (self.v_b / self.rho)
                self.w_b = vel_msg.angular.z
                
            else:
                if self.rho_n <= self.gripper_threshold_n:
                    self.open_gripper()

                if self.rho_n > self.threshold_n:
                    vel_msg = Twist()
                    vel_msg.linear.x = self.v_b
                    vel_msg.angular.z = (self.k_beta * self.beta) * (self.v_b / self.rho_n)
                    self.w_b = vel_msg.angular.z
                else:
                    self.is_desired_pose_reached = True
                    end_time = time.time()
                    self.history["time"] = round(end_time - start_time,1)
                    vel_msg = Twist()
                    vel_msg.linear.x = 0.0
                    vel_msg.angular.z = 0.0
            
            self.history["v"].append(self.v_b)
            self.history["w"].append(self.w_b)
            self.velocity_publisher.publish(vel_msg)
            self.rate.sleep()
            t += dt
            
        return self.history


def main():
    controller = ArmBaseController()
    history = controller.start()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    savefile_path = "results/qp_test.pkl"
    output_path = os.path.join(script_dir, savefile_path)

    with open(output_path, "wb") as pkl_file:
        pkl.dump(history,pkl_file,protocol=pkl.HIGHEST_PROTOCOL)