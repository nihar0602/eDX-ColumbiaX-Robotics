#!/usr/bin/env python

import math
import numpy
import time
from threading import Thread, Lock
import numpy as np 

import rospy
import tf
from geometry_msgs.msg import Transform
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from urdf_parser_py.urdf import URDF


def S_matrix(w):    # Skew matrix
    S = numpy.zeros((3,3))
    S[0,1] = -w[2]
    S[0,2] =  w[1]
    S[1,0] =  w[2]
    S[1,2] = -w[0]
    S[2,0] = -w[1]
    S[2,1] =  w[0]
    return S


def rotation_matrix(matrix):
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    return R33


def translation_matrix(matrix):
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    #t43 = numpy.zeros((3,1))
    #t43[:3] = R[:3, 3:]
    t43 = R[:3, 3]
    return t43


# This is the function that must be filled in as part of the Project.
def cartesian_control(joint_transforms, b_T_ee_current, b_T_ee_desired,
                      red_control, q_current, q0_desired):
    num_joints = len(joint_transforms)
    dq = numpy.zeros(num_joints)
    #-------------------- Fill in your code here ---------------------------
################################################################################################################
    """
        Part 1: Compute delta_X

        Step 1: Inverse of Base to Current Robot Frame (b_T_ee_current)
        Step 2: transform frame from ee_current with respect to base To ee_current to ee_desired
        Step 3: delta_x_translation = Extract Translation
        Step 4: delta_x_rotation = Extract Rotation
    """
    # print("No of Joints: ", num_joints)

    ee_T_b  = tf.transformations.inverse_matrix(b_T_ee_current)
    ee_T_ee = np.dot(ee_T_b, b_T_ee_desired)

    # print(ee_T_b.shape)
    # print(ee_T_ee.shape)

    del_x_trans = tf.transformations.translation_from_matrix(ee_T_ee)
    del_x_rot = tf.transformations.rotation_from_matrix(ee_T_ee) 

    ROT = np.dot(del_x_rot[0],del_x_rot[1])
    # print(ee_T_ee)

    # print("Trans: ", del_x_trans)
    # print("Rot: ", del_x_rot)
    

    lin_gain = 3 
    rot_gain = 1.5 

    delta_x = np.append(del_x_trans * lin_gain , ROT * rot_gain)


####################################################################################################################
    """
        Part 2: Compute x_dot = gain * delta_X
    """

    x_dot = 1 * delta_x

#####################################################################################################################

    """
        Part 3: Compute v_ee

        Step 1: Get ee_R_b
        Step 2: Compute v_ee_translation and v_ee_rotation
        Step 3: Normalize v_ee_translation and v_ee_rotation
        Step 4: Build v_ee

    """

    b_R_ee = ee_T_ee[:3,:3]
    ee_R_b = tf.transformations.inverse_matrix(b_R_ee)

    # print(ee_R_b.shape)
    
    x11 = np.concatenate((ee_R_b,np.zeros((3,3))),axis=0)
    x12 = np.concatenate((np.zeros((3,3)),ee_R_b),axis=0)

    xx = np.concatenate((x11,x12),axis=1)

    V_ee = np.dot(xx,x_dot)
    # print("HEHE", V_ee)

    V_ee_trans = V_ee[:3]
    # print(V_ee_trans)

    V_ee_rot = V_ee[3:]
    # print(V_ee_rot)


    J = numpy.empty((6, 0))

#####################################################################################################################

    """
    Part 4: Compute Numerical Jacobian

    For each joint:
    Step 1: Compute j_T_ee = j_T_b * b_T_ee
    Step 2: Get ee_R_j = (j_R_ee)^T
    Step 3: Get skew of j_translation_ee
    Step 4: - ee_R_j * S(j_translation_ee)
    Step 5: Build V_j
    Step 6: Get joint axis
    Step 7: Logic to keep only relevant column
    Step 8: Append to build J
    """

    for i in range(num_joints):

        j_T_b = tf.transformations.inverse_matrix(joint_transforms[i])
        j_T_ee = np.dot(j_T_b, b_T_ee_current)

        # print("YEH HAI j_T_ee: ",j_T_ee)
        # print(j_T_ee.shape)

        ee_T_j = tf.transformations.inverse_matrix(j_T_ee)

        j_t_ee = tf.transformations.translation_from_matrix(j_T_ee)

        j_R_ee = j_T_ee[:3,:3]
        # print(j_R_ee.shape)

        ee_R_j = tf.transformations.inverse_matrix(j_R_ee)

        S = S_matrix(j_t_ee)

        v_j_12 = np.dot(-ee_R_j,S)
        # v_j_11 = ee_R_j #for_Reference 
        v_j_21 = np.zeros((3,3))
        # v_j_22 = ee_R_j #for_reference

        V_j_1 = np.concatenate((ee_R_j,v_j_12),axis=1)  #1st row - 3rd row and all 6 columns  
        V_j_2 = np.concatenate((v_j_21,ee_R_j),axis=1)  #4th row - 6throw and all 6 columns 
        V_j = np.concatenate((V_j_1,V_j_2),axis=0)

        # print('YEH HAI V_J ka shape: ', V_j.shape)
        # print(v_j_12.shape)

        J = np.column_stack((J, V_j[:,5]))
##########################################################################################################################################
    """
    Part 5: Compute pseduo-Inverse: J+
    """

    J_pinv = np.linalg.pinv(J,rcond=1e-2)
    # print(J_pinv.shape)

##########################################################################################################################################
    """
    Part 6: q_dot = J_plus*V_ee(i.e x_dot)
    """

    dq = np.dot(J_pinv,x_dot)

###################################################-----------BONUS: NULL SPACE CONTROL ----------########################################

    if red_control == True:


        J_pinv = np.linalg.pinv(J,rcond=0)

        identity = np.identity(7)
        J_dot_p = np.dot(J_pinv,J)

        # print("q0_desired: " , q0_desired)
        # print("q0_shape: ", type(q0_desired))

        # print("q0_current: " , q_current)
        # print("q0_shape: ", len(q_current))

        q_diff = q0_desired - q_current[0]

        q_diff_array = np.array([q_diff,0,0,0,0,0,0])

        p = 50

        dq_null = np.dot((identity - J_dot_p),(p*q_diff_array))

        dq = np.dot(J_pinv,x_dot) + dq_null

    

    #----------------------------------------------------------------------
    return dq
    
def convert_from_message(t):
    trans = tf.transformations.translation_matrix((t.translation.x,
                                                  t.translation.y,
                                                  t.translation.z))
    rot = tf.transformations.quaternion_matrix((t.rotation.x,
                                                t.rotation.y,
                                                t.rotation.z,
                                                t.rotation.w))
    T = numpy.dot(trans,rot)
    return T

# Returns the angle-axis representation of the rotation contained in the input matrix
# Use like this:
# angle, axis = rotation_from_matrix(R)
def rotation_from_matrix(matrix):
    R = numpy.array(matrix, dtype=numpy.float64, copy=False)
    R33 = R[:3, :3]
    # axis: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = numpy.linalg.eig(R33.T)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    axis = numpy.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = numpy.linalg.eig(R)
    i = numpy.where(abs(numpy.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    # rotation angle depending on axis
    cosa = (numpy.trace(R33) - 1.0) / 2.0
    if abs(axis[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
    elif abs(axis[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
    angle = math.atan2(sina, cosa)
    return angle, axis

class CartesianControl(object):

    #Initialization
    def __init__(self):
        #Loads the robot model, which contains the robot's kinematics information
        self.robot = URDF.from_parameter_server()

        #Subscribes to information about what the current joint values are.
        rospy.Subscriber("/joint_states", JointState, self.joint_callback)

        #Subscribes to command for end-effector pose
        rospy.Subscriber("/cartesian_command", Transform, self.command_callback)

        #Subscribes to command for redundant dof
        rospy.Subscriber("/redundancy_command", Float32, self.redundancy_callback)

        # Publishes desired joint velocities
        self.pub_vel = rospy.Publisher("/joint_velocities", JointState, queue_size=1)

        #This is where we hold the most recent joint transforms
        self.joint_transforms = []
        self.q_current = []
        self.x_current = tf.transformations.identity_matrix()
        self.R_base = tf.transformations.identity_matrix()
        self.x_target = tf.transformations.identity_matrix()
        self.q0_desired = 0
        self.last_command_time = 0
        self.last_red_command_time = 0

        # Initialize timer that will trigger callbacks
        self.mutex = Lock()
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

    def command_callback(self, command):
        self.mutex.acquire()
        self.x_target = convert_from_message(command)
        self.last_command_time = time.time()
        self.mutex.release()

    def redundancy_callback(self, command):
        self.mutex.acquire()
        self.q0_desired = command.data
        self.last_red_command_time = time.time()
        self.mutex.release()        
        
    def timer_callback(self, event):
        msg = JointState()
        self.mutex.acquire()
        if time.time() - self.last_command_time < 0.5:
            dq = cartesian_control(self.joint_transforms, 
                                   self.x_current, self.x_target,
                                   False, self.q_current, self.q0_desired)
            msg.velocity = dq
        elif time.time() - self.last_red_command_time < 0.5:
            dq = cartesian_control(self.joint_transforms, 
                                   self.x_current, self.x_current,
                                   True, self.q_current, self.q0_desired)
            msg.velocity = dq
        else:            
            msg.velocity = numpy.zeros(7)
        self.mutex.release()
        self.pub_vel.publish(msg)
        
    def joint_callback(self, joint_values):
        root = self.robot.get_root()
        T = tf.transformations.identity_matrix()
        self.mutex.acquire()
        self.joint_transforms = []
        self.q_current = joint_values.position
        self.process_link_recursive(root, T, joint_values)
        self.mutex.release()

    def align_with_z(self, axis):
        T = tf.transformations.identity_matrix()
        z = numpy.array([0,0,1])
        x = numpy.array([1,0,0])
        dot = numpy.dot(z,axis)
        if dot == 1: return T
        if dot == -1: return tf.transformation.rotation_matrix(math.pi, x)
        rot_axis = numpy.cross(z, axis)
        angle = math.acos(dot)
        return tf.transformations.rotation_matrix(angle, rot_axis)

    def process_link_recursive(self, link, T, joint_values):
        if link not in self.robot.child_map: 
            self.x_current = T
            return
        for i in range(0,len(self.robot.child_map[link])):
            (joint_name, next_link) = self.robot.child_map[link][i]
            if joint_name not in self.robot.joint_map:
                rospy.logerror("Joint not found in map")
                continue
            current_joint = self.robot.joint_map[joint_name]        

            trans_matrix = tf.transformations.translation_matrix((current_joint.origin.xyz[0], 
                                                                  current_joint.origin.xyz[1],
                                                                  current_joint.origin.xyz[2]))
            rot_matrix = tf.transformations.euler_matrix(current_joint.origin.rpy[0], 
                                                         current_joint.origin.rpy[1],
                                                         current_joint.origin.rpy[2], 'rxyz')
            origin_T = numpy.dot(trans_matrix, rot_matrix)
            current_joint_T = numpy.dot(T, origin_T)
            if current_joint.type != 'fixed':
                if current_joint.name not in joint_values.name:
                    rospy.logerror("Joint not found in list")
                    continue
                # compute transform that aligns rotation axis with z
                aligned_joint_T = numpy.dot(current_joint_T, self.align_with_z(current_joint.axis))
                self.joint_transforms.append(aligned_joint_T)
                index = joint_values.name.index(current_joint.name)
                angle = joint_values.position[index]
                joint_rot_T = tf.transformations.rotation_matrix(angle, 
                                                                 numpy.asarray(current_joint.axis))
                next_link_T = numpy.dot(current_joint_T, joint_rot_T) 
            else:
                next_link_T = current_joint_T

            self.process_link_recursive(next_link, next_link_T, joint_values)
        
if __name__ == '__main__':
    rospy.init_node('cartesian_control', anonymous=True)
    cc = CartesianControl()
    rospy.spin()
 