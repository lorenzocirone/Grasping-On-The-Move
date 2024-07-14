import pinocchio
import numpy as np
import sys
import os
from os.path import dirname, join, abspath 


def getReducedModel(model_xml):
    model = pinocchio.buildModelFromXML(model_xml, pinocchio.JointModelFreeFlyer())
    
    # Create a list of joints to take
    jointsToLock = [
        'caster_back_left_1_joint',
        'caster_back_left_2_joint',
        'caster_back_right_1_joint',
        'caster_back_right_2_joint',
        'caster_front_left_1_joint',
        'caster_front_left_2_joint',
        'caster_front_right_1_joint',
        'caster_front_right_2_joint',
        'suspension_left_joint',
        'wheel_left_joint',
        'suspension_right_joint',
        'wheel_right_joint',
        'torso_lift_joint',
        'head_1_joint',
        'head_2_joint',
        'gripper_left_joint',
        'gripper_right_joint'
    ]
    
    jointsToLockIDs = []
    for jn in jointsToLock:
        if model.existJointName(jn):
            jointsToLockIDs.append(model.getJointId(jn))
        else:
            print('Warning: joint ' + str(jn) + ' does not belong to the model!')

    # Set initial position of both fixed and revolute joints
    initialJointConfig = pinocchio.neutral(model)                                
 
    # Option 1: Only build the reduced model in case no display needed:
    model_reduced = pinocchio.buildReducedModel(model, jointsToLockIDs, initialJointConfig)
    
    return model_reduced
