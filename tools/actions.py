import numpy as np

# import upkie_description

# robot = upkie_description.load_in_pinocchio(root_joint=None)
# model = robot.model

joints = {
    ('left_hip', 0),
    ('left_knee', 1),
    ('left_wheel', 2),
    ('right_hip', 3),
    ('right_knee', 4),
    ('right_wheel', 5)
}

# joints_id = [
#     model.getJointId(joint)-1 for joint in joints
# ]

def create_action_dict(
    position = None,
    velocity = None,
    torque = None
):

    servo_dict = {
        joint : {
            'position': position[i],
            'velocity': velocity[i],
            'torque': torque[i]
            }
        for joint, i in joints
    }

    return {'servo': servo_dict}
