import pinocchio as pin
import upkie_description
from scipy.spatial.transform import Rotation as R
import numpy as np


# Load robot
robot = upkie_description.load_in_pinocchio(root_joint=None)

# Random configuration
# q0 = pin.randomConfiguration(robot.model)
print(robot.model)
print(robot.data)
print(robot.q0)

# Print out the placement of each joint of the kinematic tree

def pseudo_height(observation):
    # Get positions
    position = observation[:6]
    # Get IMU
    imu = observation[18:22]
    imu_rot = R.from_quat(imu).as_matrix()
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(robot.model, robot.data, position)
    heights = [(imu_rot @ oMi.translation.T.flat)[2] for oMi in robot.data.oMi]

    # for name, oMi in zip(robot.model.names, robot.data.oMi):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #         .format( name, *oMi.translation.T.flat )))
    # print(heights)

    return (0 - np.min(heights))

def pseudo_height_2(observation):
    # Get positions
    position = observation[:6]
    # Get IMU
    imu = observation[18:22]
    imu_rot = R.from_quat(imu).as_matrix()
    # Perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(robot.model, robot.data, position)
    heights = [(imu_rot @ oMi.translation.T.flat)[2] for oMi in robot.data.oMi][1:]

    # for name, oMi in zip(robot.model.names, robot.data.oMi):
        #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
        #         .format( name, *oMi.translation.T.flat )))
    # print(heights)

    return (np.max(heights) - np.min(heights))


if __name__ == "__main__":
    obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0])
    r1 = pseudo_height(obs)
    r2 = pseudo_height_2(obs)
    print(r1, r2)
    