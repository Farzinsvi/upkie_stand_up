import pinocchio as pin
import upkie_description
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Tuple


# Load robot
robot = upkie_description.load_in_pinocchio(root_joint=None)

# Random configuration
# q0 = pin.randomConfiguration(robot.model)
# print(robot.model)
# print(robot.data)
# print(robot.q0)

# Print out the placement of each joint of the kinematic tree

def rotation_matrix_from_quaternion(
    quat: Tuple[float, float, float, float]
) -> np.ndarray:
    """
    Convert a unit quaternion to the matrix representing the same rotation.

    Args:
        quat: Unit quaternion to convert, in ``[w, x, y, z]`` format.

    Returns:
        Rotation matrix corresponding to this quaternion.

    See `Conversion between quaternions and rotation matrices`_.

    .. _`Conversion between quaternions and rotation matrices`:
        https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Rotation_matrices
    """
    if abs(np.dot(quat, quat) - 1.0) > 1e-5:
        raise ValueError(f"Quaternion {quat} is not normalized")
    qw, qx, qy, qz = quat
    return np.array(
        [
            [
                1 - 2 * (qy ** 2 + qz ** 2),
                2 * (qx * qy - qz * qw),
                2 * (qw * qy + qx * qz),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx ** 2 + qz ** 2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx ** 2 + qy ** 2),
            ],
        ]
    )

def compute_height(position, imu_orientation):
    # perform the forward kinematics over the kinematic tree
    pin.forwardKinematics(robot.model, robot.data, position)
    # get IMU id
    imu_id = robot.model.getFrameId("imu")
    # get IMU translation
    imu_translation = robot.data.oMi[imu_id].translation.T
    # Get IMU rotation
    # print(imu_orientation)
    # imu_rotation = R.from_quat(imu_orientation).as_matrix()
    imu_rotation = rotation_matrix_from_quaternion(imu_orientation)

    heights = [(imu_rotation @ (oMi.translation.T - imu_translation))[2]
        for oMi in robot.data.oMi][1:]

    # for name, oMi in zip(robot.model.names, robot.data.oMi):
    #     print(("{:<24} : {: .2f} {: .2f} {: .2f}"
    #         .format( name, *oMi.translation.T.flat )))


    return (0 - heights[2])


if __name__ == "__main__":
    position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    imu = np.array([0.0, 0.0, 0.0, 1.0])
    height = compute_height(position, imu)
    print(height)
    