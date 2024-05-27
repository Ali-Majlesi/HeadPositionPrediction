import numpy as np

def euler_to_quaternion(yaw, pitch, roll):

    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = (np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2))
    qz = (np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2))
    qw = (np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2))

    q = np.stack([qx, qy, qz, qw],axis=1)

    for i in range(q.shape[0] - 1):
        # Calculate the sum of absolute differences for the current row
        sum_diff_subtract = np.sum(np.abs(q[i+1] - q[i]))
        sum_diff_add = np.sum(np.abs(q[i+1] + q[i]))

        # Determine the condition
        condition = sum_diff_subtract < sum_diff_add

        q[i+1] = np.where(condition, q[i+1], -q[i+1])

    return q

def quaternion_to_euler(qx, qy, qz, qw):
    phi = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
    a1 = 2*(qw*qy - qx*qz)
    theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1+a1), np.sqrt(1-a1))
    psi = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))
    return phi, theta, psi