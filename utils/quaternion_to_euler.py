import numpy as np

def quaternion_to_euler(qx, qy, qz, qw):
    phi = np.arctan2(2*(qw*qx + qy*qz), 1-2*(qx**2 + qy**2))
    a1 = 2*(qw*qy - qx*qz)
    theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1+a1), np.sqrt(1-a1))
    psi = np.arctan2(2*(qw*qz + qx*qy), 1-2*(qy**2 + qz**2))
    return phi, theta, psi