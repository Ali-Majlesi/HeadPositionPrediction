import numpy as np
import cv2

def calculate_speed_acceleration(data_frame, dt, Q_position_block, R_position):
  transitionMatrix = np.array([[1, dt, 0.5*dt**2, 0, 0, 0, 0, 0, 0],
              [0, 1, dt, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, dt, 0.5*dt**2, 0, 0, 0],
              [0, 0, 0, 0, 1, dt, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, dt, 0.5*dt**2],
              [0, 0, 0, 0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

  observationMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0]], dtype=np.float32)

  Q = np.block([
      [Q_position_block, np.zeros((3, 3)), np.zeros((3, 3))],
      [np.zeros((3, 3)), Q_position_block, np.zeros((3, 3))],
      [np.zeros((3, 3)), np.zeros((3, 3)), Q_position_block]
  ]).astype(np.float32)

  R = np.eye(3, dtype=np.float32) * R_position # Example values

  num_states = 9
  num_observations = 3

  kalman = cv2.KalmanFilter(num_states, num_observations)
  kalman.transitionMatrix = transitionMatrix
  kalman.measurementMatrix = observationMatrix
  kalman.measurementNoiseCov = R
  kalman.processNoiseCov = Q

  columns =['x', 'y', 'z']
  estimated_stateval = []
  measurements = data_frame[columns].values
  prediction = np.zeros([9,1], dtype=np.float32)
  y_preds = []
  x0 = measurements[0]
  v0 = (measurements[6] - measurements[0])/(6*dt)
  a0 = (measurements[12] - 2*measurements[6] + measurements[0])/(2*(6*dt)**2)

  intial_states =  np.array([x0[0],v0[0],a0[0],x0[1],v0[1],a0[1],x0[2],v0[2],a0[2]], dtype=np.float32)
  prediction[:,0] = intial_states
  intial_states = prediction
  kalman.statePost = intial_states
  #kalman.statePre = intial_states*1.1

  for measurement in measurements[0:10]:
    measurement_matrix = measurement.reshape(-1, 1)
    kalman.correct(measurement_matrix.astype(np.float32))
    prediction = kalman.predict()

  for measurement in measurements:
          measurement_matrix = measurement.reshape(-1, 1)
          kalman.correct(measurement_matrix.astype(np.float32)) # Expand the measurement matrix
          estimated_stateval.append(prediction)
          prediction = kalman.predict()

            # Ensure correct type
  predicted = np.stack(estimated_stateval)
  data_frame.loc[:,'vx'] = predicted[:,1]
  data_frame.loc[:,'ax'] = predicted[:,2]
  data_frame.loc[:,'vy'] = predicted[:,4]
  data_frame.loc[:,'ay'] = predicted[:,5]
  data_frame.loc[:,'vz'] = predicted[:,7]
  data_frame.loc[:,'az'] = predicted[:,8]

  return data_frame


def calculate_angular_velocity(data_frame, dt, Q_angle_block, R_angle):
  transitionMatrix = np.array([[1, dt, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, dt, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, dt, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, dt],
              [0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

  observationMatrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0]], dtype=np.float32)

  Q = np.block([
      [Q_angle_block, np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))],
      [np.zeros((2, 2)), Q_angle_block,  np.zeros((2, 2)), np.zeros((2, 2))],
      [np.zeros((2, 2)), np.zeros((2, 2)), Q_angle_block, np.zeros((2, 2))],
      [np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), Q_angle_block],
  ]).astype(np.float32)

  R = np.eye(4, dtype=np.float32) * R_angle # Example values

  num_states = 8
  num_observations = 4

  kalman = cv2.KalmanFilter(num_states, num_observations)
  kalman.transitionMatrix = transitionMatrix
  kalman.measurementMatrix = observationMatrix
  kalman.measurementNoiseCov = R
  kalman.processNoiseCov = Q

  columns =['qx', 'qy', 'qz', 'qw']
  estimated_stateval = []
  measurements = data_frame[columns].values
  prediction = np.zeros([8,1], dtype=np.float32)
  y_preds = []
  x0 = measurements[0]
  v0 = (measurements[6] - measurements[0])/(6*dt)

  intial_states =  np.array([x0[0],v0[0],x0[1],v0[1],x0[2],v0[2],x0[3],v0[3]], dtype=np.float32)
  prediction[:,0] = intial_states
  intial_states = prediction
  kalman.statePost = intial_states
  #kalman.statePre = intial_states*1.1

  for measurement in measurements[0:10]:
    measurement_matrix = measurement.reshape(-1, 1)
    kalman.correct(measurement_matrix.astype(np.float32))
    prediction = kalman.predict()

  for measurement in measurements:
          measurement_matrix = measurement.reshape(-1, 1)
          kalman.correct(measurement_matrix.astype(np.float32)) # Expand the measurement matrix
          estimated_stateval.append(prediction)
          prediction = kalman.predict()

            # Ensure correct type
  predicted = np.stack(estimated_stateval)
  data_frame.loc[:,'dqx'] = predicted[:,1]
  data_frame.loc[:,'dqy'] = predicted[:,3]
  data_frame.loc[:,'dqz'] = predicted[:,5]
  data_frame.loc[:,'dqw'] = predicted[:,7]

  qw = data_frame['qw'].values
  qx = data_frame['qx'].values
  qy = data_frame['qy'].values
  qz = data_frame['qz'].values

  dqx = data_frame['dqx'].values
  dqy = data_frame['dqy'].values
  dqw = data_frame['dqw'].values
  dqz = data_frame['dqw'].values

  #w1 = 2*(dqy*(q1*q3-q2*q4) + dqx*(q1*q2+q3*q4) + dqw*(q1**2 + q4**2))/q4
  #w2 = 2*(dqy*(q2*q3+q1*q4) + dqw*(q1*q2-q3*q4) + dqx*(q2**2 + q4**2))/q4
  #w3 = 2*(dqx*(q2*q3-q1*q4) + dqw*(q1*q3+q2*q4) + dqy*(q3**2 + q4**2))/q4



  #dphi = w1 + np.sin(phi)*np.tan(theta)*w2 + np.cos(phi)*np.tan(theta)*w3
  #dtheta = np.cos(phi)*w2 - np.sin(phi)*w3
  #dpsi = np.sin(psi)/np.cos(theta)*w2 + np.cos(phi)/np.cos(theta)*w3

  a1 = 1-2*(qx**2+qy**2)
  a2 = qw*qx + qy*qz
  dphi = ((4*a2*(2*qx*dqx + 2*qy* dqy))/(a1**2) + (2*(qx*dqw + qw*dqx + qz*dqy + qy*qz))/a1)/(1 + (4*(a2)**2)/(a1)**2)
  dtheta = (2*np.sqrt ((1 + 2*qw* qy - 2 *qx *qz)/(1 - 2* qw* qy + 2 *qx*qz))*(qy*dqw - qz *dqx + qw *dqy - qx* dqz))/(1 + 2*qw* qy - 2*qx* qz)
  a3 = 1 - 2 *(qy ** 2 + qz ** 2)
  b3 = qx* qy + qw* qz
  dpsi = ((2 *(qz*dqw + qy* dqx + qx* dqy + qw* dqz))/a3 +
          (4*b3 *(2*qy*dqy + 2*qz*dqz ))/a3 ** 2)/(1 + (4 *b3 ** 2)/a3 ** 2)


  data_frame['dphi'] = dphi
  data_frame['dtheta'] = dtheta
  data_frame['dpsi'] = dpsi
  #data_frame = data_frame.assign(phi=phi, theta=theta, psi=psi, dphi=dphi, dtheta=dtheta, dpsi=dpsi)

  return data_frame