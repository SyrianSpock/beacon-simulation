from math import *
import numpy as np
import matplotlib.pyplot as plt

from beacon_ekf import BeaconEKF

BEACON_DISTANCE_NOISE_STDDEV = 0.03 # 3cm
SIMULATION_UPDATE_RATE = 100 # hz
SIMULATION_DURATION = 7 # second
SIMULATION_PLOT_ONCE_IN_X_ITERATION = 1

BEACON_1_POSITION = np.array([-1.5, 0, 0.35])
BEACON_2_POSITION = np.array([1.5, 1, 0.35])
BEACON_3_POSITION = np.array([1.5, -1, 0.35])

MEASUREMENT_COVARIANCE_MATRIX = (BEACON_DISTANCE_NOISE_STDDEV ** 2) * np.eye(3)

def distance_from_beacon(pos, beacon_pos):
    return np.linalg.norm(pos - beacon_pos) \
            + np.random.normal(0, BEACON_DISTANCE_NOISE_STDDEV)

def lemniscate_of_bernoulli(t):
    scale = 2.5 / (3 - cos(2*t))
    x = scale * cos(t)
    y = scale * sin(2*t) / 1.5

    return np.array([x, y])

def generate_full_trajectory():
    traj_t = np.linspace(0, SIMULATION_DURATION, SIMULATION_DURATION * SIMULATION_UPDATE_RATE)
    traj_x = []
    traj_y = []
    for i in traj_t:
        pos = lemniscate_of_bernoulli(i)
        traj_x.append(pos[0])
        traj_y.append(pos[1])

    return traj_x, traj_y

def figure_clear_and_update(fig, traj_x, traj_y):
    traj_, = fig.plot(traj_x, traj_y, color='red', label='trajectory')
    plt.legend([traj_], ['Trajectory'])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.0, 1.0)

    plt.draw()
    plt.pause(0.00001)
    fig.cla()

def plot_position(fig, pos, color):
    fig.plot(pos[0], pos[1], color=color, marker='o', linewidth=1.0)

def plot_beacon_distance_circle(fig, beacon_pos, beacon_dist):
    circle = plt.Circle(beacon_pos, beacon_dist, fill=False)
    fig.add_artist(circle)
    plot_position(fig, beacon_pos, 'black')

def plot_results(traj_x, traj_y, ekf_x, ekf_y):
    plt.figure()

    plt.plot(traj_x, traj_y, color='red')
    plt.plot(ekf_x, ekf_y, color='blue')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.0, 1.0)

    plt.show()

def plot_errors(traj_x, traj_y, ekf_x, ekf_y):
    err_x = np.array(ekf_x).reshape([700,]) - np.array(traj_x)
    err_y = np.array(ekf_y).reshape([700,]) - np.array(traj_y)

    plt.figure()
    plt.plot(err_x, color='red')
    plt.plot(err_y, color='blue')

    plt.figure()
    fig_x = plt.subplot(1, 2, 1)
    fig_x.hist(err_x, color='red')
    fig_y = plt.subplot(1, 2, 2)
    fig_y.hist(err_y, color='blue')

    fig_x.set_title('mean {:.6f}, dev {:.6f}'.format(np.mean(err_x), np.std(err_x)))
    fig_y.set_title('mean {:.6f}, dev {:.6f}'.format(np.mean(err_y), np.std(err_y)))

    plt.show()

def main():
    plt.figure()
    fig = plt.subplot(1, 1, 1)

    plot_flag = 0
    traj_x, traj_y = generate_full_trajectory()
    ekf_x, ekf_y = [], []

    kalman = BeaconEKF(BEACON_1_POSITION, BEACON_2_POSITION, BEACON_3_POSITION,
                       1 / SIMULATION_UPDATE_RATE)
    kalman.reset(np.array([1.25, 0, 0.35, 0, 0, 0]).reshape([6,1]),
                 np.square(np.diag([1, 1, 1, 1e1, 1e1, 1e1])))

    for i in range(int(SIMULATION_DURATION * SIMULATION_UPDATE_RATE)):
        # Get real position and plot it
        robot_position = np.array([traj_x[i], traj_y[i], 0.35])

        if plot_flag == SIMULATION_PLOT_ONCE_IN_X_ITERATION:
            plot_position(fig, robot_position, 'black')

        # Get beacon distances
        d1 = distance_from_beacon(robot_position, BEACON_1_POSITION)
        d2 = distance_from_beacon(robot_position, BEACON_2_POSITION)
        d3 = distance_from_beacon(robot_position, BEACON_3_POSITION)

        # Update Kalman filter
        kalman.predict(0)
        kalman.measure(np.array([d1, d2, d3]).reshape([3,1]), MEASUREMENT_COVARIANCE_MATRIX)
        ekf_x.append(kalman.x[0])
        ekf_y.append(kalman.x[1])

        if plot_flag == SIMULATION_PLOT_ONCE_IN_X_ITERATION:
            plot_position(fig, kalman.x[0:2], 'green')

        # Update plot
        if plot_flag == SIMULATION_PLOT_ONCE_IN_X_ITERATION:
            figure_clear_and_update(fig, traj_x, traj_y)
            plot_flag = 0
        else:
            plot_flag += 1

    plot_results(traj_x, traj_y, ekf_x, ekf_y)
    plot_errors(traj_x, traj_y, ekf_x, ekf_y)

if __name__ == '__main__':
    main()
