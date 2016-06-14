from math import *
import numpy as np
import matplotlib.pyplot as plt

SIMULATION_UPDATE_RATE = 100 # hz
SIMULATION_DURATION = 7 # second
SIMULATION_PLOT_ONCE_IN_X_ITERATION = 10

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
    fig.plot(traj_x, traj_y, color='red')
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.0, 1.0)

    plt.draw()
    plt.pause(0.00001)
    fig.cla()

def plot_position(fig, pos, color):
    fig.plot(pos[0], pos[1], color=color, marker='o', linewidth=1.0)


def main():
    plt.figure()
    fig = plt.subplot(1, 1, 1)

    plot_flag = 0
    traj_x, traj_y = generate_full_trajectory()

    for i in range(int(SIMULATION_DURATION * SIMULATION_UPDATE_RATE)):
        # Get real position and plot it
        robot_position = np.array([traj_x[i], traj_y[i]])

        if plot_flag == SIMULATION_PLOT_ONCE_IN_X_ITERATION:
            plot_position(fig, robot_position, 'black')

        if plot_flag == SIMULATION_PLOT_ONCE_IN_X_ITERATION:
            figure_clear_and_update(fig, traj_x, traj_y)
            plot_flag = 0
        else:
            plot_flag += 1


if __name__ == '__main__':
    main()

