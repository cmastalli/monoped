import numpy as np
import pinocchio
import crocoddyl
import time
import os
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import conf

def plotConfig(q, animation, lengths):
    O=np.array([0,0,q[0]])
    points=[O]
    for i, qi in enumerate(q[1:]):
        p = points[-1] + lengths[i] * [np.sin(qi), np.cos(qi), 0]
        points.append(p)
    animation.plot(points)

def plotMonoped(ddp, scaling=1):
    xs = np.array(i for i in ddp.xs)
    lenghts = scaling * np.ones(len(xs[0]))
    plt.figure('monoped_animation')
    animation=plt.plot()
    for q in xs:
        plotConfig(q, animation, lenghts)
        plt.show()

def actuated_joints_id(model, actuated_rf_labels):
    '''
    Returns the id of a specific joint
    '''
    rf_id = []
    for label in actuated_rf_labels:
        if model.existFrame(label):
            rf_id.append(model.getFrameId(label))
        else:
            print(label + ' not found in model')
    return rf_id

def extract(npzfile, tag, index=0):
    '''
    Function used to extract a specific component of the saved data
    it handles the exception in which the index is an integer
    '''
    tmp_array = []
    for i in npzfile[tag]:
        try:
            tmp_array.append(i[index])
        except:
            tmp_array.append(i)
    return np.array(tmp_array)

def plotOCSolution(ddp, image_folder = None, extension = 'pdf', fig_title='solution'):
    '''
    Plots the ddp solution, xs, us
    '''
    log = ddp.getCallbacks()[0]
    xs, us = log.xs, log.us

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = xs[0].shape[0]
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = us[0].shape[0]
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212

    T = np.arange(start=0, stop=conf.dt*(conf.T) + conf.dt, step=conf.dt)

    plt.figure(fig_title)

    # Plotting the state trajectories
    if xs is not None:
        plt.title('Solution trajectory')
        plt.subplot(xsPlotIdx)
        [plt.plot(T, X[i], label="$x_{" + str(i) + '}$') for i in range(nx)]
        plt.legend()
    plt.grid(True)

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(T[:conf.T], U[i], label="$u_{" + str(i) + '}$') for i in range(nu)]
        plt.legend()
        #plt.title('Control trajectory')
        plt.xlabel("time [s]")
    plt.grid(True)
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()


def plotConvergence(ddp, image_folder = None, extension = 'pdf', fig_title="convergence"):
    '''
    Plots the ddp callbacks
    '''
    log = ddp.getCallbacks()[0]
    costs, muLM, muV, gamma, theta, alpha = log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps

    plt.figure(fig_title)

    # Plotting the total cost sequence
    plt.title(fig_title)
    plt.subplot(511)
    plt.ylabel("Cost")
    plt.plot(costs)

    # Ploting mu sequences
    plt.subplot(512)
    plt.ylabel("$\mu$")
    plt.plot(muLM, label="LM")
    plt.plot(muV, label="V")
    plt.legend()

    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(513)
    plt.ylabel("$\gamma$")
    plt.plot(gamma)
    plt.subplot(514)
    plt.ylabel("$\\theta$")
    plt.plot(theta)

    # Plotting the alpha sequence
    plt.subplot(515)
    plt.ylabel("$\\alpha$")
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel("Iteration")
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

def frame_position(ddp, frame_name):
    '''
    Returns the position of a frame for a given configuration
    '''
    robot_data = ddp.robot_model.createData()
    frame_id = ddp.robot_model.getFrameId(frame_name)
    x = []
    y = []
    z = []

    for i in ddp.xs:
        pinocchio.updateFramePlacements(ddp.robot_model, robot_data)
        pinocchio.forwardKinematics(ddp.robot_model, robot_data, i[:ddp.robot_model.nq], i[ddp.robot_model.nq:])
        # changed for pinocchio array
        x.append(robot_data.oMf[frame_id].translation[0])
        y.append(robot_data.oMf[frame_id].translation[1])
        z.append(robot_data.oMf[frame_id].translation[2])
    return x, y, z

def plot_frame_trajectory(ddp, frame_name, image_folder = None, extension = 'pdf'):
    '''
    Plots a specific frame trajectory in time
    '''
    x, y, z = frame_position(ddp, frame_name)

    fig_title = 'foot_reference'
    plt.figure('Foot_reference_frame_traj')
    ax = plt.axes(projection = '3d')
    ax.scatter(x[1], y[1], z[1], color = 'black')
    ax.scatter(x[-1], y[-1], z[-1], marker = '*', color = 'green')
    ax.plot3D(x[1:], y[1:], z[1:], 'red')
    plt.title('Foot trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)

    # Make axes limits
    xyzlim = np.array([ax.get_xlim3d(),ax.get_ylim3d(),ax.get_zlim3d()]).T
    XYZlim = [min(xyzlim[0]),max(xyzlim[1])]
    ax.set_xlim3d(XYZlim)
    ax.set_ylim3d(XYZlim)
    ax.set_zlim3d(XYZlim)
    try:
        ax.set_aspect('equal')
    except NotImplementedError:
        pass

    plt.show()
