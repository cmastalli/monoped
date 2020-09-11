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

def animateMonoped(ddp, saveAnimation=False, frameNames = None):
    anim = plt.figure()
    robot_data = ddp.robot_model.createData()
    if frameNames is None:
        frameNames = [frame.name for frame in ddp.robot_model.frames]
        frameNames.remove('universe')
    img = []

    for i in ddp.xs:
        X = []
        Z = []
        pinocchio.updateFramePlacements(ddp.robot_model, robot_data)
        pinocchio.forwardKinematics(ddp.robot_model, robot_data, i[:ddp.robot_model.nq], i[ddp.robot_model.nq:])
        for frame_name in frameNames:
            frame_id = ddp.robot_model.getFrameId(frame_name)
            X.append(robot_data.oMf[frame_id].translation[0])
            Z.append(robot_data.oMf[frame_id].translation[2])
        img.append(plt.plot(X,Z, color='grey', marker='o', linewidth=2, markerfacecolor='black'))

    import matplotlib.animation as animation
    im_ani = animation.ArtistAnimation(anim, img, interval=conf.dt*conf.T, repeat_delay=1000,
                                   blit=True)
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.scatter(conf.target[0], conf.target[2], marker = 'x', color = 'red')
    plt.plot([0,0],[-2,2.5], ls='--', color='blue')
    plt.title('Monoped task')
    plt.xlim([-.3, .3])
    plt.ylim([0, 1])
    if saveAnimation:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(1/conf.dt), metadata=dict(artist='G. Fadini'), bitrate=-1)
        im_ani.save('task_animation.mp4', writer=writer)
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

def append_cost_to_vector(data, collector, tag):
    try:
        if len(data.differential.tolist()) > 1:
            # RK4 integrator, weight the cost
            collector.append(
                    np.sum(
                        np.array([d_.costs.costs[tag].cost for d_ in data.differential]) * np.array([1/3, 1/6, 1/6, 1/3])
                        )
                    )
    except:
        try:
            # Euler case
            collector.append(data.differential.costs.costs[tag].cost)
        except:
            collector.append(0)

def cost_stats(ddp):
    '''
    Takes the costs from the running datas and returns them as arrays.
    If the cost is not there, returns an empty.
    '''
    ddp_data = ddp.problem.runningDatas

    u_cost, pf_cost, pm_cost, pt_cost =  list([] for _ in range(4))

    for data in ddp_data:
        append_cost_to_vector(data, u_cost, 'control_bound')
        append_cost_to_vector(data, pf_cost, 'joint_friction')
        append_cost_to_vector(data, pm_cost, 'mech_power')
        append_cost_to_vector(data, pt_cost, 'joule_dissipation')

    return np.array(u_cost), np.array(pf_cost), np.array(pm_cost), np.array(pt_cost)

def energy_stats(ddp, pm, pt_cost, pf_cost):
    '''
    Computes the energy required by the motion from two positions with zero initial and final velocities TODO add kinetic energy
    Compares the result with crocoddyl mechanical power consumptions and dissipation
    '''
    import conf
    pin_data = ddp.robot_model.createData()
    q0 = ddp.xs[0][:ddp.robot_model.nq]
    qf = ddp.xs[-1][:ddp.robot_model.nq]
    v0 = ddp.xs[0][ddp.robot_model.nq:]
    vf = ddp.xs[-1][ddp.robot_model.nq:]
    idealPotential = pinocchio.computePotentialEnergy(ddp.robot_model, pin_data, qf) - pinocchio.computePotentialEnergy(ddp.robot_model, pin_data, q0)
    idealKinetic = pinocchio.computeKineticEnergy(ddp.robot_model, pin_data, qf, vf) - pinocchio.computeKineticEnergy(ddp.robot_model, pin_data, q0, v0)
    ideal = idealPotential + idealKinetic
    mechanical = np.sum(pm)*conf.dt
    print('Mechanical energy: {:0} J, ideal: {:1}, error: {:2.2} %'.format(mechanical, ideal, (mechanical-ideal)/ideal * 1e2))
    print('Thermal dissipation: {:0} J'.format(np.sum(pt_cost)*conf.dt))
    print('Friction dissipation: {:0} J'.format(np.sum(pf_cost)*conf.dt))
    print('Total Energy needed: {:0} J'.format(np.sum(pt_cost)*conf.dt + np.sum(pf_cost)*conf.dt + mechanical))

def plot_power(ddp, image_folder = None, extension = 'pdf'):
    '''
    Given a already solved ddp problem, plot the various power components in time
    It also prints a summary of the energetic expenditure to the terminal
    '''

    u_cost, pf_cost, pm_cost, pt_cost = cost_stats(ddp)
    pm = []
    T = np.arange(start=0, stop=conf.dt*(conf.T), step=conf.dt)
    for i, torque in enumerate(ddp.us):
        pm.append(np.sum(torque * ddp.xs[i][-ddp.problem.nu_max:]))
    pm =  np.array(pm)
    pe = pt_cost + pf_cost + pm
    energy_stats(ddp, pm, pt_cost, pf_cost)
    fig_title = 'energy_comparison'
    plt.figure(fig_title)
    plt.plot(T, pm, color ='blue')
    plt.plot(T, pf_cost, color = 'magenta')
    plt.plot(T, pt_cost, color = 'red')
    plt.plot(T, pe, color = 'green')
    plt.ylabel('[W]')
    plt.title('Power components')
    plt.legend(['$P_m$', '$P_f$', '$P_t$', '$P_{el}$'])
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.grid(True)
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

def plot_frame_trajectory(ddp, frame_names, image_folder = None, extension = 'pdf', trid = True):
    '''
    Plots a specific or multiple frame trajectory in time, 2D or 3D
    '''
    fig_title = 'foot_reference'
    plt.figure('Foot_reference_frame_traj')
    initial_positions = np.array([])
    final_positions = np.array([])
    if trid:
        ax = plt.axes(projection = '3d')
        for frame_name in frame_names:
            x, y, z = frame_position(ddp, frame_name)
            ax.plot3D(x[1:], y[1:], z[1:])
            ax.scatter(x[1], y[1], z[1], color = 'black')
            ax.scatter(x[-1], y[-1], z[-1], marker = '*', color = 'green')
        ax.scatter(*conf.target, marker = 'X', color = 'red')
        # Make axes limits
        xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()]).T
        XYZlim = [min(xyzlim[0]), max(xyzlim[1])]
        ax.set_xlim3d(XYZlim)
        ax.set_ylim3d(XYZlim)
        ax.set_zlim3d(XYZlim)
        plt.legend(frame_names)
        try:
            ax.set_aspect('equal')
        except NotImplementedError:
            pass
    else:
        ax = plt.axes()
        for frame_name in frame_names[1:]:
            x, y, z = frame_position(ddp, frame_name)
            ax.plot(x[1:], z[1:])
            initial_positions = np.append(initial_positions, np.array([x[1], z[1]]))
            final_positions = np.append(final_positions, np.array([x[-1], z[-1]]))
        ax.scatter(conf.target[0], conf.target[2], marker = 'x', color = 'red')
        ax.scatter(initial_positions[0::2], initial_positions[1::2], color = 'black')
        ax.scatter(final_positions[0::2], final_positions[1::2], color = 'blue') 
        ax.set_aspect('equal')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        ax.legend(frame_names[1:] + ['target', 'initial', 'final'], loc='center right', bbox_to_anchor=(1.6, 0.5), fancybox=True, shadow=True)
        ax.plot(initial_positions[0::2], initial_positions[1::2], color = 'grey', alpha = 0.7)
        ax.plot(final_positions[0::2], final_positions[1::2], color = 'grey', alpha = 0.7)

    plt.title('Monoped Trajectory')
    plt.xlabel('x [m]')
    if trid:
        plt.ylabel('y [m]')
    else:
        plt.ylabel('z [m]')
    plt.grid(True)
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()