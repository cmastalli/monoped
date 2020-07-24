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

def plot_codesign_results(npzfile, image_folder = None, extension = 'pdf'):

    fig_title = 'cost_mass'
    plt.figure(fig_title)
    x, y = extract(npzfile, 'motor_mass'), np.log(extract(npzfile, 'cost'))
    plt.scatter(x, y, color = 'blue', s=5**2)
    plt.title('Cost and motor mass')
    plt.xlabel('$m_m$')
    plt.ylabel('Natural log cost value')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'cost_transmission'
    plt.figure('cost_transmission')
    x, y = extract(npzfile, 'n_gear'), np.log(extract(npzfile, 'cost'))
    plt.scatter(x, y, color = 'red', s=5**2)
    plt.title('Cost and transmission')
    plt.xlabel('$n$')
    plt.ylabel('Natural log cost value')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'cost_scale'
    plt.figure('Cost_scale')
    x, y = extract(npzfile, 'lambda_l'), np.log(extract(npzfile, 'cost'))
    plt.scatter(x, y, color = 'orange', s=5**2)
    plt.title('Cost and scaling')
    plt.xlabel('$\\lambda_l$')
    plt.ylabel('Natural log cost value')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'cost_evo'
    plt.figure('Cost_evo')
    y = np.log(extract(npzfile, 'cost'))
    plt.plot(y, marker='o', linestyle='none', markersize = 2, color = 'blue')
    plt.title('Cost evolution during the optimization')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Natural log cost value')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'error_evo'
    plt.figure('Error evo')
    y = np.log(extract(npzfile, 'error'))
    plt.plot(y, marker='o', linestyle='none', markersize = 2, color = 'red')
    plt.title('Error evolution during the optimization')
    plt.xlabel('Number of Iteration')
    plt.ylabel('Natural log error to reference')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'motor_lambda'
    plt.figure(fig_title)
    x, y, z = extract(npzfile, 'lambda_l'), extract(npzfile, 'motor_mass'), np.log(extract(npzfile, 'cost'))
    # condition = z<=0
    # z = np.extract(condition, z)
    # x = np.extract(condition, x)
    # y = np.extract(condition, y)
    scatter = plt.scatter(x, y,  c = z, cmap = 'BuPu_r', s=5**2)
    plt.colorbar(scatter)
    plt.title('Motor mass and scaling')
    plt.xlabel('$\\lambda_l$')
    plt.ylabel('$m_m$')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=np.min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'motor_transmission'
    plt.figure(fig_title)
    x, y, z = extract(npzfile, 'n_gear'), extract(npzfile, 'motor_mass'), np.log(extract(npzfile, 'cost'))
    # condition = z<=0
    # z = np.extract(condition, z)
    # x = np.extract(condition, x)
    # y = np.extract(condition, y)
    scatter = plt.scatter(x, y,  c = z, cmap = 'BuPu_r', s=5**2)
    plt.colorbar(scatter)
    plt.title('Motor mass and transmission')
    plt.xlabel('$n$')
    plt.ylabel('$m_m$')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=np.min(y))
    #plt.colorbar()
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = 'transmission_lambda'
    plt.figure(fig_title)
    x, y, z = extract(npzfile, 'lambda_l'), extract(npzfile, 'n_gear'), np.log(extract(npzfile, 'cost'))
    # condition = z<=0
    # z = np.extract(condition, z)
    # x = np.extract(condition, x)
    # y = np.extract(condition, y)
    scatter = plt.scatter(x, y,  c = z, cmap = 'BuPu_r', s=5**2)
    plt.colorbar(scatter)
    plt.title('Motor mass and scaling')
    plt.xlabel('$\\lambda_l$')
    plt.ylabel('$n$')
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

    fig_title = '3d_plot1'
    plt.figure(fig_title)
    x, y, z = extract(npzfile, 'lambda_l'), extract(npzfile, 'n_gear'), np.log(extract(npzfile, 'cost'))
    # condition = z<=0
    # z = np.extract(condition, z)
    # x = np.extract(condition, x)
    # y = np.extract(condition, y)
    ax = plt.axes(projection = '3d')
    ax.scatter(x, y, z, c = z, cmap = 'BuPu_r', s=5**2)
    ax.set_zlim3d(np.min(z),np.max(z))
    plt.colorbar(
                    matplotlib.cm.ScalarMappable(
                        norm=matplotlib.colors.Normalize(
                                        np.min(z), np.max(z), clip=True), cmap='BuPu_r'),
                        ax=ax
                )
    plt.title(fig_title)
    ax.set_xlabel('$\\lambda_l$')
    ax.set_ylabel('$n$')
    ax.set_zlabel('cost', linespacing=3.4)
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    def rotate(angle):
        ax.view_init(30, angle)
        plt.draw()
    N = 100
    from matplotlib import animation
    ani = animation.FuncAnimation(plt.figure(fig_title), rotate, N, interval=360/N, blit=False)
    if image_folder is not None:
        ani.save(image_folder + fig_title + '.gif', writer='imagemagick', progress_callback = lambda i, n: print('Saving frame {i} of {n}'))
        animation_js = ani.to_jshtml()
        js_file=open(image_folder + fig_title + '.html', "w")
        js_file.write(animation_js)
        js_file.close()
    plt.show()

    fig_title = '3d_plot2'
    plt.figure(fig_title)
    x, y, z = extract(npzfile, 'motor_mass'), extract(npzfile, 'n_gear'), np.log(extract(npzfile, 'cost'))
    # condition = z<=0
    # z = np.extract(condition, z)
    # x = np.extract(condition, x)
    # y = np.extract(condition, y)
    ax = plt.axes(projection = '3d')
    ax.scatter(x, y, z, c = z, cmap = 'BuPu_r', s=5**2)
    ax.set_zlim3d(np.min(z),np.max(z))
    plt.colorbar(
                    matplotlib.cm.ScalarMappable(
                        norm=matplotlib.colors.Normalize(
                                        np.min(z), np.max(z), clip=True), cmap='BuPu_r'),
                        ax=ax
                )
    plt.title(fig_title)
    ax.set_xlabel('$m_m$', linespacing=3.4)
    ax.set_ylabel('$n$', linespacing=3.4)
    ax.set_zlabel('cost', linespacing=3.4)
    # if sum(element > 0 for element in y): plt.ylim(top=0)
    plt.ylim(bottom=min(y))
    if image_folder is not None:
        ani.save(image_folder + fig_title + '.gif', writer='imagemagick', progress_callback = lambda i, n: print(f'Saving frame {i} of {n}'))
        animation_js = ani.to_jshtml()
        js_file=open(image_folder + fig_title + '.html', "w")
        js_file.write(animation_js)
        js_file.close()
    plt.show()

def cost_stats(ddp):
    '''
    Takes the costs from the running datas and returns them as arrays.
    If the cost is not there, returns an empty.
    '''
    running_data = ddp.problem.runningDatas
    terminal_data = ddp.problem.terminalData
    ddp_data = running_data

    u_cost, pf_cost, pm_cost, pt_cost, pt_cost =  list([] for _ in range(5))

    for data in ddp_data:
        try:
            u_cost.append(data.differential.costs.costs['control_bound'].cost)
        except:
            u_cost.append(0)
        try:
            pf_cost.append(data.differential.costs.costs['joint_friction'].cost)
        except:
            pf_cost.append(0)
        try:
            pm_cost.append(data.differential.costs.costs['mech_power'].cost)
        except:
            pm_cost.append(0)
        try:
            pt_cost.append(data.differential.costs.costs['joule_dissipation'].cost)
        except:
            pt_cost.append(0)

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
    ideal = pinocchio.computePotentialEnergy(ddp.robot_model, pin_data, qf) - pinocchio.computePotentialEnergy(ddp.robot_model, pin_data, q0)
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
    # changed for pinocchio array
    S = np.zeros( (ddp.robot_model.nv, ddp.robot_model.nq + ddp.robot_model.nv) )
    S[0:, ddp.robot_model.nq:] = np.identity(ddp.robot_model.nv)
    for i, torque in enumerate(ddp.us):
        pm.append(np.asscalar(np.matrix(torque) * S @ ddp.xs[i]))
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

def visualize_movement(ddp, n_times = 3):
    '''
    Visualizing the solution in gepetto-viewer TODO see if works with the GV fix
    '''
    for _ in range(n_times):
        ddp.display.displayFromSolver(ddp)
        time.sleep(2)

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

def plotPhaseSpace(ddp, image_folder = None, extension = 'pdf', fig_title='phase_plot'):
    '''
    Phase space plot for a 1DOF pendulum case
    '''
    log = ddp.getCallbacks()[0]
    xs = log.xs
    nq = ddp.problem.runningModels[0].state.nq
    nv = ddp.problem.runningModels[0].state.nv
    #nx = ddp.models()[0].state.nx

    if nq != nv:
        print('Cannot find a proper state space representation')
    else:
        # Plotting the state trajectories
        if xs is not None:
            q = list([] for _ in range(nq))
            qdot = list([] for _ in range(nv))
            for timeframe in xs:
                for j, val in enumerate(timeframe):
                    if j < nq:
                        q[j].append(np.asscalar(val))
                    else:
                        qdot[j - nq].append(np.asscalar(val))

    plt.figure(fig_title)
    for index in range(nq):
        plt.title(fig_title)
        plt.plot(q[index], qdot[index])
        plt.xlabel('$q_' + str(index) + '$')
        plt.ylabel('$\dot{q}_' + str(index) + '$')
    plt.grid(True)
    add_pendulum_plot(fig_title)
    plt.gca().set_aspect('equal', adjustable='box')
    if image_folder is not None:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        plt.savefig(image_folder + fig_title + '.' + extension, format = extension)
    plt.show()

def plot_solution(ddp, image_folder = None, extension = 'pdf'):
    '''
    Plotting the solution and the DDP convergence
    '''
    plotOCSolution(ddp, image_folder, extension)
    plotConvergence(
                    ddp,
                    image_folder,
                    extension)

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

    fig_title = 'gripper_reference'
    plt.figure('Gripper_reference_frame_traj')
    ax = plt.axes(projection = '3d')
    ax.scatter(x[1], y[1], z[1], 'red')
    ax.plot3D(x[1:], y[1:], z[1:], 'red')
    plt.title('Gripper trajectory')
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

def acc(x):
    return np.sin(x)*1*9.81/2

def pendulum_dyn(state, dt = 1e-3):
    a = acc(state[0][-1])
    v = a*dt + state[1][-1]
    state[1].append(v)
    state[0].append(state[0][-1] + v*dt)

def plot_trajectory(states, name = 'figure', color = 'grey'):
    plt.figure(name)
    plt.plot(states[0], states[1], color = color, linewidth = .5, linestyle=':')
    plt.plot(np.array(states[0]), - np.array(states[1]), color = color, linewidth = .5, linestyle=':')

def add_pendulum_plot(figure):
    for vel in range(-2, 3):
        for pos in range(-2, 3):
            states = [[np.pi*(pos/2)], [vel]]
            # print(states)
            for _ in range(int(2e3)):
                pendulum_dyn(states)
            plot_trajectory(states, figure)
    # adding the separatrix
    states = [[np.pi], [-np.sqrt(9.812 * 2)]]
    for _ in range(int(3e3)):
        pendulum_dyn(states)
    plot_trajectory(states, figure, color = 'grey')
    states = [[-np.pi], [np.sqrt(9.812 * 2)]]
    for _ in range(int(3e3)):
        pendulum_dyn(states)
    plot_trajectory(states, figure, color = 'grey')
