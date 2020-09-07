import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import monoped
import actuation
from utils import plotOCSolution, plotConvergence, plot_frame_trajectory, animateMonoped
import conf

T = conf.T
dt = conf.dt

# MONOPED MODEL
# Create the monoped and actuator
monoped = monoped.createMonopedWrapper(nbJoint = conf.n_links)
import slice_model
monoped = slice_model.loadSoloLeg(solo8 = True)
robot_model = monoped.model
state = crocoddyl.StateMultibody(robot_model)
robot_model.effortLimit = 30 * np.ones(3)

# ACTUATION TYPE
# actuation = crocoddyl.ActuationModelFull(state)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# GRAVITY
# robot_model.gravity.linear = np.zeros(3)

# INITIAL CONFIGURATION
# to make the robot start with det(J) != 0 more options are given

q0 = np.zeros(1 + conf.n_links)

# OPTION 1 Select the angle of the first joint wrt vertical
# angle = np.pi/4
# q0[0] = 2 * np.cos(angle)
# q0[1] = np.pi - angle
# q0[2] = 2 * angle

# OPTION 2 Initial configuration distributing the joints in a semicircle with foot in O (scalable if n_joints > 2)
# q0[0] = 0.16 / np.sin(np.pi/(2 * conf.n_links))
# q0[1:] = np.pi/conf.n_links
# q0[1] = np.pi/2 + np.pi/(2 * conf.n_links)

# OPTION 3 Solo, (the convention used has negative displacements)
q0[0] = 0.16 / np.sin(np.pi/(2 * conf.n_links))
q0[1] = np.pi/4
q0[2] = -np.pi/2

x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# COSTS
# Create a cost model for the running and terminal action model
# Setting the final position goal with variable angle
# angle = np.pi/2
# s = np.sin(angle)
# c = np.cos(angle)
# R = np.matrix([ [c,  0, s],
#                 [0,  1, 0],
#                 [-s, 0, c]
#              ])
# target = np.array([np.sin(angle), 0, np.cos(angle)]))
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
target = np.array(conf.target)
footName = 'FL_FOOT'
footFrameID = robot_model.getFrameId(footName)
assert(robot_model.existFrame(footName))
Pref = crocoddyl.FrameTranslation(footFrameID,
                                target
                                )
# If also the orientation is useful for the task use
# Mref = crocoddyl.FramePlacement(footFrameID,
#                                pinocchio.SE3(R, conf.n_links * np.matrix([[np.sin(angle)], [0], [np.cos(angle)]])))
footTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)
Vref = crocoddyl.FrameMotion(footFrameID, pinocchio.Motion(np.zeros(6)))
footFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)
# simulating the cost on the power with a cost on the control
power_act =  crocoddyl.ActivationModelQuad(conf.n_links)
u2 = crocoddyl.CostModelControl(state, power_act, actuation.nu) # joule dissipation cost without friction, for benchmarking
stateAct = crocoddyl.ActivationModelWeightedQuad(np.concatenate([np.zeros(state.nq), np.ones(state.nv)]))
v2 = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)
bounds = crocoddyl.ActivationBounds(np.array([-1e3]*state.nx), np.array([0]*state.nx))
stateAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([np.ones(1), np.zeros(state.nx - 1)]))
nonPenetration = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)
# TODO see why not possible to set up the state regularization
# vel = crocoddyl.CostModelState(state, power_act, x0)

# CONTACT MODEL
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contact_location = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., 0.]))
supportContactModel = crocoddyl.ContactModel2D(state, contact_location, actuation.nu, np.array([0., 1/dt]))
# according to Andrea setting the damping to 1/dt makes the velocity drift disappear in one timestep
contactModel.addContact("foot_contact", supportContactModel)

# FRICTION CONE
# the friction cone can also have the [min, maximum] force parameters
# 4 is the number of faces for the approximation
mu = 0.7
normalDirection = np.array([0, 0, 1])
minForce = 0
maxForce = 200
cone = crocoddyl.FrictionCone(normalDirection, mu, 4, True, minForce, maxForce)
coneBounds = crocoddyl.ActivationBounds(cone.lb, cone.ub)
#coneActivation = crocoddyl.ActivationModelQuadraticBarrier(cone_bounds) # weighted quadratic barrier [0..2..]
# with this quadratic barrier we select just the first two columns of the cone friction approx in the x, z direction
coneActivation = crocoddyl.ActivationModelWeightedQuadraticBarrier(coneBounds, np.array([1, 1, 0, 0]))
frameFriction = crocoddyl.FrameFrictionCone(footFrameID, cone)
frictionCone = crocoddyl.CostModelContactFrictionCone(state,
        coneActivation,
        frameFriction,
        actuation.nu)

# Creating the action model for the KKT dynamics with simpletic Euler integration scheme
contactCostModel = crocoddyl.CostModelSum(state, actuation.nu)
contactCostModel.addCost('frictionCone', frictionCone, 1e-6)
contactCostModel.addCost("jouleDissipation", u2, 2e-5)
contactDifferentialModel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel,
        contactCostModel,
        0, # inv_damping
        True) # bool enable force
contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, dt)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("jouleDissipation", u2, 2e-5)
terminalCostModel.addCost("footPose", footTrackingCost, 1e2)
terminalCostModel.addCost("footVelocity", footFinalVelocity, 1e0)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

# Setting the nodes of the problem with a sliding variable
ratioContactTotal = 0.5
contactNodes = int(conf.T * ratioContactTotal)
flyingNodes = conf.T - contactNodes
problem_with_contact = crocoddyl.ShootingProblem(x0,
                                                [contactPhase] * contactNodes + [runningModel] * flyingNodes,
                                                terminalModel)
problem_without_contact = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


###  STANDING PROBLEM - keep the configuration still at the static value without beaking contact
if input('Solve standing problem [y/N]'):
        xref = crocoddyl.FrameTranslation(footFrameID, np.zeros(3))
        supportContactModel = crocoddyl.ContactModel2D(state, xref, actuation.nu, np.array([0., 0.]))
        contactModelStanding = crocoddyl.ContactModelMultiple(state, actuation.nu)
        contactModelStanding.addContact("foot_contact", supportContactModel)
        PrefStanding = crocoddyl.FramePlacement(1,
                                        pinocchio.SE3(
                                                np.identity(3),
                                                np.array([0, 0, q0[0]])))
        VrefStanding = crocoddyl.FrameMotion(1, pinocchio.Motion(np.zeros(6)))
        costModelStanding = crocoddyl.CostModelSum(state, actuation.nu)
        standingState = crocoddyl.CostModelState(state, x0, actuation.nu)
        costModelStanding.addCost('standing', standingState, 1)
        # standingCost = crocoddyl.CostModelFramePlacement(state, PrefStanding, actuation.nu)
        # standingVelCost = crocoddyl.CostModelFrameVelocity(state, VrefStanding, actuation.nu)
        standingControl = crocoddyl.CostModelControl(state, power_act, actuation.nu)
        # costModelStanding.addCost("basePose", standingCost, 1e3)
        # costModelStanding.addCost("baseVel", standingVelCost, 1e3)
        costModelStanding.addCost("control", standingControl, 1e-5)
        # costModelStanding.addCost('frictionCone', frictionCone, 1e-7)
        dmodelStanding = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
                actuation,
                contactModelStanding,
                costModelStanding,
                1e-12, # inv_damping for Cholesky decomposition of JMinvJt
                True) # bool enable computation of force jacobians
        runningModelStanding = crocoddyl.IntegratedActionModelEuler(dmodelStanding, dt)
        terminalModelStanding = crocoddyl.IntegratedActionModelEuler(dmodelStanding, 0.)
        problemStanding = crocoddyl.ShootingProblem(x0, [runningModelStanding] * T, terminalModelStanding)

        # Creating the DDP solver for this OC problem, defining a logger
        ddpStanding = crocoddyl.SolverFDDP(problemStanding)
        ddpStanding.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
        # Additionally also modify ddp.th_stop and ddp.th_grad
        ddpStanding.th_stop = 1e-9

        # Solving it with the DDP algorithm
        ddpStanding.solve([],[], maxiter = int(1e3))
        ddpStanding.robot_model = robot_model

        # SHOWING THE RESULTS
        plotOCSolution(ddpStanding)
        plotConvergence(ddpStanding)
        plot_frame_trajectory(ddpStanding, ['root_joint', 'FL_HFE', 'FL_KFE', 'FL_FOOT'], trid = False)
        animateMonoped(ddpStanding, frameNames=['FL_HFE', 'FL_KFE', 'FL_FOOT'])

        # QUASISTATIC
        data = runningModelStanding.createData()
        ddpStandingStatic = crocoddyl.SolverFDDP(problemStanding)
        ddpStandingStatic.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
        #xs = [x0] * (conf.T + 1)
        us = [runningModelStanding.quasiStatic(data, x0)] * conf.T
        ddpStandingStatic.solve([],[], maxiter = int(1e3))
        ddpStanding.robot_model = robot_model
        plotOCSolution(ddpStandingStatic)
        plotConvergence(ddpStandingStatic)


# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem_with_contact)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# Additionally also modify ddp.th_stop and ddp.th_grad
ddp.th_stop = 1e-9

# Solving it with the DDP algorithm
ddp.solve([],[], maxiter = int(1e3))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, ['FL_HFE', 'FL_KFE', 'FL_FOOT'], trid = False)
animateMonoped(ddp, saveAnimation=True)

# CHECK THE CONTACT FORCE FRICTION CONE CONDITION

r_data = robot_model.createData()
contactFrameID = robot_model.getFrameId(footName)

# METHOD 1
# Recovering the contact forces for the contact phase only
# Fx, Fz = list([] for _ in range(2))
# for i in range(int(conf.T*ratioContactTotal)):
#         tau = np.concatenate([np.zeros(1), ddp.us[i]])
#         q = ddp.xs[i][:robot_model.nq]
#         v = ddp.xs[i][robot_model.nq:]
#         pinocchio.computeAllTerms(robot_model, r_data, q, v)
#         J = pinocchio.getFrameJacobian(robot_model, r_data, contactFrameID, pinocchio.WORLD)
#         J_cont = J[[0,2],:]
#         pinocchio.forwardDynamics(robot_model, r_data, q, v, tau, J_cont, np.zeros(2)) # NOT correct since the gamma may be non zero!
#         Fx.append(r_data.lambda_c[0])
#         Fz.append(r_data.lambda_c[1])
# ratio = np.array(Fx)/np.array(Fz)

# METHOD 2, recover data from crocoddyl solution
Fx_, Fz_ = list([] for _ in range(2))
for i in range(int(conf.T*ratioContactTotal)):
        # convert the contact information to dictionary
        contactData = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['foot_contact']
        for force, vector in zip(contactData.f.linear, [Fx_, [], Fz_]):
                vector.append(force)
ratio = np.array(Fx_)/np.array(Fz_)
percentageContactViolation=len(ratio[ratio>mu])/contactNodes*100
assert((ratio<mu)).all(), 'The friction cone condition is violated for {:0.1f}% of the contact phase ({:0.3f}s)'.format(percentageContactViolation, len(ratio[ratio>mu])*conf.dt)
import matplotlib.pyplot as plt
Fz_clean=Fz_
Fz_clean.remove(max(Fz_))
plt.plot(Fz_clean)
plt.title('$F_z$')
plt.ylabel('[N]')
plt.show()

# TESTING THE SOLUTION
xs=ddp.xs
us=ddp.us
ddp2 = crocoddyl.SolverFDDP(problem_with_contact)
ddp2.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# Additionally also modify ddp.th_stop and ddp.th_grad
ddp2.th_stop = 1e-9

# Solving it with the FDDP algorithm
ddp2.solve(xs,us, maxiter = int(1e3))
ddp2.robot_model = robot_model
plotOCSolution(ddp2)
plotConvergence(ddp2)


# INITIALIZATION
data = runningModel.createData()
us = [runningModel.quasiStatic(data, x0)]* contactNodes + [np.zeros(conf.n_links)] * flyingNodes
xs = [x0] * (conf.T + 1)
ddp2 = crocoddyl.SolverFDDP(problem_with_contact)
ddp2.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
us = [runningModel.quasiStatic(data, x0)]* contactNodes + [np.zeros(conf.n_links)] * flyingNodes
ddp2.solve(xs,us, maxiter = int(1e3))
ddp2.robot_model = robot_model
plotOCSolution(ddp2)
plotConvergence(ddp2)
