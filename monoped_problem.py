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
robot_model = monoped.model
state = crocoddyl.StateMultibody(robot_model)
robot_model.effortLimit = 15 * np.ones(3)

# ACTUATION TYPE
# actuation = crocoddyl.ActuationModelFull(state)
actuation = crocoddyl.ActuationModelFloatingBase(state)

# GRAVITY
# robot_model.gravity.linear = np.zeros(3)

# INITIAL CONFIGURATION
# to make the robot start with det(J) != 0
# Initial configuration distributing the joints in a semicircle with foot in O (scalable if n_joints > 2)
q0 = np.zeros(1 + conf.n_links)
q0[0] = 1 / np.sin(np.pi/(2 * conf.n_links))
q0[1:] = np.pi/conf.n_links
q0[1] = np.pi/2 + np.pi/(2 * conf.n_links)

x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# COSTS
# Create a cost model for the running and terminal action model
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
target = np.array(conf.target)
footFrameID = robot_model.getFrameId("foot")
Pref = crocoddyl.FrameTranslation(footFrameID,
                                target
                                )

footTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)
Vref = crocoddyl.FrameMotion(footFrameID, pinocchio.Motion(np.zeros(6)))
footFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)
# simulating the cost on the power with:
# a quadratic cost on the control
# a quadratic cost on velocities
power_act =  crocoddyl.ActivationModelQuad(conf.n_links)
u2 = crocoddyl.CostModelControl(state, power_act, actuation.nu) # joule dissipation cost without friction, for benchmarking
stateAct = crocoddyl.ActivationModelWeightedQuad(np.concatenate([np.zeros(state.nq), np.ones(state.nv)]))
v2 = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)

# PENALIZATIONS
bounds = crocoddyl.ActivationBounds(np.concatenate([np.zeros(1), -1e3* np.ones(state.nx-1)]), 1e3*np.ones(state.nx))
stateAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([np.ones(1), np.zeros(state.nx - 1)]))
nonPenetration = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)

# MAXIMIZATION
jumpBounds = crocoddyl.ActivationBounds(-1e3*np.ones(state.nx), np.concatenate([np.zeros(1), +1e3* np.ones(state.nx-1)]))
jumpAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([-np.ones(1), np.zeros(state.nx - 1)]))
maximizeJump = crocoddyl.CostModelState(state, jumpAct, np.ones(state.nx), actuation.nu)

# CONTACT MODEL
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contact_location = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., 0.]))
supportContactModel = crocoddyl.ContactModel2D(state, contact_location, actuation.nu, np.array([0., 1/dt])) # makes the velocity drift disappear in one timestep
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
coneActivation = crocoddyl.ActivationModelWeightedQuadraticBarrier(coneBounds, np.array([1, 1, 0, 0]))
frameFriction = crocoddyl.FrameFrictionCone(footFrameID, cone)
frictionCone = crocoddyl.CostModelContactFrictionCone(state,
        coneActivation,
        frameFriction,
        actuation.nu)

# Creating the action model for the KKT dynamics with simpletic Euler integration scheme
contactCostModel = crocoddyl.CostModelSum(state, actuation.nu)
contactCostModel.addCost('frictionCone', frictionCone, 1e-6)
contactCostModel.addCost("jouleDissipation", u2, 1e-2)
contactCostModel.addCost("velocityRegularization", v2, 2e0)
contactCostModel.addCost('nonPenetration', nonPenetration, 1e5)
contactDifferentialModel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel,
        contactCostModel,
        0, # inv_damping
        True) # bool enable force
contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, dt)

runningCostModel.addCost("jouleDissipation", u2, 1e-2)
runningCostModel.addCost("velocityRegularization", v2, 2e0)
runningCostModel.addCost("nonPenetration", nonPenetration, 1e5)
runningCostModel.addCost("maxJump", maximizeJump, 1e3)
terminalCostModel.addCost("footPose", footTrackingCost, 1e3)
# terminalCostModel.addCost("footVelocity", footFinalVelocity, 1e0)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

# Setting the nodes of the problem with a sliding variable
ratioContactTotal = 0.4/(conf.dt*T) # expressed as ratio in [s]
contactNodes = int(conf.T * ratioContactTotal)
flyingNodes = conf.T - contactNodes
problem_with_contact = crocoddyl.ShootingProblem(x0,
                                                [contactPhase] * contactNodes + [runningModel] * flyingNodes,
                                                terminalModel)
problem_without_contact = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# SOLVE
ddp = crocoddyl.SolverFDDP(problem_with_contact)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
ddp.th_stop = 1e-9
ddp.solve([],[], maxiter = int(1e4))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, [frame.name for frame in robot_model.frames[0:]], trid = False)
animateMonoped(ddp, saveAnimation=False)

# CHECK THE CONTACT FORCE FRICTION CONE CONDITION
# using directly crocoddyl TO RETRIEVE THE DATA
r_data = robot_model.createData()
contactFrameID = robot_model.getFrameId('foot')
Fx_, Fz_ = list([] for _ in range(2))
for i in range(int(conf.T*ratioContactTotal)):
        # convert the contact information to dictionary
        contactData = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['foot_contact']
        for force, vector in zip(contactData.f.linear, [Fx_, [], Fz_]):
                vector.append(force)
ratio = np.array(Fx_)/np.array(Fz_)
percentageContactViolation=len(ratio[ratio>cone.mu])/contactNodes*100
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
ddp2.th_stop = 1e-10
ddp2.solve(xs, us, maxiter = int(2e0))
ddp2.robot_model = robot_model