import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import monoped
import actuation, slice_model
from utils import plotOCSolution, plotConvergence, plot_frame_trajectory, animateMonoped, plot_power
from power_costs import CostModelJointFriction, CostModelJointFrictionSmooth, CostModelJouleDissipation
import modify_model
import conf

T = conf.T
dt = conf.dt

# MONOPED MODEL
# Create the monoped and actuator
# monoped = monoped.createMonopedWrapper(nbJoint = conf.n_links, linkLength=0.16, floatingMass=0.37, linkMass=0.1)
# monoped = monoped.createSoloTBWrapper()
# import slice_model
monoped = slice_model.loadSoloLeg(solo8 = True)
robot_model = monoped.model
state = crocoddyl.StateMultibody(robot_model)
motor_mass, n_gear, lambda_l = [np.array([53e-3] * conf.n_links), np.array([9] * conf.n_links), np.array([1] * (conf.n_links + 1))]
modify_model.update_model(robot_model,  motor_mass, n_gear, lambda_l)
robot_model.effortLimit = 2.5 * np.ones(3)
#robot_model.T_mu = np.zeros(conf.n_links)
state.robot_model = robot_model

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
stateAct = crocoddyl.ActivationModelWeightedQuad(np.concatenate([np.zeros(state.nq + 1), np.ones(state.nv - 1)]))
v2 = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)
joint_friction = CostModelJointFrictionSmooth(state, power_act, actuation.nu)
joule_dissipation = CostModelJouleDissipation(state, power_act, actuation.nu)

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
# contactCostModel.addCost('frictionCone', frictionCone, 1e-6)
contactCostModel.addCost('joule_dissipation', joule_dissipation, 5e-3)
contactCostModel.addCost('joint_friction', joint_friction, 5e-3)
# contactCostModel.addCost('velocityRegularization', v2, 1e-1)
contactCostModel.addCost('nonPenetration', nonPenetration, 1e5)
contactDifferentialModel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel,
        contactCostModel,
        0, # inv_damping
        True) # bool enable force
contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, dt)

runningCostModel.addCost("joule_dissipation", joule_dissipation, 5e-3)
runningCostModel.addCost('joint_friction', joint_friction, 5e-3)
# runningCostModel.addCost("velocityRegularization", v2, 1e0)
runningCostModel.addCost("nonPenetration", nonPenetration, 1e6)
# runningCostModel.addCost("maxJump", maximizeJump, 1e2)
terminalCostModel.addCost("footPose", footTrackingCost, 5e3)
# terminalCostModel.addCost("footVelocity", footFinalVelocity, 1e0)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

runningModel.u_lb = -robot_model.effortLimit[-actuation.nu:]
runningModel.u_ub = robot_model.effortLimit[-actuation.nu:]

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
# Additionally also modify ddp.th_stop and ddp.th_grad
ddp.th_stop = 1e-9
ddp.solve([],[], maxiter = int(1e3))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, ['FL_HFE', 'FL_KFE', 'FL_FOOT'], trid = False)
animateMonoped(ddp, saveAnimation=False)
plot_power(ddp)

# CHECK THE CONTACT FORCE FRICTION CONE CONDITION

r_data = robot_model.createData()
contactFrameID = robot_model.getFrameId(footName)
Fx_, Fz_ = list([] for _ in range(2))
for i in range(int(conf.T*ratioContactTotal)):
        # convert the contact information to dictionary
        contactData = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['foot_contact']
        for force, vector in zip(contactData.f.linear, [Fx_, [], Fz_]):
                vector.append(force)
ratio = np.array(Fx_)/np.array(Fz_)
percentageContactViolation=len(ratio[ratio>cone.mu])/contactNodes*100
#assert((ratio<cone.mu)).all(), 'The friction cone condition is violated for {:0.1f}% of the contact phase ({:0.3f}s)'.format(percentageContactViolation, len(ratio[ratio>cone.mu])*conf.dt)
import matplotlib.pyplot as plt
Fz_clean=Fz_
Fz_clean.remove(max(Fz_))
plt.plot(Fz_clean)
plt.title('$F_z$')
plt.ylabel('[N]')
plt.show()

# POLISHING THE SOLUTION
# xs=ddp.xs
# us=ddp.us
# ddp2 = crocoddyl.SolverBoxFDDP(problem_with_contact)
# ddp2.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# ddp2.th_stop = 1e-6
# ddp2.solve(xs, us, maxiter = int(2e2))
# ddp2.robot_model = robot_model

# plotOCSolution(ddp2)
# plotConvergence(ddp2)
# plot_power(ddp2)
# animateMonoped(ddp2)

# CHECKING THE PARTIAL DERIVATIVES
# runningModel.differential.costs.removeCost('joule_dissipation')
mnd = crocoddyl.DifferentialActionModelNumDiff(runningModel.differential)
dnd = mnd.createData()
m=runningModel.differential
d=m.createData()
x=ddp.xs[3];u=ddp.us[3]

cm=m.costs.costs['joule_dissipation'].cost
#cm.gamma=1
#cm.T_mu=1
#cm.n[:]=1
#cm.K[:,:]=1

# CHECK THE GRADIENT  AND THE HESSIAN
n_joints = conf.n_links
x=.5*np.ones(2 * (n_joints + 1))
u=np.ones(n_joints)
m.calc(d,x,u)
mnd.calc(dnd,x,u)
m.calcDiff(d,x,u)
mnd.calcDiff(dnd,x,u)
lu=d.Lu.copy()
lx=d.Lx.copy()
m.costs.costs['joule_dissipation'].cost

eps=1e-6

# Lx, Lu
print(d.Lx-dnd.Lx)
print(d.Lu-dnd.Lu)
# Lux
m.calc(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))),u)
m.calcDiff(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))),u)
if n_joints > 1:
    print(np.vstack((np.zeros((n_joints + 2, n_joints)),np.diag((d.Lu-lu)/eps)))-d.Lxu)
else:
    print(np.hstack((0,(d.Lu-lu)/eps))-d.Lxu)
# Lxx
m.calc(d, x + np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))), u)
m.calcDiff(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))), u)
print(np.diag((d.Lx-lx)/eps)-d.Lxx)
# Luu
m.calc(d,x,u+eps*np.ones(n_joints))
m.calcDiff(d,x,u+eps*np.ones(n_joints))
print(np.diag((d.Lu-lu)/eps)-d.Luu)

def print_data(data):
    print('Lx', data.Lx)
    print('Lu', data.Lu)
    print('Lxx', data.Lxx)
    print('Lxu', data.Lxu)
    print('Luu', data.Luu)
    print('Fu', data.Fu)
    print('Fx', data.Fx)
    print('cost', data.cost)
