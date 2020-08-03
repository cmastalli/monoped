import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import monoped
import actuation
from utils import plotOCSolution, plotConvergence, plot_frame_trajectory
import conf

T = conf.T
dt = conf.dt
# Create the monoped and actuator
monoped = monoped.createMonopedWrapper(nbJoint = conf.n_links)
robot_model = monoped.model
# setting gravity to 0 if needed
robot_model.gravity.linear=np.zeros(3)

# Create a cost model per the running and terminal action model
state = crocoddyl.StateMultibody(robot_model)
# actuation = crocoddyl.ActuationModelFull(state)
actuation = crocoddyl.ActuationModelFloatingBase(state)
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

# Initial configuration to make the robot start with det(J) != 0
angle = np.pi/4
q0 = np.zeros(1 + conf.n_links)
q0[0] = 2 * np.cos(angle)
q0[1] = np.pi - angle
q0[2] = 2 * angle
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# Setting the final position goal with variable angle
# angle = np.pi/2
# s = np.sin(angle)
# c = np.cos(angle)
# R = np.matrix([ [c,  0, s],
#                 [0,  1, 0],
#                 [-s,  0, c]
#              ])
# target = np.array([np.sin(angle), 0, np.cos(angle)]))

target = np.array(conf.target)
footFrameID = robot_model.getFrameId("foot")
Pref = crocoddyl.FrameTranslation(footFrameID,
                                target
                                )
# If also the orientation is useful for the task use
# Mref = crocoddyl.FramePlacement(footFrameID,
#                                pinocchio.SE3(R, conf.n_links * np.matrix([[np.sin(angle)], [0], [np.cos(angle)]])))
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)

Vref = crocoddyl.FrameMotion(footFrameID, pinocchio.Motion(np.zeros(6)))
goalFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)

# simulating the cost on the power with a cost on the control
power_act =  crocoddyl.ActivationModelQuad(conf.n_links)
u2 = crocoddyl.CostModelControl(state, power_act, actuation.nu) # joule dissipation cost without friction, for benchmarking

# CONTACT MODEL
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contact_location = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., 0.]))
supportContactModel = crocoddyl.ContactModel2D(state, contact_location, actuation.nu, np.array([0., 50.]))
contactModel.addContact("foot_contact", supportContactModel)

# FRICTION CONE
# the friction cone can also have the [min, maximum] force parameters
# 4 is the number of faces for teh approximation
mu = 0.5
normalDirection = np.array([0, 0, 1])
cone = crocoddyl.FrictionCone(normalDirection, mu, 4, True)
coneBounds = crocoddyl.ActivationBounds(cone.lb, cone.ub)
#coneActivation = crocoddyl.ActivationModelQuadraticBarrier(cone_bounds) # weightred quadratic barrier [0..2..]
# with this quadratic barrier we select just the first two columns of the cone friction approx in the x, z direction
coneActivation = crocoddyl.ActivationModelWeightedQuadraticBarrier(coneBounds, np.array([1, 1, 0, 0])) 
frameFriction = crocoddyl.FrameFrictionCone(footFrameID, cone)
frictionCone = crocoddyl.CostModelContactFrictionCone(state,
        coneActivation,
        frameFriction,
        actuation.nu)

# Creating the action model for the KKT dynamics with simpletic Euler integration scheme
contactCostModel = crocoddyl.CostModelSum(state, actuation.nu)
contactCostModel.addCost('frictionCone', frictionCone, 0)
contactDifferentialModel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel,
        contactCostModel,
        0., # inv_damping
        True) # bool enable force
contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, dt)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("jouleDissipation", u2, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e1)
terminalCostModel.addCost("gripperVelocity", goalFinalVelocity, 1e-1)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

problem_with_contact = crocoddyl.ShootingProblem(x0, [contactPhase] * 2 + [runningModel] * (T - 2), terminalModel)
problem_without_contact = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)


###  STANDING PROBLEM
"""
xref = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., -2.]))
supportContactModel = crocoddyl.ContactModel2D(state, xref, actuation.nu, np.array([0., 50.]))
contactModel2 = crocoddyl.ContactModelMultiple(state, actuation.nu)
contactModel2.addContact("foot_contact", supportContactModel)

Pref2 = crocoddyl.FrameTranslation(1,
                                np.array([0,0,np.sqrt(2)]))
goalTrackingCost2 = crocoddyl.CostModelFrameTranslation(state, Pref2, actuation.nu)
costModel2 = crocoddyl.CostModelSum(state, actuation.nu)
dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel2,
        costModel2,
        0., # inv_damping
        True) # bool enable force
modelr = crocoddyl.IntegratedActionModelEuler(dmodel, dt)
modelt = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
q02 = np.zeros(6)
q02[1] = np.pi
problem_standing = crocoddyl.ShootingProblem(np.zeros(6), [modelr] * T, modelt)
"""

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem_with_contact)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# Adittionally also modify ddp.th_stop and ddp.th_grad

# Solving it with the DDP algorithm
ddp.solve([],[], maxiter = int(1e3))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, ['foot', 'base', 'revolute_1'], trid = False)