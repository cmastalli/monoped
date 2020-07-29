import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import monoped
import actuation
from utils import plotOCSolution, plotConvergence, plot_frame_trajectory
import conf

n_joints = conf.n_links
T = conf.T
dt = conf.dt
# Create the monoped and actuator
monoped = monoped.createMonopedWrapper(nbJoint = n_joints)
robot_model = monoped.model

# Create a cost model per the running and terminal action model
state = crocoddyl.StateMultibody(robot_model)
# actuation = crocoddyl.ActuationModelFull(state)
actuation = crocoddyl.ActuationModelFloatingBase(state)
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

q0 = np.zeros(1 + conf.n_links)
q0[1] = np.pi
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# Setting the final position goal with variable angle
angle = np.pi/2
s = np.sin(angle)
c = np.cos(angle)
R = np.matrix([ [c,  0, s],
                [0,  1, 0],
                [-s,  0, c]
             ])

footFrameID = robot_model.getFrameId("foot")
Pref = crocoddyl.FrameTranslation(footFrameID,
                                np.matrix([[np.sin(angle)], [0], [np.cos(angle)]]))
Vref = crocoddyl.FrameMotion(footFrameID, pinocchio.Motion(np.zeros(6)))
# If also the orientation is useful for the task
# Mref = crocoddyl.FramePlacement(footFrameID,
#                                pinocchio.SE3(R, n_joints * np.matrix([[np.sin(angle)], [0], [np.cos(angle)]])))
# goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)
goalFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)
power_act =  crocoddyl.ActivationModelQuad(n_joints)

# FRICTION CONE
mu = 0.5
normal_direction = np.array([0, 0, 1])
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)

xref = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., 0.]))
supportContactModel = crocoddyl.ContactModel3D(state, xref, actuation.nu, np.array([0., 50.]))
contactModel.addContact("foot_contact", supportContactModel)

# the friction cone can also have the [min, maximum] force parameters
# the number of faces

cone = crocoddyl.FrictionCone(normal_direction, mu, 4, False)
cone_bounds = crocoddyl.ActivationBounds(cone.lb, cone.ub)
cone_activation = crocoddyl.ActivationModelQuadraticBarrier(cone_bounds),
frame_friction = crocoddyl.FrameFrictionCone(footFrameID, cone)
frictionCone = crocoddyl.CostModelContactFrictionCone(state,
        cone_activation[0],
        frame_friction,
        actuation.nu)

# Creating the action model for the KKT dynamics with simpletic Euler
# integration scheme
costModel = crocoddyl.CostModelSum(state, actuation.nu)
costModel.addCost('frictionCone', frictionCone, 1e1)
dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(state,
        actuation,
        contactModel,
        costModel,
        0., # inv_damping
        True) # bool enable force
model = crocoddyl.IntegratedActionModelEuler(dmodel, dt)


u2 = crocoddyl.CostModelControl(state, power_act, actuation.nu) # joule dissipation cost without friction, for benchmarking

# Then let's added the running and terminal cost functions
runningCostModel.addCost("jouleDissipation", u2, 1e-2)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
terminalCostModel.addCost("gripperVelocity", goalFinalVelocity, 1)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# Adittionally also modify ddp.th_stop and ddp.th_grad

# Solving it with the DDP algorithm
ddp.solve([],[], maxiter = int(1e2))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, 'foot')
