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
actuation = actuation.ActuationModelMonoped(state, conf.n_links, False)
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)

q0 = np.zeros(1 + conf.n_links)
q0[1] = np.pi/2
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# Setting the final position goal with variable angle
angle = np.pi/3
s = np.sin(angle)
c = np.cos(angle)
R = np.matrix([ [c,  0, s],
                [0,  1, 0],
                [-s,  0, c]
             ])
# NOT SURE IF I WANT THIS "orientation" component
Pref = crocoddyl.FrameTranslation(robot_model.getFrameId("foot"),
                                np.matrix([[np.sin(angle)], [0], [np.cos(angle)]]))
Vref = crocoddyl.FrameMotion(robot_model.getFrameId("foot"), pinocchio.Motion(np.zeros(6)))
#Mref = crocoddyl.FramePlacement(robot_model.getFrameId("tip"),
#                                pinocchio.SE3(R, n_joints * np.matrix([[np.sin(angle)], [0], [np.cos(angle)]])))
#goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)
goalFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)
power_act =  crocoddyl.ActivationModelQuad(n_joints)

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
# ddp.th_stop = 1e-6
# ddp.th_grad = 1e-18

# Solving it with the DDP algorithm
ddp.solve([],[], maxiter = int(1e2))
ddp.robot_model = robot_model

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, 'foot')
