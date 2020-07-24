import monoped
import os
import sys
import crocoddyl
import pinocchio
import numpy as np
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
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)
actuation = crocoddyl.ActuationModelFull(state)

# Setting the final position goal with variable angle
angle = 0
s = np.sin(angle)
c = np.cos(angle)
R = np.matrix([ [c,  0, s],
                [0,  1, 0],
                [-s, 0, c]
             ])
Mref = crocoddyl.FramePlacement(robot_model.getFrameId("foot"),
                                pinocchio.SE3(R, n_joints * np.matrix([[np.sin(angle)], [0], [np.cos(angle)]])))
Vref = crocoddyl.FrameMotion(robot_model.getFrameId("foot"), pinocchio.Motion(np.zeros(6)))
goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
goalFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref)
power_act =  crocoddyl.ActivationModelQuad(robot_model.nv)

u2 = crocoddyl.CostModelControl(state, power_act) # joule dissipation cost without friction, for benchmarking

# Then let's added the running and terminal cost functions
runningCostModel.addCost("jouleDissipation", u2, 1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)
terminalCostModel.addCost("gripperVelocity", goalFinalVelocity, 1)

q0 = np.array([np.pi] * 2 + [0] * (n_joints-1))
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverFDDP(problem)
ddp.robot_model = robot_model
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# ddp.th_stop = 1e-6
# ddp.th_grad = 1e-18

# Solving it with the DDP algorithm
ddp.solve([],[], maxiter = int(1e3))

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, 'foot')