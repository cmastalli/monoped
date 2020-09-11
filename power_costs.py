import crocoddyl
import numpy as np
import pinocchio
pinocchio.switchToNumpyArray()

class CostModelJointFriction(crocoddyl.CostModelAbstract):
    '''
        Describes the Coulomb friction power losses P_f
        T_f = T_mu sign(omega_m) [Nm]
        P_f = T_mu abs(omega_m) [W]
    '''
    def __init__(self, state, activation, nu):
        if not hasattr(state, 'robot_model'):
            raise Exception('State needs to have the model parameters, add the model to the state')
        if not hasattr(state.robot_model, 'T_mu'):
            state.T_mu = 0.00
        if not hasattr(state.robot_model, 'K_m'):
            state.K_m = 0.00
        self.nv = state.nv
        self.T_mu = state.robot_model.T_mu
        self.n = state.robot_model.rotorGearRatio
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu = nu)

    def calc(self, data, x, u):
        # exact formulation
        data.cost = np.sum(self.T_mu * self.n * np.abs(x[-self.nu:]))

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)

        # Gradient
        data.Lx[:-self.nu] = 0
        data.Lx[-self.nu:] = self.T_mu * self.n * np.sign(x[-self.nu:])
        data.Lu[:] = 0

        # Hessian
        data.Lxx.fill(0)
        data.Luu.fill(0)
        data.Lxu.fill(0)

class CostModelJointFrictionSmooth(crocoddyl.CostModelAbstract):
    '''
        Describes the Coulomb friction power losses P_f
        T_f = T_mu sign(omega_m) [Nm]
        P_f = T_mu abs(omega_m) [W]
        the absolute value is approximated for better convergence
    '''
    def __init__(self, state, activation, nu):
        if not hasattr(state, 'robot_model'):
            raise Exception('State needs to have the model parameters, add the model to the state')
        if not hasattr(state.robot_model, 'T_mu'):
            state.T_mu = 0.00
        self.nv = state.nv
        self.nq = state.nq
        self.nx = state.nx
        self.nfb = 1
        self.T_mu = state.robot_model.T_mu
        self.n = state.robot_model.rotorGearRatio[-nu:]
        self.gamma = 1
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu = nu)

    def calc(self, data, x, u):
        # exact formulation
        data.cost = np.sum(self.T_mu * self.n *  np.tanh(self.gamma * x[-self.nu:]) * x[-self.nu:])

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)

        # Gradient
        data.Lx[:-self.nu] = np.zeros(self.nx - self.nu)
        data.Lx[-self.nu:] = self.T_mu * self.n * np.tanh(self.gamma * x[-self.nu:])
        data.Lu[:] = 0

        # Hessian
        data.Lxx[:,:] = 0
        data.Lxx[-self.nu:, -self.nu:] = - np.diag(self.T_mu * self.n * self.gamma * (np.tanh(self.gamma * x[-self.nu:])**2 - 1))
        data.Luu.fill(0)
        data.Lxu.fill(0)

class CostModelJouleDissipation(crocoddyl.CostModelAbstract):
    '''
        This cost is taking into account in the Joule dissipation P_t
        to the motor torque to drive to morion it's also added the Coulomb friction torque
        T_f = T_mu sign(omega_m) [Nm]
        P_t = (T_m + T_f).T [K] (T_m + T_f) [W]
    '''
    def __init__(self, state, activation, nu):

        self.nv = state.nv
        self.nx = state.nx
        self.nq = state.nq
        # self.nu = nu
        if not hasattr(state, 'robot_model'):
            raise Exception('State needs to have the model parameters, reference the model in state')
        if not hasattr(state.robot_model, 'T_mu'):
            # if not specified otherwise, no friction
            self.T_mu = 0.00
        else:
            self.T_mu = state.robot_model.T_mu

        if not hasattr(state.robot_model, 'K_m'):
            # if not specified otherwise, no cost on torque
            self.K = np.eye(nu)
        else:
            self.K = np.array(1/state.robot_model.K_m)
        self.n = state.robot_model.rotorGearRatio

        # Modifies the parameters to match the u**2 cost
        nominal = False
        if nominal:
            self.K = np.ones(nu)/2
            self.T_mu = np.zeros(nu)
            self.n = np.ones(nu)

        self.eps = 1e-3
        self.gamma = 1e0 * 1e2/self.n
        # the partial derivative with respect to the control is stored once, since it is constant
        self.dTmdu = 1/self.n
        # next lines are just for the abstraction
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu = nu)

    def calc(self, data, x, u):
        # shortname the velocity
        data.v = np.asarray(x[-self.nu:])
        # compute the torques
        data.T_f = np.asarray(self.T_mu * np.tanh(self.gamma * data.v))
        data.T_m = np.asarray(np.divide(u, self.n))
        data.T_tot = data.T_f + data.T_m
        # compute the Joule power loss cost
        data.cost = np.sum(self.K * data.T_tot**2)

    def calcDiff(self, data, x, u, recalc = True):
        if recalc:
            self.calc(data, x, u)

        # As preliminary step, compute also the partial derivatives
        # APPROXIMATION sign(v) = tanh(gamma * x)
        data.dTfdx = - self.T_mu * self.gamma * (np.tanh(self.gamma*data.v)**2 - 1)
        data.d2Tfdx2 = 2 * self.T_mu * self.gamma**2 *(np.tanh(self.gamma*data.v)**2 - 1) * np.tanh(self.gamma*data.v)

        # Gradient
        data.Lx[:] = np.zeros(self.nx)
        data.Lx[-self.nu:] = 2 * data.T_tot * self.K * data.dTfdx
        data.Lu[:] = 2 * self.K * data.T_tot / self.n

        # Hessian
        data.Lxx[:,:] = np.zeros((self.nx, self.nx))
        data.Lxx[-self.nu:, -self.nu:] = np.diag(2 * (data.T_tot * self.K) * data.d2Tfdx2 + 2 * self.K * data.dTfdx**2)
        data.Luu[:,:] = 2 * np.diag(self.K  / self.n**2)

        if self.nu > 1:
            data.Lxu[:,:] = np.zeros((self.nx, self.nu))
            data.Lxu[-self.nu:, :] = 2 * np.diag(data.dTfdx * self.K / self.n)
        else:
            data.Lxu[:] = np.zeros(self.nx)
            data.Lxu[-self.nu:] = 2 * np.diag(data.dTfdx * self.K / self.n)