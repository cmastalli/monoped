import crocoddyl
crocoddyl.switchToNumpyArray()
import numpy as np

class IntegratedActionModelLPF(crocoddyl.ActionModelAbstract):
    '''
        Add a low pass effect on the torque dynamics
            tau+ = alpha * tau + (1 - alpha) * w
        where alpha is a parameter depending of the memory of the system
        tau is the filtered torque included in the state and w the unfiltered control
        The state is augmented so that it includes the filtered torque
            y = [x, tau].T
    '''
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True, alpha=0):
            crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(diffModel.state.nx + diffModel.nu), diffModel.nu)
            self.differential = diffModel
            self.timeStep = timeStep
            self.withCostResiduals = withCostResiduals
            self.alpha = alpha
            self.nx = diffModel.state.nx
            self.nw = diffModel.nu # augmented dimension
            self.ny = self.nu + self.nx
            # print('INIT : nx ', self.nx, ' ny : ', self.ny, ' nw : ', self.nw)
            if self.timeStep == 0:
                self.enable_integration_ = False
            else:
                self.enable_integration_ = True
    
    def set_alpha(self, f_c = None):
        '''
            Sets the parameter alpha according to the cut-off frequency f_c
            alpha = 1 / (1 + 2pi dt f_c)
        '''
        if f_c > 0:
            omega = 1/(2 * np.pi * self.timeStep * f_c)
            self.alpha = omega/(omega + 1)
        else:
            self.alpha = 0

    def calc(self, data, x, u=None):
        return self.my_calc(data, x, u)

    def my_calc(self, data, y, w):
        # print("> START CALC")
        x = y[0:self.differential.state.nx] 
        # Brutally recreate data every time 
        data.dd = self.differential.createData()
        tau_plus = np.resize(np.array([self.alpha * y[self.nx:] + (1 - self.alpha) * w]), (self.nu,))
        self.differential.calc(data.dd, x, tau_plus) 
        data.xnext = np.zeros(self.ny) 
        if self.withCostResiduals: 
            data.r = data.dd.r 
        if self.enable_integration_: 
            data.cost = self.timeStep * data.dd.cost             
            data.dx = np.concatenate([x[self.differential.state.nq:] * self.timeStep + data.dd.xout * self.timeStep**2, data.dd.xout * self.timeStep]) 
            # print('x : ', x)
            # print('dx : ', data.dx)
            # print('xnext : ', data.xnext)
            # print('state.int : ', self.differential.state.integrate(x, data.dx))
            data.xnext[:self.nx] = self.differential.state.integrate(x, data.dx)
            data.xnext[self.nx:] = tau_plus 
        else:
            data.dx = np.zeros(len(y))
            data.xnext[:] = y
            data.cost = data.dd.cost
        
        # print('y : ', y)
        # print('x : ', x) 
        # print('w : ', w)
        # print('t+ : ', tau_plus)
        # print('self.alpha : ', self.alpha)
        # print('dx : ', data.dx) 
        # print('xout : ', data.dd.xout)
        # print('r : ', data.dd.r)
        # print('xnext : ', xnext)

        # print('X END CALC')

        return data.xnext, data.cost

    def calcDiff(self, data, y, w=None):
        # print("> START CALC DIFF")
        self.my_calc(data, y, w)

        x = y[:-self.differential.nu]
        tau_plus = np.resize(np.array([self.alpha * y[2] + (1 - self.alpha) * w]), (self.nw,))
        self.differential.calcDiff(data.dd, x, tau_plus)
        dd = data.dd
        dxnext_dx, dxnext_ddx = self.differential.state.Jintegrate(x, data.dx)
        da_dx, da_du = dd.Fx, np.resize(dd.Fu, (self.differential.state.nv, self.differential.nu))
        ddx_dx = np.vstack([da_dx * self.timeStep, da_dx])
        ddx_dx[range(self.differential.state.nv), range(self.differential.state.nv, 2 * self.differential.state.nv)] += 1
        ddx_du = np.vstack([da_du * self.timeStep, da_du])

        # print('y : ', y)
        # print('x : ', x) 
        # print('w : ', w)
        # print('t+ : ', tau_plus)
        # print('alpha : ', self.alpha)
        # print('dx : ', data.dx) 
        # print('xout : ', data.dd.xout)
        # print('r : ', data.dd.r)
        # print('dx+/dx, ddx+/dx : ', dxnext_dx, dxnext_ddx)
        # print('Fx : ', da_dx)
        # print('Fu : ', da_du)
        # print('ddx+/ddx', ddx_dx)

        # In this scope the data.* are in the augmented state coordinates
        # while all the differential dd are in the canonical x coordinates
        # we must set correctly the quantities where needed
        Fx = dxnext_dx + self.timeStep * np.dot(dxnext_ddx, ddx_dx)
        Fu = self.timeStep * np.dot(dxnext_ddx, ddx_du) # wrong according to NUM DIFF, no timestep
        # print('Fx : ', Fx)
        # print('Fu : ', Fu)

        # TODO why is this not multiplied by timestep?
        data.Fx[:self.nx, :self.nx] = Fx
        data.Fx[:self.nx, self.nx:self.ny] = self.alpha * Fu
        data.Fx[self.nx:, self.nx:] = self.alpha * np.eye(self.nu)
        # print('Fy : ', data.Fx)
        # TODO CHECKING WITH NUMDIFF, NO TIMESTEP HERE
        if self.nu == 1:
            data.Fu.flat[:self.nx] = (1 - self.alpha) * Fu
            data.Fu.flat[self.nx:] = (1 - self.alpha) * np.eye(self.nu)
        else:
            data.Fu[:self.nx, :self.nw] = (1 - self.alpha) * Fu
            data.Fu[self.nx:, :self.nw] = (1 - self.alpha) * np.eye(self.nu)                    

        if self.enable_integration_:

            data.Lx[:self.nx] = self.timeStep * dd.Lx
            data.Lx[self.nx:] = self.timeStep * self.alpha * dd.Lu

            data.Lu[:] = self.timeStep * (1 - self.alpha) * dd.Lu

            data.Lxx[:self.nx,:self.nx] = self.timeStep * dd.Lxx
            # TODO reshape is not the best, see better how to cast this
            data.Lxx[:self.nx,self.nx:] = self.timeStep * self.alpha * np.reshape(dd.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.timeStep * self.alpha * np.reshape(dd.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.timeStep * self.alpha**2 * dd.Luu

            data.Lxu[:self.nx] = self.timeStep * (1 - self.alpha) * dd.Lxu
            data.Lxu[self.nx:] = self.timeStep * (1 - self.alpha) * self.alpha * dd.Luu

            data.Luu[:, :] = self.timeStep * (1 - self.alpha)**2 * dd.Luu
        
        else:

            data.Lx[:self.nx] = dd.Lx
            data.Lx[self.nx:] = self.alpha * dd.Lu

            data.Lu[:] = (1 - self.alpha) * self.timeStep * dd.Lu

            data.Lxx[:self.nx,:self.nx] = dd.Lxx
            data.Lxx[:self.nx,self.nx:] = self.alpha * np.reshape(dd.Lxu, (self.nx, self.nu))
            data.Lxx[self.nx:,:self.nx] = self.alpha * np.reshape(dd.Lxu, (self.nu, self.nx))
            data.Lxx[self.nx:,self.nx:] = self.alpha**2 * dd.Luu

            data.Lxu[:self.nx] = (1 - self.alpha) * dd.Lxu
            data.Lxu[self.nx:] = (1 - self.alpha) * self.alpha * dd.Luu

            data.Luu[:, :] = (1 - self.alpha)**2 * dd.Luu