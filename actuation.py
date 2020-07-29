import crocoddyl
import numpy as np
import pinocchio

class ActuationModelMonoped(crocoddyl.ActuationModelAbstract):
    '''
        Creates the prismatic guide floating base actuated monoped
            n_links is the number of links in the monoped
            actuated base toggles the control of the base e.g. linear motor
    '''
    def __init__(self, state, n_links, actuated_base = False):
        crocoddyl.ActuationModelAbstract.__init__(self, state, n_links)
        self.nv = state.nv
        self.actuated_base = actuated_base
        self.n_links = n_links

    def calc(self, data, x, u):
        S = pinocchio.utils.zero((self.nv, self.nu))
        if self.actuated_base == True:
            S[0,:] = np.concatenate((np.array([1]), np.zeros(self.n_links-1)), axis=0)
        else:
            S[0,:] = np.concatenate((np.array([0]), np.zeros(self.n_links-1)), axis=0)

        S[-self.n_links:,:] = np.identity(self.n_links)
        data.tau[:] = S @ u

    def calcDiff(self, data, x, u):
        S = np.zeros((self.nv, self.nu))
        if self.actuated_base == True:
            S[0,:] = np.concatenate((np.array([1]), np.zeros(self.n_links-1)), axis=0)
        else:
            S[0,:] = np.concatenate((np.array([0]), np.zeros(self.n_links-1)), axis=0)

        S[-self.n_links:,:] = np.identity(self.n_links)
        data.dtau_du = S
