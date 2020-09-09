import numpy as np

# CORRELATION found for BRUSHLESS MOTORS
def remy_correlation(motor_mass, n_gear=1):
    T0 = 0.570 * np.power(motor_mass, 1.2)
    omega0 = 1.35e3 * np.power(motor_mass, -.182)
    I_m = 2.85 * 1e-5 * np.power(motor_mass, 1.72)
    K_m = 5.67 * 1e-3 * np.power(motor_mass, 1.8)
    eta = gearbox_efficiency(n_gear)
    T_mu = T0 * (1 - eta)/eta
    return T0, omega0, I_m, K_m, T_mu

def timing_belt_correlation(motor_mass, n_gear=1):
    # makes sense in range 0 - 1 kg
    # Antigravity
    # T-Motor Antigravity 4004 300kV
    # Solo motor_mass 53g
    T0 = 5.48 * np.power(motor_mass, .966)
    # identified for range of motors
    # for Solo actuator -> 2.88 Nm, 15% surplus
    omega0 = 1.35e3 * np.power(motor_mass, -.182) # TODO
    I_m = 5.38 * 1e-6 * np.power(motor_mass, 5/3)/(np.power(53e-3, 5/3))
    # isotropic increase in mass is roughly m**1/3, I increases as m**5/3 (1.66 close to 1.72)
    K_m = 0.15 * np.power(motor_mass, 1.39) # huge increase wrt to other option
    # eta = gearbox_efficiency(n_gear)
    # Friction supposing made up by a number of equal reductions
    k = np.log(n_gear)/np.log(9) # range from 1 to infinity
    T_mu = T0/(2.88/9) * k * 5.3e-3 * n_gear/9
    return np.array(T0), np.array(omega0), np.array(I_m), np.array(K_m), np.array(T_mu) # friction proportional to the torque and the number of stages

def remy_mass_from_torque(T0):
    return np.power(T0/.57, 1/1.2)

# CORRELATION found for GEARBOX TRANSMISSION EFFICIENCY
def gearbox_efficiency(n_gear):
    gearbox_efficiency = np.power(n_gear,-0.0952)
    return gearbox_efficiency