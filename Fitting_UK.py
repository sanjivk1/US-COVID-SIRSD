import numpy as np
import pandas as pd
import time
import math
import concurrent.futures
import multiprocessing
from scipy.optimize import minimize
import matplotlib

import matplotlib.pyplot as plt
import sys
from sklearn.metrics import r2_score, mean_squared_error
from SIRfunctions import SEIARG, SEIARG_fixed, weighted_deviation, weighted_relative_deviation, computeBeta_combined
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

np.set_printoptions(threshold=sys.maxsize)
# Geo = 0.98
Geo = 0.96
num_para = 14

# num_threads = 200
num_threads = 10
num_CI = 1000
# num_CI = 5
start_dev = 0

num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
beta_range = (0.1, 100)
gammaE_range = (0.2, 0.3)
alpha_range = (0.1, 0.9)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
gamma3_range = (0.04, 0.2)
# sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.001, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.95)
c1_fixed = (0.9, 0.9)
c1_range = (0.8, 1)
h_range = (1 / 30, 1 / 14)
Hiding_init_range = (0.2, 0.9)
k_range = (0.1, 2)
k2_range = (0.1, 2)
I_initial_range = (0, 1)
start_date = '2020-12-01'
reopen_date = '2021-04-25'
end_date = '2021-06-08'
vac_date = '2021-01-01'
daily_vspeed = 0.005
v_period1 = 14
v_period2 = 7 * 12
v_eff1 = 0.65
v_eff2 = 0.8
# release_duration = 30
# k_drop = 14
# p_m = 1
# Hiding = 0.33
# delay = 7
# change_eta2 = False
size_ext = 150

fig_row = 5
fig_col = 3

states = ['England', 'Northern Ireland', 'Scotland', 'Wales']


def simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h,
                      Hiding_init, eta, c1, n_0, reopen_day):
    result = True
    H0 = H[0]
    eta2 = eta * (1 - Hiding_init)
    r = h * H0
    betas = [beta]
    for i in range(1, size):

        if i > reopen_day:
            release = min(H[-1], r)
            S[-1] += release
            H[-1] -= release

        delta = SEIARG(i,
                       [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1],
                        beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta2, n_0, c1, H[-1], H0])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        A.append(A[-1] + delta[3])
        IH.append(IH[-1] + delta[4])
        IN.append(IN[-1] + delta[5])
        D.append(D[-1] + delta[6])
        R.append(R[-1] + delta[7])
        G.append(G[-1] + delta[8])
        H.append(H[-1])
        betas.append(delta[9])
        if S[-1] < 0:
            result = False
            break
    return result, [S, E, I, A, IH, IN, D, R, G, H, betas]


def simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, h,
                 Hiding_init, eta, c1, n_0, reopen_day, vaccine_day, vac_speed, vac_period1, vac_period2, vac_eff1,
                 vac_eff2):
    result = True

    vaccine_speeds = [vac_speed] * size

    # no dose
    S0 = S.copy()
    E0 = E.copy()
    I0 = I.copy()
    A0 = A.copy()
    IH0 = IH.copy()
    IN0 = IN.copy()
    D0 = D.copy()
    R0 = R.copy()
    G0 = G.copy()
    H0 = H.copy()
    betas = [beta]

    # 1st dose
    S1 = [0]
    E1 = [0]
    I1 = [0]
    A1 = [0]
    IH1 = [0]
    IN1 = [0]
    D1 = [0]
    R1 = [0]
    G1 = [0]
    H1 = [0]

    # 2nd dose (or 2 weeks after 1st dose for it to take effect)
    S2 = [0]
    E2 = [0]
    I2 = [0]
    A2 = [0]
    IH2 = [0]
    IN2 = [0]
    D2 = [0]
    R2 = [0]
    G2 = [0]
    H2 = [0]

    # fully vaccinated
    S3 = [0]
    E3 = [0]
    I3 = [0]
    A3 = [0]
    IH3 = [0]
    IN3 = [0]
    D3 = [0]
    R3 = [0]
    G3 = [0]
    H3 = [0]

    eta2 = eta * (1 - Hiding_init)
    Hiding0 = H0[0]
    r = h * H0[0]
    # HH = [release_size * n_0]
    # daily_release = release_speed * release_size * n_0
    for i in range(1, size):

        vaccine_speed = vaccine_speeds[i]

        beta_t = computeBeta_combined(beta, eta2, n_0,
                                      S0[-1] + S1[-1] + S2[-1] + S3[-1],
                                      0,
                                      H0[-1] + H1[-1] + H2[-1] + H3[-1],
                                      c1, Hiding0)
        dS0 = -beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
        dE0 = beta_t * S0[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
              E0[-1]
        dI0 = (1 - alpha) * gammaE * E0[-1] - (gamma + gamma2) * I0[-1]
        dA0 = alpha * gammaE * E0[-1] - gamma3 * A0[-1]
        dIH0 = gamma * I0[-1] - (a1 + a2) * IH0[-1]
        dIN0 = gamma2 * I0[-1] - a3 * IN0[-1]
        dD0 = a2 * IH0[-1]
        dR0 = a1 * IH0[-1] + a3 * IN0[-1] + gamma3 * A0[-1]
        dG0 = (1 - alpha) * gammaE * E0[-1]

        dS1 = -beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
        dE1 = beta_t * S1[-1] * (I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * \
              E1[-1]
        dI1 = (1 - alpha) * gammaE * E1[-1] - (gamma + gamma2) * I1[-1]
        dA1 = alpha * gammaE * E1[-1] - gamma3 * A1[-1]
        dIH1 = gamma * I1[-1] - (a1 + a2) * IH1[-1]
        dIN1 = gamma2 * I1[-1] - a3 * IN1[-1]
        dD1 = a2 * IH1[-1]
        dR1 = a1 * IH1[-1] + a3 * IN1[-1] + gamma3 * A1[-1]
        dG1 = (1 - alpha) * gammaE * E1[-1]

        dS2 = -beta_t * (1 - vac_eff1) * S2[-1] * (
                I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
        dE2 = beta_t * (1 - vac_eff1) * S2[-1] * (
                I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E2[-1]
        dI2 = (1 - alpha) * gammaE * E2[-1] - (gamma + gamma2) * I2[-1]
        dA2 = alpha * gammaE * E2[-1] - gamma3 * A2[-1]
        dIH2 = gamma * I2[-1] - (a1 + a2) * IH2[-1]
        dIN2 = gamma2 * I2[-1] - a3 * IN2[-1]
        dD2 = a2 * IH2[-1]
        dR2 = a1 * IH2[-1] + a3 * IN2[-1] + gamma3 * A2[-1]
        dG2 = (1 - alpha) * gammaE * E2[-1]

        dS3 = -beta_t * (1 - vac_eff2) * S3[-1] * (
                I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0
        dE3 = beta_t * (1 - vac_eff2) * S3[-1] * (
                I0[-1] + I1[-1] + I2[-1] + I3[-1] + A0[-1] + A1[-1] + A2[-1] + A3[-1]) / n_0 - gammaE * E3[-1]
        dI3 = (1 - alpha) * gammaE * E3[-1] - (gamma + gamma2) * I3[-1]
        dA3 = alpha * gammaE * E3[-1] - gamma3 * A3[-1]
        dIH3 = gamma * I3[-1] - (a1 + a2) * IH3[-1]
        dIN3 = gamma2 * I3[-1] - a3 * IN3[-1]
        dD3 = a2 * IH3[-1]
        dR3 = a1 * IH3[-1] + a3 * IN3[-1] + gamma3 * A3[-1]
        dG3 = (1 - alpha) * gammaE * E3[-1]

        S0.append(S0[-1] + dS0)
        E0.append(E0[-1] + dE0)
        I0.append(I0[-1] + dI0)
        A0.append(A0[-1] + dA0)
        IH0.append(IH0[-1] + dIH0)
        IN0.append(IN0[-1] + dIN0)
        D0.append(D0[-1] + dD0)
        R0.append(R0[-1] + dR0)
        G0.append(G0[-1] + dG0)

        S1.append(S1[-1] + dS1)
        E1.append(E1[-1] + dE1)
        I1.append(I1[-1] + dI1)
        A1.append(A1[-1] + dA1)
        IH1.append(IH1[-1] + dIH1)
        IN1.append(IN1[-1] + dIN1)
        D1.append(D1[-1] + dD1)
        R1.append(R1[-1] + dR1)
        G1.append(G1[-1] + dG1)

        S2.append(S2[-1] + dS2)
        E2.append(E2[-1] + dE2)
        I2.append(I2[-1] + dI2)
        A2.append(A2[-1] + dA2)
        IH2.append(IH2[-1] + dIH2)
        IN2.append(IN2[-1] + dIN2)
        D2.append(D2[-1] + dD2)
        R2.append(R2[-1] + dR2)
        G2.append(G2[-1] + dG2)

        S3.append(S3[-1] + dS3)
        E3.append(E3[-1] + dE3)
        I3.append(I3[-1] + dI3)
        A3.append(A3[-1] + dA3)
        IH3.append(IH3[-1] + dIH3)
        IN3.append(IN3[-1] + dIN3)
        D3.append(D3[-1] + dD3)
        R3.append(R3[-1] + dR3)
        G3.append(G3[-1] + dG3)

        H0.append(H0[-1])
        H1.append(H1[-1])
        H2.append(H2[-1])
        H3.append(H3[-1])
        # HH0.append(HH0[-1])
        # HH1.append(HH1[-1])
        # HH2.append(HH2[-1])
        # HH3.append(HH3[-1])

        betas.append(beta_t)

        dS12 = S1[i] / vac_period1
        dS23 = S2[i] / vac_period2
        S1[i] -= dS12
        S2[i] = S2[i] - dS23 + dS12
        S3[i] += dS23

        dH12 = H1[i] / vac_period1
        dH23 = H2[i] / vac_period2
        H1[i] -= dH12
        H2[i] = H2[i] - dH23 + dH12
        H3[i] += dH23

        # if i >= reopen_day:
        # 	release = min(H0[-1], r)
        # 	S0[-1] += release
        # 	H0[-1] -= release

        total_H = H0[-1] + H1[-1] + H2[-1] + H3[-1]
        if i >= reopen_day and total_H > 0:
            release = min(r, total_H)
            frac0 = H0[-1] / total_H
            frac1 = H1[-1] / total_H
            frac2 = H2[-1] / total_H
            frac3 = H3[-1] / total_H
            S0[-1] += release * frac0
            S1[-1] += release * frac1
            S2[-1] += release * frac2
            S3[-1] += release * frac3
            H0[-1] -= release * frac0
            H1[-1] -= release * frac1
            H2[-1] -= release * frac2
            H3[-1] -= release * frac3
        # Hiding0 += release

        if i >= vaccine_day:
            S1[-1] += S0[-1] * vaccine_speed
            S0[-1] -= S0[-1] * vaccine_speed
            H1[-1] += H0[-1] * vaccine_speed
            H0[-1] -= H0[-1] * vaccine_speed

        if S0[-1] < 0:
            result = False
            break

    if result:
        S = [S0[i] + S1[i] + S2[i] + S3[i] for i in range(size)]
        E = [E0[i] + E1[i] + E2[i] + E3[i] for i in range(size)]
        I = [I0[i] + I1[i] + I2[i] + I3[i] for i in range(size)]
        A = [A0[i] + A1[i] + A2[i] + A3[i] for i in range(size)]
        IH = [IH0[i] + IH1[i] + IH2[i] + IH3[i] for i in range(size)]
        IN = [IN0[i] + IN1[i] + IN2[i] + IN3[i] for i in range(size)]
        D = [D0[i] + D1[i] + D2[i] + D3[i] for i in range(size)]
        R = [R0[i] + R1[i] + R2[i] + R3[i] for i in range(size)]
        G = [G0[i] + G1[i] + G2[i] + G3[i] for i in range(size)]
        H = [H0[i] + H1[i] + H2[i] + H3[i] for i in range(size)]

    return result, [S, E, I, A, IH, IN, D, R, G, H,
                    S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0,
                    S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
                    S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2,
                    S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
                    betas]


def simulate_release(size, S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                     a3, h, Hiding_init, eta, c1, n_0, reopen_day, release_day, release_size, daily_speed):
    S = S0.copy()
    E = E0.copy()
    I = I0.copy()
    A = A0.copy()
    IH = IH0.copy()
    IN = IN0.copy()
    D = D0.copy()
    R = R0.copy()
    G = G0.copy()
    H = H0.copy()
    result = True
    H0 = H[0]
    eta2 = eta * (1 - Hiding_init)
    r = h * H0
    betas = [beta]
    HH = [release_size * n_0]
    daily_release = daily_speed * HH[-1]
    for i in range(1, size):

        if i > reopen_day:
            release = min(H[-1], r)
            S[-1] += release
            H[-1] -= release

        if i > release_day:
            release = min(daily_release, HH[-1])
            S[-1] += release
            HH[-1] -= release
            H0 += release

        delta = SEIARG(i,
                       [S[i - 1], E[i - 1], I[i - 1], A[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1],
                        beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta2, n_0, c1, H[-1], H0])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        A.append(A[-1] + delta[3])
        IH.append(IH[-1] + delta[4])
        IN.append(IN[-1] + delta[5])
        D.append(D[-1] + delta[6])
        R.append(R[-1] + delta[7])
        G.append(G[-1] + delta[8])
        H.append(H[-1])
        HH.append(HH[-1])
        betas.append(delta[9])
        if S[-1] < 0:
            result = False
            break
    return result, [S, E, I, A, IH, IN, D, R, G, H, HH, betas]


def loss_combined(point, c1, confirmed, death, n_0, reopen_day):
    size = len(confirmed)
    beta = point[0]
    gammaE = point[1]
    alpha = point[2]
    gamma = point[3]
    gamma2 = point[4]
    gamma3 = point[5]
    a1 = point[6]
    a2 = point[7]
    a3 = point[8]
    eta = point[9]
    h = point[10]
    Hiding_init = point[11]
    I_initial = point[12]
    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                            a3, h, Hiding_init, eta, c1, n_0, reopen_day)

    if not result:
        return 1000000

    size1 = reopen_day
    size2 = size - size1
    weights1 = [Geo ** n for n in range(size1)]
    weights1.reverse()
    weights2 = [Geo ** n for n in range(size2)]
    weights2.reverse()
    weights = weights1
    weights.extend(weights2)
    if len(weights) != size:
        print('wrong weights!')

    weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
    weighted_G = [G[i] * weights[i] for i in range(size)]
    weighted_death = [death[i] * weights[i] for i in range(size)]
    weighted_D = [D[i] * weights[i] for i in range(size)]

    metric0 = r2_score(weighted_confirmed, weighted_G)
    metric1 = r2_score(weighted_death, weighted_D)

    return -(0.9 * metric0 + 0.1 * metric1)


def loss_vac(point, c1, confirmed, death, n_0, reopen_day, vaccine_day):
    size = len(confirmed)
    beta = point[0]
    gammaE = point[1]
    alpha = point[2]
    gamma = point[3]
    gamma2 = point[4]
    gamma3 = point[5]
    a1 = point[6]
    a2 = point[7]
    a3 = point[8]
    eta = point[9]
    h = point[10]
    Hiding_init = point[11]
    I_initial = point[12]
    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    result, [S, E, I, A, IH, IN, D, R, G, H,
             S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
             S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
             betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
                                   a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, vaccine_day, daily_vspeed,
                                   v_period1, v_period2, v_eff1, v_eff2)

    if not result:
        return 1000000

    size1 = reopen_day
    size2 = size - size1
    weights1 = [Geo ** n for n in range(size1)]
    weights1.reverse()
    weights2 = [Geo ** n for n in range(size2)]
    weights2.reverse()
    weights = weights1
    weights.extend(weights2)
    if len(weights) != size:
        print('wrong weights!')

    weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
    weighted_G = [G[i] * weights[i] for i in range(size)]
    weighted_death = [death[i] * weights[i] for i in range(size)]
    weighted_D = [D[i] * weights[i] for i in range(size)]

    metric0 = r2_score(weighted_confirmed, weighted_G)
    metric1 = r2_score(weighted_death, weighted_D)

    return -(0.9 * metric0 + 0.1 * metric1)


def fit(confirmed0, death0, reopen_day_gov, n_0):
    np.random.seed()
    confirmed = confirmed0.copy()
    death = death0.copy()
    size = len(confirmed)
    # if metric2 != 0 or metric1 != 0:
    # 	scale1 = pd.Series(np.random.normal(1, metric1, size))
    # 	confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
    # 	scale2 = pd.Series(np.random.normal(1, metric2, size))
    # 	death = [max(death[i] * scale2[i], 1) for i in range(size)]
    c_max = 0
    min_loss = 10000
    for reopen_day in range(reopen_day_gov, reopen_day_gov + 14):
        for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
            # optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
            optimal = minimize(loss_combined, [uni(beta_range[0], beta_range[1]),
                                               uni(gammaE_range[0], gammaE_range[1]),
                                               uni(alpha_range[0], alpha_range[1]),
                                               uni(gamma_range[0], gamma_range[1]),
                                               uni(gamma2_range[0], gamma2_range[1]),
                                               uni(gamma3_range[0], gamma3_range[1]),
                                               uni(a1_range[0], a1_range[1]),
                                               uni(a2_range[0], a2_range[1]),
                                               uni(a3_range[0], a3_range[1]),
                                               uni(eta_range[0], eta_range[1]),
                                               uni(h_range[0], h_range[1]),
                                               uni(Hiding_init_range[0], Hiding_init_range[1]),
                                               uni(I_initial_range[0], I_initial_range[1])],
                               args=(c1, confirmed, death, n_0, reopen_day), method='L-BFGS-B',
                               bounds=[beta_range,
                                       gammaE_range,
                                       alpha_range,
                                       gamma_range,
                                       gamma2_range,
                                       gamma3_range,
                                       a1_range,
                                       a2_range,
                                       a3_range,
                                       eta_range,
                                       h_range,
                                       Hiding_init_range,
                                       I_initial_range])
            current_loss = loss_combined(optimal.x, c1, confirmed, death, n_0, reopen_day)
            if current_loss < min_loss:
                # print(f'updating loss={current_loss} with c1={c1}')
                min_loss = current_loss
                c_max = c1
                reopen_max = reopen_day
                beta = optimal.x[0]
                gammaE = optimal.x[1]
                alpha = optimal.x[2]
                gamma = optimal.x[3]
                gamma2 = optimal.x[4]
                gamma3 = optimal.x[5]
                a1 = optimal.x[6]
                a2 = optimal.x[7]
                a3 = optimal.x[8]
                eta = optimal.x[9]
                h = optimal.x[10]
                Hiding_init = optimal.x[11]
                I_initial = optimal.x[12]

    c1 = c_max
    reopen_day = reopen_max
    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    # Betas = [beta]

    result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                            a3, h, Hiding_init, eta, c1, n_0, reopen_day)

    # data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
    # data2 = [(death[i] - D[i]) / death[i] for i in range(size)]

    size1 = reopen_day
    size2 = size - size1
    weights1 = [Geo ** n for n in range(size1)]
    weights1.reverse()
    weights2 = [Geo ** n for n in range(size2)]
    weights2.reverse()
    weights = weights1
    weights.extend(weights2)

    # weights = [Geo ** n for n in range(size)]
    # weights.reverse()

    # sum_wt = sum(weights)
    # metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
    #                     /
    #                     ((size - 12) * sum_wt / size)
    #                     )
    # metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
    #                     /
    #                     ((size - 12) * sum_wt / size)
    #                     )
    metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, num_para)
    metric2 = weighted_relative_deviation(weights, death, D, start_dev, num_para)

    r1 = r2_score(confirmed, G)
    r2 = r2_score(death, D)

    return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1,
            metric2, r1, r2, reopen_day], min_loss


def fit_vac(confirmed0, death0, reopen_day_gov, vaccine_day, n_0):
    np.random.seed()
    confirmed = confirmed0.copy()
    death = death0.copy()
    size = len(confirmed)
    # if metric2 != 0 or metric1 != 0:
    # 	scale1 = pd.Series(np.random.normal(1, metric1, size))
    # 	confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
    # 	scale2 = pd.Series(np.random.normal(1, metric2, size))
    # 	death = [max(death[i] * scale2[i], 1) for i in range(size)]
    c_max = 0
    min_loss = 10000
    for reopen_day in range(reopen_day_gov, min(reopen_day_gov + 14, size)):
        for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
            # optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
            optimal = minimize(loss_vac, [uni(beta_range[0], beta_range[1]),
                                          uni(gammaE_range[0], gammaE_range[1]),
                                          uni(alpha_range[0], alpha_range[1]),
                                          uni(gamma_range[0], gamma_range[1]),
                                          uni(gamma2_range[0], gamma2_range[1]),
                                          uni(gamma3_range[0], gamma3_range[1]),
                                          uni(a1_range[0], a1_range[1]),
                                          uni(a2_range[0], a2_range[1]),
                                          uni(a3_range[0], a3_range[1]),
                                          uni(eta_range[0], eta_range[1]),
                                          uni(h_range[0], h_range[1]),
                                          uni(Hiding_init_range[0], Hiding_init_range[1]),
                                          uni(I_initial_range[0], I_initial_range[1])],
                               args=(c1, confirmed, death, n_0, reopen_day, vaccine_day), method='L-BFGS-B',
                               bounds=[beta_range,
                                       gammaE_range,
                                       alpha_range,
                                       gamma_range,
                                       gamma2_range,
                                       gamma3_range,
                                       a1_range,
                                       a2_range,
                                       a3_range,
                                       eta_range,
                                       h_range,
                                       Hiding_init_range,
                                       I_initial_range])
            current_loss = loss_vac(optimal.x, c1, confirmed, death, n_0, reopen_day, vaccine_day)
            if current_loss < min_loss:
                # print(f'updating loss={current_loss} with c1={c1}')
                min_loss = current_loss
                c_max = c1
                reopen_max = reopen_day
                beta = optimal.x[0]
                gammaE = optimal.x[1]
                alpha = optimal.x[2]
                gamma = optimal.x[3]
                gamma2 = optimal.x[4]
                gamma3 = optimal.x[5]
                a1 = optimal.x[6]
                a2 = optimal.x[7]
                a3 = optimal.x[8]
                eta = optimal.x[9]
                h = optimal.x[10]
                Hiding_init = optimal.x[11]
                I_initial = optimal.x[12]

    c1 = c_max
    reopen_day = reopen_max
    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    # Betas = [beta]

    result, [S, E, I, A, IH, IN, D, R, G, H,
             S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
             S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
             betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
                                   a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, vaccine_day, daily_vspeed,
                                   v_period1, v_period2, v_eff1, v_eff2)

    # data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
    # data2 = [(death[i] - D[i]) / death[i] for i in range(size)]

    size1 = min(reopen_day, size)
    size2 = size - size1
    weights1 = [Geo ** n for n in range(size1)]
    weights1.reverse()
    weights2 = [Geo ** n for n in range(size2)]
    weights2.reverse()
    weights = weights1
    weights.extend(weights2)

    # weights = [Geo ** n for n in range(size)]
    # weights.reverse()

    # sum_wt = sum(weights)
    # metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
    #                     /
    #                     ((size - 12) * sum_wt / size)
    #                     )
    # metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
    #                     /
    #                     ((size - 12) * sum_wt / size)
    #                     )
    metric1 = weighted_relative_deviation(weights, confirmed, G, start_dev, num_para)
    metric2 = weighted_relative_deviation(weights, death, D, start_dev, num_para)

    r1 = r2_score(confirmed, G)
    r2 = r2_score(death, D)

    return [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1,
            metric2, r1, r2, reopen_day], min_loss


def MT_fitting(confirmed, death, n_0, reopen_day_gov):
    para_best = []
    min_loss = 10000
    with concurrent.futures.ProcessPoolExecutor() as executor:
        t1 = time.perf_counter()
        results = [executor.submit(fit, confirmed, death, reopen_day_gov, n_0) for _ in range(num_threads)]

        threads = 0
        for f in concurrent.futures.as_completed(results):
            para, current_loss = f.result()
            threads += 1
            # print(f'thread {threads} returned')
            if current_loss < min_loss:
                min_loss = current_loss
                para_best = para
                print(f'best paras updated at {threads} with loss={min_loss}')
        # if threads % 10 == 0:
        # 	print(f'{threads}/{num_threads} thread(s) completed')

        t2 = time.perf_counter()
        print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

    print('initial best fitting completed\n')
    return para_best


def MT_fitting_vac(state, confirmed, death, n_0, reopen_day_gov, vaccine_day):
    para_best = []
    min_loss = 10000
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # t1 = time.perf_counter()
        results = [executor.submit(fit_vac, confirmed, death, reopen_day_gov, vaccine_day, n_0) for _ in
                   range(num_threads)]

        threads = 0
        for f in concurrent.futures.as_completed(results):
            para, current_loss = f.result()
            threads += 1
            # print(f'thread {threads} returned')
            if current_loss < min_loss:
                min_loss = current_loss
                para_best = para
                print(f'{state} #{threads} loss={min_loss}')
            else:
                print(f'{state} #{threads}')
    # if threads % 10 == 0:
    # 	print(f'{threads}/{num_threads} thread(s) completed')

    # t2 = time.perf_counter()
    # print(f'{round(t2 - t1, 3)} seconds\n{round((t2 - t1) / num_threads, 3)} seconds per job')

    # print('initial best fitting completed\n')
    return para_best


def fit_state(state, ConfirmFile, DeathFile, PopFile):
    t1 = time.perf_counter()
    state_path = f'UK/fitting_{end_date}/{state}'
    if not os.path.exists(state_path):
        os.makedirs(state_path)

    print(state)
    print()

    # read population
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    # select confirmed and death data
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    df2 = pd.read_csv(DeathFile)
    death = df2[df2.iloc[:, 0] == state]

    # for start_date in confirmed.columns[1:]:
    # 	# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
    # 	if confirmed.iloc[0].loc[start_date] >= I_0:
    # 		break

    days = list(confirmed.columns)
    days = days[days.index(start_date):days.index(end_date) + 1]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    confirmed = confirmed.iloc[0].loc[start_date: end_date]
    death = death.iloc[0].loc[start_date: end_date]
    for i in range(len(death)):
        if death.iloc[i] == 0:
            death.iloc[i] = 0.01
    death = death.tolist()

    reopen_day_gov = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))

    # fitting
    para = MT_fitting(confirmed, death, n_0, reopen_day_gov)
    [S, E, I, A, IH, IN, D, R, G, H, betas] = plot_combined(state, confirmed, death, days, n_0, para, state_path)

    save_sim_combined([S, E, I, A, IH, IN, D, R, G, H, betas], days, state_path)

    para[-1] = days[para[-1]]
    save_para_combined([para], state_path)
    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 60, 3)} minutes in total for {state}\n')

    return


def fit_state_vac(state, ConfirmFile, DeathFile, PopFile):
    t1 = time.perf_counter()
    state_path = f'UK/fitting_{end_date}_{reopen_date}/{state}'
    if not os.path.exists(state_path):
        os.makedirs(state_path)

    print(state)
    print()

    # read population
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    # select confirmed and death data
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    df2 = pd.read_csv(DeathFile)
    death = df2[df2.iloc[:, 0] == state]

    # for start_date in confirmed.columns[1:]:
    # 	# if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
    # 	if confirmed.iloc[0].loc[start_date] >= I_0:
    # 		break

    days = list(confirmed.columns)
    days = days[days.index(start_date):days.index(end_date) + 1]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    confirmed = confirmed.iloc[0].loc[start_date: end_date]
    death = death.iloc[0].loc[start_date: end_date]
    for i in range(len(death)):
        if death.iloc[i] == 0:
            death.iloc[i] = 0.01
    death = death.tolist()

    reopen_day_gov = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))
    vaccine_day = days.index(datetime.datetime.strptime(vac_date, '%Y-%m-%d'))

    # fitting
    para = MT_fitting_vac(state, confirmed, death, n_0, reopen_day_gov, vaccine_day)

    [S, E, I, A, IH, IN, D, R, G, H,
     S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
     S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
     betas] = plot_vac(state, confirmed, death, days, n_0, para, state_path, vaccine_day)

    save_sim_vac([S, E, I, A, IH, IN, D, R, G, H,
                  S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
                  S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
                  betas], days, state_path)

    para[-1] = days[para[-1]]
    save_para_vac([para], state_path)
    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 60, 1)} minutes in total for {state}\n')

    return


def save_para_combined(paras, state_path):
    para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
                  'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
    df = pd.DataFrame(paras, columns=para_label)
    df.to_csv(f'{state_path}/para.csv', index=False, header=True)

    print('parameters saved\n')

    return


def save_para_vac(paras, state_path):
    para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
                  'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
    df = pd.DataFrame(paras, columns=para_label)
    df.to_csv(f'{state_path}/para.csv', index=False, header=True)

    # print('parameters saved\n')

    return


def save_sim_combined(data, days, state_path):
    days = [day.strftime('%Y-%m-%d') for day in days]
    c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
    df = pd.DataFrame(data, columns=days)
    df.insert(0, 'series', c0)
    df.to_csv(f'{state_path}/sim.csv', index=False)
    print('simulation saved\n')

    return


def save_sim_vac(data, days, state_path):
    days = [day.strftime('%Y-%m-%d') for day in days]
    c0 = ['S', 'E', 'I', 'A', 'IH', 'IN', 'D', 'R', 'G', 'H',
          'S0', 'E0', 'I0', 'A0', 'IH0', 'IN0', 'D0', 'R0', 'G0', 'H0',
          'S1', 'E1', 'I1', 'A1', 'IH1', 'IN1', 'D1', 'R1', 'G1', 'H1',
          'S2', 'E2', 'I2', 'A2', 'IH2', 'IN2', 'D2', 'R2', 'G2', 'H2',
          'S3', 'E3', 'I3', 'A3', 'IH3', 'IN3', 'D3', 'R3', 'G3', 'H3',
          'beta']
    df = pd.DataFrame(data, columns=days)
    df.insert(0, 'series', c0)
    df.to_csv(f'{state_path}/sim.csv', index=False)
    # print('simulation saved\n')

    return


def plot_combined(state, confirmed, death, days, n_0, para, state_path):
    [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1,
     metric2, r1, r2, reopen_day] = para
    para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
                  'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
    for i in range(len(para)):
        print(f'{para_label[i]}={para[i]} ', end=' ')
        if i % 4 == 1:
            print()

    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    size = len(days)
    result, [S, E, I, A, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2,
                            a3, h, Hiding_init, eta, c1, n_0, reopen_day)

    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(state)
    ax = fig.add_subplot(321)
    # ax.set_title(state)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
    ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
    ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
    ax.plot(days, [i / 1000 for i in G], label='G')
    ax2.plot(days, [i / 1000 for i in D], label='D')
    ax3.plot(days, betas, label='beta')
    ax5.plot(days, [i / 1000 for i in S], label='S')
    ax5.plot(days, [i / 1000 for i in H], label='H')
    diff_G = pd.Series(np.diff(G))
    diff_confirmed = pd.Series(np.diff(confirmed))
    ax4.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
    ax4.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')
    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    fig.autofmt_xdate()
    fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
    # fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
    plt.close(fig)
    return [S, E, I, A, IH, IN, D, R, G, H, betas]


def plot_vac(state, confirmed, death, days, n_0, para, state_path, vaccine_day):
    [beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1,
     metric2, r1, r2, reopen_day] = para
    # para_label = ['beta', 'gammaE', 'alpha', 'gamma', 'gamma2', 'gamma3', 'a1', 'a2', 'a3', 'eta', 'h', 'Hiding_init',
    #               'c1', 'I_initial', 'metric1', 'metric2', 'r1', 'r2', 'reopen']
    # for i in range(len(para)):
    # 	print(f'{para_label[i]}={para[i]} ', end=' ')
    # 	if i % 4 == 1:
    # 		print()

    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    size = len(days)
    result, [S, E, I, A, IH, IN, D, R, G, H,
             S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
             S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
             betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
                                   a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, vaccine_day, daily_vspeed,
                                   v_period1, v_period2, v_eff1, v_eff2)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(state)
    ax = fig.add_subplot(421)
    # ax.set_title(state)
    ax2 = fig.add_subplot(422)
    ax3 = fig.add_subplot(423)
    ax4 = fig.add_subplot(424)
    ax5 = fig.add_subplot(425)
    ax6 = fig.add_subplot(426)
    ax7 = fig.add_subplot(427)
    ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax4.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax5.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
    ax6.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax7.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
    ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
    ax.plot(days, [i / 1000 for i in G], label='G')
    ax2.plot(days, [i / 1000 for i in D], label='D')

    ax3.plot(days, [i / 1000 for i in S0], label='S0')
    ax3.plot(days, [i / 1000 for i in S1], label='S1')
    ax3.plot(days, [i / 1000 for i in S2], label='S2')
    ax3.plot(days, [i / 1000 for i in S3], label='S3')

    ax4.plot(days, [i / 1000 for i in H0], label='H0')
    ax4.plot(days, [i / 1000 for i in H1], label='H1')
    ax4.plot(days, [i / 1000 for i in H2], label='H2')
    ax4.plot(days, [i / 1000 for i in H3], label='H3')

    ax5.plot(days, betas, label='beta')

    ax7.plot(days, [i / 1000 for i in S], label='S')
    ax7.plot(days, [i / 1000 for i in H], label='H')

    diff_G = pd.Series(np.diff(G))
    diff_confirmed = pd.Series(np.diff(confirmed))
    ax6.plot(days[-len(diff_confirmed):], [i / 1000 for i in diff_confirmed], label='daily new cases')
    ax6.plot(days[-len(diff_G):], [i / 1000 for i in diff_G], label='dG')

    ax.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    ax5.legend()
    ax6.legend()
    ax7.legend()
    fig.autofmt_xdate()
    fig.savefig(f'{state_path}/sim.png', bbox_inches="tight")
    # fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
    plt.close(fig)
    return [S, E, I, A, IH, IN, D, R, G, H,
            S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
            S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
            betas]


def fit_all():
    t1 = time.perf_counter()
    matplotlib.use('Agg')
    # states = ['mz', 'dn_dd', 'ld']
    for state in states:
        fit_state_vac(state, 'UK/UK_confirmed.csv', 'UK/UK_death.csv', 'UK/state_population.csv')

    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 3600, 3)} hours for all states')
    return


def fit_all_vac():
    t1 = time.perf_counter()
    matplotlib.use('Agg')
    # states = ['mz', 'dn_dd', 'ld']
    with concurrent.futures.ProcessPoolExecutor() as executor:
        [executor.submit(fit_state_vac, state, 'UK/UK_confirmed.csv', 'UK/UK_death.csv', 'UK/state_population.csv') for
         state in states]

    # for state in states:
    # 	fit_state_vac(state, 'UK/UK_confirmed.csv', 'UK/UK_death.csv', 'UK/state_population.csv')

    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 3600, 3)} hours for all states')
    return


def tmp():
    l = [1, 3, 5, 7, 9, 11]
    l2 = moving_avg(l, 3)
    print(l2)

    return


def extend_state_vac(state, ConfirmFile, DeathFile, PopFile, ParaFile, sim_enddate, data_enddate):
    state_path = f'UK/extended_{sim_enddate}/{state}'
    if not os.path.exists(state_path):
        os.makedirs(state_path)
    df = pd.read_csv(ParaFile)
    beta, gammaE, alpha, gamma, gamma2, gamma3, a1, a2, a3, eta, h, Hiding_init, c1, I_initial, metric1, metric2, r1, r2, reopen_date = \
        df.iloc[0]

    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    df2 = pd.read_csv(DeathFile)
    death = df2[df2.iloc[:, 0] == state]
    dates = list(confirmed.columns)
    dates = dates[dates.index(start_date):dates.index(data_enddate) + 1]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates]
    confirmed = confirmed.iloc[0].loc[start_date: data_enddate]
    death = death.iloc[0].loc[start_date: data_enddate]

    d_confirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]
    d_confirmed.insert(0, 0)
    d_death = [death[i] - death[i - 1] for i in range(1, len(death))]
    d_death.insert(0, 0)

    reopen_day = days.index(datetime.datetime.strptime(reopen_date, '%Y-%m-%d'))
    sim_endday = days.index(datetime.datetime.strptime(sim_enddate, '%Y-%m-%d'))
    data_endday = days.index(datetime.datetime.strptime(data_enddate, '%Y-%m-%d'))
    vaccine_day = days.index(datetime.datetime.strptime(vac_date, '%Y-%m-%d'))

    S = [n_0 * eta * (1 - Hiding_init)]
    E = [0]
    I = [n_0 * eta * I_initial * (1 - alpha)]
    A = [n_0 * eta * I_initial * alpha]
    IH = [0]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [n_0 * eta * Hiding_init]
    # H = [0]
    size = len(days)

    result, [S, E, I, A, IH, IN, D, R, G, H,
             S0, E0, I0, A0, IH0, IN0, D0, R0, G0, H0, S1, E1, I1, A1, IH1, IN1, D1, R1, G1, H1,
             S2, E2, I2, A2, IH2, IN2, D2, R2, G2, H2, S3, E3, I3, A3, IH3, IN3, D3, R3, G3, H3,
             betas] = simulate_vac(size, S, E, I, A, IH, IN, D, R, G, H, beta, gammaE, alpha, gamma, gamma2, gamma3, a1,
                                   a2, a3, h, Hiding_init, eta, c1, n_0, reopen_day, vaccine_day, daily_vspeed,
                                   v_period1, v_period2, v_eff1, v_eff2)

    dG = [G[i] - G[i - 1] for i in range(1, len(G))]
    dG.insert(0, 0)
    dD = [D[i] - D[i - 1] for i in range(1, len(D))]
    dD.insert(0, 0)

    # if sim_endday + 7 <= data_endday:
    #     error_ratio = 0
    #     for i in range(sim_endday + 1, sim_endday + 8):
    #         error_ratio += (abs(dG[i] - d_confirmed[i]) / d_confirmed[i])
    #     error_ratio /= 7
    #     print(f'{state} 7-day error ratio from {sim_enddate}={round(error_ratio, 4)}')
    #
    # if sim_endday + 14 <= data_endday:
    #     error_ratio = 0
    #     for i in range(sim_endday + 1, sim_endday + 15):
    #         error_ratio += (abs(dG[i] - d_confirmed[i]) / d_confirmed[i])
    #     error_ratio /= 14
    #     print(f'{state} 14-day error ratio from {sim_enddate}={round(error_ratio, 4)}')

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(state)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label='reopen')
    ax.axvline(days[sim_endday], linestyle='dashed', color='tab:red', label=sim_enddate)
    ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
    ax.plot(days, [i / 1000 for i in G], label='G')

    # ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label='reopen')
    ax2.axvline(days[sim_endday], linestyle='dashed', color='tab:red', label=sim_enddate)
    ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
    ax2.plot(days, [i / 1000 for i in D], label='D')

    ax.legend()
    ax2.legend()

    fig.autofmt_xdate()
    fig.savefig(f'{state_path}/ext.png', bbox_inches="tight")
    plt.close(fig)
    return G, D, dG, dD, confirmed, death, d_confirmed, d_death, days


def extend_all_vac(sim_enddate):
    matplotlib.use('Agg')

    # sim_enddate = '2021-06-15'
    data_enddate = '2021-06-25'
    sim_releasedate = '2021-04-25'
    sim_folder = f'UK/fitting_{sim_enddate}_{sim_releasedate}'
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    # 	results = [executor.submit(extend_state_vac, state, 'UK/UK_confirmed.csv', 'UK/UK_death.csv',
    # 	                           'UK/state_population.csv', f'{sim_folder}/{state}/para.csv', sim_enddate,
    # 	                           data_enddate) for state in states]
    UK_G = []
    UK_D = []
    UK_dG = []
    UK_dD = []
    UK_confirmed = []
    UK_death = []
    UK_d_confirmed = []
    UK_d_death = []
    for state in states:
        G, D, dG, dD, confirmed, death, d_confirmed, d_death, days = \
            extend_state_vac(state, 'UK/UK_confirmed.csv', 'UK/UK_death.csv', 'UK/state_population.csv',
                             f'{sim_folder}/{state}/para.csv', sim_enddate, data_enddate)

        if len(UK_G) == 0:
            UK_G = G.copy()
            UK_dG = dG.copy()
            UK_D = D.copy()
            UK_dD = dD.copy()
            UK_confirmed = confirmed.copy()
            UK_d_confirmed = d_confirmed.copy()
            UK_death = death.copy()
            UK_d_death = d_death.copy()
        else:
            UK_G = [UK_G[i] + G[i] for i in range(len(G))]
            UK_dG = [UK_dG[i] + dG[i] for i in range(len(G))]
            UK_D = [UK_D[i] + D[i] for i in range(len(G))]
            UK_dD = [UK_dD[i] + dD[i] for i in range(len(G))]
            UK_confirmed = [UK_confirmed[i] + confirmed[i] for i in range(len(G))]
            UK_d_confirmed = [UK_d_confirmed[i] + d_confirmed[i] for i in range(len(G))]
            UK_death = [UK_death[i] + death[i] for i in range(len(G))]
            UK_d_death = [UK_d_death[i] + d_death[i] for i in range(len(G))]

    sim_endday = days.index(datetime.datetime.strptime(sim_enddate, '%Y-%m-%d'))
    data_endday = days.index(datetime.datetime.strptime(data_enddate, '%Y-%m-%d'))

    if sim_endday + 7 <= data_endday:
        error_ratio = 0
        MA_UK_dG = moving_avg(UK_dG, 7)
        MA_UK_d_confirmed = moving_avg(UK_d_confirmed, 7)
        for i in range(sim_endday + 1, sim_endday + 8):
            error_ratio += (abs(MA_UK_dG[i] - MA_UK_d_confirmed[i]) / MA_UK_d_confirmed[i])
        error_ratio /= 7
        print(f'UK  7-day error ratio from {sim_enddate} = {round(error_ratio, 4)}')

    if sim_endday + 14 <= data_endday:
        error_ratio = 0
        MA_UK_dG = moving_avg(UK_dG, 7)
        MA_UK_d_confirmed = moving_avg(UK_d_confirmed, 7)
        for i in range(sim_endday + 1, sim_endday + 15):
            error_ratio += (abs(MA_UK_dG[i] - MA_UK_d_confirmed[i]) / MA_UK_d_confirmed[i])
        error_ratio /= 14
        print(f'UK 14-day error ratio from {sim_enddate} = {round(error_ratio, 4)}')

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle('UK')
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label='reopen')
    ax.axvline(days[sim_endday], linestyle='dashed', color='tab:red', label=sim_enddate)
    ax.plot(days, [i / 1000 for i in UK_confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
    ax.plot(days, [i / 1000 for i in UK_G], label='G')

    # ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label='reopen')
    ax2.axvline(days[sim_endday], linestyle='dashed', color='tab:red', label=sim_enddate)
    ax2.plot(days, [i / 1000 for i in UK_death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
    ax2.plot(days, [i / 1000 for i in UK_D], label='D')

    ax.legend()
    ax2.legend()

    fig.autofmt_xdate()
    fig.savefig(f'UK/extended_{sim_enddate}/UK_ext.png', bbox_inches="tight")
    plt.close(fig)

    return


def save_para_all_vac():
    fitting_folder = f'UK/fitting_{end_date}'
    out_table = []
    for state in states:
        df = pd.read_csv(f'{fitting_folder}/{state}/para.csv')
        cols = df.columns
        row = list(df.iloc[0])
        row.insert(0, state)
        out_table.append(row)
    cols = list(cols)
    cols.insert(0, 'state')
    out_df = pd.DataFrame(out_table, columns=cols)
    out_df.to_csv(f'{fitting_folder}/paras.csv', index=False)
    return


def moving_avg(original_list, days):
    MA_list = pd.Series(original_list)
    MA_list = MA_list.rolling(days).mean()
    return list(MA_list)


def main():
    # fit_all_vac()
    # save_para_all_vac()
    extend_all_vac('2021-06-08')
    extend_all_vac('2021-06-15')
    # tmp()
    return


if __name__ == '__main__':
    main()
