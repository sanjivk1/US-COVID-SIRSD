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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from SIRfunctions import SIRG_combined, SIRG, weighting, SEIRG_sd, SEIRG
import datetime
from numpy.random import uniform as uni
import os
import warnings
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# matplotlib.use('Agg')

np.set_printoptions(threshold=sys.maxsize)
Geo = 0.98

num_threads = 200
# num_threads = 5
num_CI = 1000
# num_CI = 5

num_threads_dist = 0

# weight of G in initial fitting
theta = 0.7
# weight of G in release fitting
theta2 = 0.8

I_0 = 5
beta_range = (0.1, 100)
beta_SEIR_range = (0.1, 50)
beta_SEIR_SD_range = (0.1, 100)
betaEI_range = (0.1, 0.5)
gamma_range = (0.04, 0.2)
gamma2_range = (0.04, 0.2)
# sigma_range = (0.001, 1)
a1_range = (0.01, 0.5)
a2_range = (0.01, 0.2)
a3_range = (0.01, 0.2)
eta_range = (0.001, 0.05)
c1_fixed = (0.9, 0.9)
c1_range = (0.9, 1)
h_range = (0, 10)
k_range = (0.1, 2)
k2_range = (0.1, 2)
Geo_range = np.arange(0.98, 1.01, 0.05)
theta_range = np.arange(1, 1.01, 0.05)
# end_date = '2020-06-10'
# end_date = '2020-08-16'
# end_date = '2020-09-23'
# end_date = '2020-09-22'
end_date = '2020-08-31'
release_duration = 30
k_drop = 14
p_m = 1
# Hiding = 0.33
delay = 7
change_eta2 = False

fig_row = 5
fig_col = 3


# save simulation of SIRG fitting to csv for initial phase only
def save_sim_init(csv_filename, data, days):
    days = [day.strftime('%Y-%m-%d') for day in days]
    c0 = ['S', 'I', 'IH', 'IN', 'D', 'R', 'G', 'H', 'beta']
    df = pd.DataFrame(data, columns=days)
    df.insert(0, 'series', c0)
    df.to_csv(csv_filename, index=False)
    print('simulation saved\n')


# save the parameters distribution to CSV for initial phase only
def save_para_init(state, paras, date):
    para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
                  'metric2', 'r1', 'r2']
    df = pd.DataFrame(paras, columns=para_label)
    df.to_csv(f'50Counties/init_only_{date}/{state}/para.csv', index=False, header=True)
    # df.to_csv(f'init_only_{end_date}/{state}/para.csv', index=False, header=True)
    print('parameters saved\n')


# simulate combined phase
def simulate_combined(size, SIRG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2, eta, c1, n_0,
                      reopen_day):
    result = True
    H0 = H[0]
    eta2 = eta
    kk = 1
    kk2 = 1
    r = h * H0
    betas = [beta]
    for i in range(1, size):

        if i > reopen_day:
            kk = k
            kk2 = k2
            release = min(H[-1], r * funcmod(i))
            S[-1] += release
            H[-1] -= release
        delta = SIRG(i, [S[i - 1], I[i - 1], IH[i - 1], IN[i - 1], D[i - 1], R[i - 1], G[i - 1], beta, kk * gamma,
                         gamma2, a1, kk2 * a2, a3, eta2, n_0, c1, H[-1], H0])
        S.append(S[-1] + delta[0])
        I.append(I[-1] + delta[1])
        IH.append(IH[-1] + delta[2])
        IN.append(IN[-1] + delta[3])
        D.append(D[-1] + delta[4])
        R.append(R[-1] + delta[5])
        G.append(G[-1] + delta[6])
        H.append(H[-1])
        betas.append(delta[7])
        if S[-1] < 0:
            result = False
            break
    return result, [S, I, IH, IN, D, R, G, H, betas]


# initial phase fitting
def loss_init(point, c1, confirmed, death, n_0, SIRG, reopen_day):
    size = len(confirmed)
    beta = point[0]
    gamma = point[1]
    gamma2 = point[2]
    a1 = point[3]
    a2 = point[4]
    a3 = point[5]
    eta = point[6]
    h = point[7]
    Hiding_init = point[8]
    k = point[9]
    k2 = point[10]
    S = [n_0 * eta]
    I = [confirmed[0]]
    IH = [I[-1] * gamma]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [Hiding_init * eta * n_0]
    result, [S, I, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, SIRG, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2, eta, c1,
                            n_0, reopen_day)

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

    # weights = [Geo ** n for n in range(size)]
    # weights.reverse()
    weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
    weighted_G = [G[i] * weights[i] for i in range(size)]
    weighted_death = [death[i] * weights[i] for i in range(size)]
    weighted_D = [D[i] * weights[i] for i in range(size)]
    # weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
    # weighted_IH = [IH[i] * weights[i] for i in range(size)]

    metric0 = r2_score(weighted_confirmed, weighted_G)
    metric1 = r2_score(weighted_death, weighted_D)
    # return -(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)
    return -(0.99 * metric0 + 0.01 * metric1)


def fit_init(confirmed0, death0, days, reopen_day, n_0, metric1, metric2):
    np.random.seed()
    confirmed = confirmed0.copy()
    death = death0.copy()
    size = len(confirmed)
    if metric2 != 0 or metric1 != 0:
        scale1 = pd.Series(np.random.normal(1, metric1, size))
        confirmed = [max(confirmed[i] * scale1[i], 1) for i in range(size)]
        scale2 = pd.Series(np.random.normal(1, metric2, size))
        death = [max(death[i] * scale2[i], 1) for i in range(size)]

    c_max = 0
    min_loss = 10000
    for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
        # optimal = minimize(loss, [10, 0.05, 0.01, 0.1, 0.1, 0.1, 0.02], args=(c1, confirmed, death, n_0, SIDRG_sd),
        optimal = minimize(loss_init, [uni(beta_range[0], beta_range[1]),
                                       uni(gamma_range[0], gamma_range[1]),
                                       uni(gamma2_range[0], gamma2_range[1]),
                                       uni(a1_range[0], a1_range[1]),
                                       uni(a2_range[0], a2_range[1]),
                                       uni(a3_range[0], a3_range[1]),
                                       uni(eta_range[0], eta_range[1]),
                                       uni(0, 1 / 14),
                                       0.5,
                                       uni(k_range[0], k_range[1]),
                                       uni(k2_range[0], k2_range[1])],
                           args=(c1, confirmed, death, n_0, SIRG_combined, reopen_day), method='L-BFGS-B',
                           bounds=[beta_range,
                                   gamma_range,
                                   gamma2_range,
                                   a1_range,
                                   a2_range,
                                   a3_range,
                                   eta_range,
                                   (0, 1 / 14),
                                   (0, 5),
                                   k_range,
                                   k2_range])
        current_loss = loss_init(optimal.x, c1, confirmed, death, n_0, SIRG_combined, reopen_day)
        if current_loss < min_loss:
            # print(f'updating loss={current_loss} with c1={c1}')
            min_loss = current_loss
            c_max = c1
            beta = optimal.x[0]
            gamma = optimal.x[1]
            gamma2 = optimal.x[2]
            a1 = optimal.x[3]
            a2 = optimal.x[4]
            a3 = optimal.x[5]
            eta = optimal.x[6]
            h = optimal.x[7]
            Hiding_init = optimal.x[8]
            k = optimal.x[9]
            k2 = optimal.x[10]

    c1 = c_max
    S = [n_0 * eta]
    I = [confirmed[0]]
    IH = [I[-1] * gamma]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [Hiding_init * n_0 * eta]
    # Betas = [beta]

    result, [S, I, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
                            eta, c1, n_0, reopen_day)

    data1 = [(confirmed[i] - G[i]) / confirmed[i] for i in range(size)]
    data2 = [(death[i] - D[i]) / death[i] for i in range(size)]

    # metric1 = math.sqrt(sum([i ** 2 for i in data1]) / (len(data1) - 8))
    # metric2 = math.sqrt(sum([i ** 2 for i in data2]) / (len(data2) - 8))

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

    sum_wt = sum(weights)
    metric1 = math.sqrt(sum([data1[i] ** 2 * weights[i] for i in range(size)])
                        /
                        ((size - 12) * sum_wt / size)
                        )
    metric2 = math.sqrt(sum([data2[i] ** 2 * weights[i] for i in range(size)])
                        /
                        ((size - 12) * sum_wt / size)
                        )

    r1 = r2_score(confirmed, G)
    r2 = r2_score(death, D)

    return [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2], min_loss


def funcmod(i):
    # return 0.5 * np.log(1 + i)
    # return 1.00 * np.power(i, -0.4)
    return 1


# fit with SD for initial phase only
def fit_state_init(state, ConfirmFile, DeathFile, PopFile, date):
    t1 = time.perf_counter()

    # if not os.path.exists(f'JHU/combined2W_{end_date}/{state}'):
    # 	os.makedirs(f'JHU/combined2W_{end_date}/{state}')
    if not os.path.exists(f'50Counties/init_only_{date}/{state}'):
        os.makedirs(f'50Counties/init_only_{date}/{state}')

    # # add the delay in dates
    # for i in range(len(dates) - 1):
    # 	date = datetime.datetime.strptime(dates[i], '%Y-%m-%d')
    # 	date += datetime.timedelta(days=delay)
    # 	dates[i] = date.strftime('%Y-%m-%d')
    print(state)
    print(date)
    print()

    # read population
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    # select confirmed and death data
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    df2 = pd.read_csv(DeathFile)
    death = df2[df2.iloc[:, 0] == state]
    for start_date in confirmed.columns[1:]:
        # if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
        if confirmed.iloc[0].loc[start_date] >= I_0:
            break
    days = list(confirmed.columns)
    # days = days[days.index(start_date):days.index(end_date) + 1]
    days = days[days.index(start_date):days.index(date) + 1]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    # confirmed = confirmed.iloc[0].loc[start_date: end_date]
    confirmed = confirmed.iloc[0].loc[start_date: date]
    # death = death.iloc[0].loc[start_date: end_date]
    death = death.iloc[0].loc[start_date: date]
    for i in range(len(death)):
        if death.iloc[i] == 0:
            death.iloc[i] = 0.01
    death = death.tolist()

    reopen_day = days.index(datetime.datetime.strptime(date, '%Y-%m-%d'))

    # fitting
    para = MT_init(confirmed, death, n_0, days, reopen_day)
    [S, I, IH, IN, D, R, G, H, betas] = plot_init(state, confirmed, death, days, n_0, reopen_day, para, date)
    # csv_file = f'JHU/combined2W_{end_date}/{state}/sim.csv'
    csv_file = f'50Counties/init_only_{date}/{state}/sim.csv'
    save_sim_init(csv_file, [S, I, IH, IN, D, R, G, H, betas], days)
    save_para_init(state, [para], date)
    t2 = time.perf_counter()
    print(f'{round(t2 - t1, 3)} seconds in total for {state}\n')

    return


# plot result for initial phase only
def plot_init(state, confirmed, death, days, n_0, reopen_day, para, date):
    [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2] = para
    para_label = ['beta', 'gamma', 'gamma2', 'a1', 'a2', 'a3', 'h', 'Hiding_init', 'k', 'k2', 'eta', 'c1', 'metric1',
                  'metric2', 'r1', 'r2']
    for i in range(len(para)):
        print(f'{para_label[i]}={para[i]} ', end=' ')
        if i % 4 == 1:
            print()

    S = [n_0 * eta]
    I = [confirmed[0]]
    IH = [I[-1] * gamma]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [Hiding_init * n_0 * eta]
    size = len(days)
    result, [S, I, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
                            eta, c1, n_0, reopen_day)

    # weights = [Geo ** n for n in range(size)]
    # weights.reverse()
    # weighted_confirmed = [confirmed[i] * weights[i] for i in range(size)]
    # weighted_G = [G[i] * weights[i] for i in range(size)]
    # weighted_death = [death[i] * weights[i] for i in range(size)]
    # weighted_D = [D[i] * weights[i] for i in range(size)]
    # # weighted_hosp = [hosp[i] * weights[i] for i in range(size)]
    # # weighted_IH = [IH[i] * weights[i] for i in range(size)]
    #
    # confirmed_derivative = np.diff(confirmed)
    # G_derivative = np.diff(G)
    # confirmed_derivative = [confirmed_derivative[i] * weights[i] for i in range(size - 1)]
    # G_derivative = [G_derivative[i] * weights[i] for i in range(size - 1)]
    # alpha = 0.5
    # metric00 = r2_score(weighted_confirmed, weighted_G)
    # metric01 = r2_score(confirmed_derivative, G_derivative)
    # # metric0 = (alpha * metric00 + (1 - alpha) * metric01)
    #
    # weighted_hosp = hosp
    # weighted_IH = IH
    #
    # metric0 = r2_score(weighted_confirmed, weighted_G)
    # metric1 = r2_score(weighted_death, weighted_D)
    # metric2 = r2_score(weighted_hosp, weighted_IH)
    # print(f'\nloss={-(theta * metric0 + 1 * (1 - theta) / 2 * metric1 + 1 * (1 - theta) / 2 * metric2)}')

    fig = plt.figure(figsize=(6, 10))
    # fig.suptitle(state)
    ax = fig.add_subplot(311)
    ax.set_title(state)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    ax.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax2.axvline(days[reopen_day], linestyle='dashed', color='tab:grey')
    ax3.axvline(days[reopen_day], linestyle='dashed', color='tab:grey', label=days[reopen_day].strftime('%Y-%m-%d'))
    ax.plot(days, [i / 1000 for i in confirmed], linewidth=5, linestyle=':', label="Cumulative\nCases")
    ax2.plot(days, [i / 1000 for i in death], linewidth=5, linestyle=':', label="Cumulative\nDeaths")
    ax.plot(days, [i / 1000 for i in G], label='G')
    ax2.plot(days, [i / 1000 for i in D], label='D')
    ax3.plot(days, betas, label='beta')
    # ax.plot(days, [i / 1000 for i in H], label='H')
    ax.legend()
    ax2.legend()
    ax3.legend()
    fig.autofmt_xdate()
    fig.savefig(f'50Counties/init_only_{date}/{state}/sim.png', bbox_inches="tight")
    # fig.savefig(f'init_only_{end_date}/{state}/sim.png', bbox_inches="tight")
    plt.close(fig)
    return [S, I, IH, IN, D, R, G, H, betas]


# read hospitalization data
def readHosp(state):
    df = pd.read_csv(f'JHU/hosp/{state}/Top_Counties_by_Cases_Full_Data_data.csv', usecols=['Date', 'Hospitalization'])
    df = df[df['Hospitalization'].notna()]

    dates = [datetime.datetime.strptime(date, '%m/%d/%Y') for date in df['Date']]
    dates = [date.strftime('%Y-%m-%d') for date in dates]

    df['Date'] = dates
    df = df.sort_values('Date')
    df = df.drop_duplicates()
    df = df.T
    df.columns = df.iloc[0]
    df = df.iloc[1]
    return df


# combined fitting for initial phase only
def MT_init(confirmed, death, n_0, days, reopen_day):
    para_best = []
    min_loss = 10000
    with concurrent.futures.ProcessPoolExecutor() as executor:
        t1 = time.perf_counter()
        results = [executor.submit(fit_init, confirmed, death, days, reopen_day, n_0, 0, 0) for _ in
                   range(num_threads)]

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


# fit all state with SD for initial phase only
def fit_50_init(date='2020-05-15'):
    t1 = time.perf_counter()
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King']
    # states = ['NY-New York']
    # date = '2020-05-15'
    for state in states:
        fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
                       'JHU/CountyPopulation.csv', date)

    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')

    return


# fit all state with SD for initial phase only
def fit_50more_init(date='2020-05-15'):
    t1 = time.perf_counter()
    states = ['PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside', 'PA-Delaware', 'CA-San Diego',
              'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin', 'WI-Milwaukee', 'CO-Denver',
              'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis', 'IL-Will', 'TX-Tarrant',
              'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson', 'UT-Salt Lake', 'GA-Fulton',
              'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex', 'NY-Dutchess',
              'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb',
              'IN-Lake', 'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    # date = '2020-05-15'
    for state in states:
        fit_state_init(state, 'JHU/JHU_Confirmed-counties.csv', 'JHU/JHU_Death-counties.csv',
                       'JHU/CountyPopulation.csv', date)

    t2 = time.perf_counter()
    print(f'{round((t2 - t1) / 60, 3)} minutes for all counties')

    return


def test():
    r1 = [1, 2, 3]
    r2 = [4, 5, 6]
    col = ['c2', 'c3', 'c4']
    c1 = [0, 5]
    df = pd.DataFrame([r1, r2], columns=col)
    print(df)
    df = df.iloc[0].loc[:'c3']
    # df.insert(0, 'c1', c1)
    print(df)
    df.to_csv('index.csv', index=True)
    df.to_csv('no_index.csv', index=False)


def fit_50_SIR(date='2020-05-15'):
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King']
    # states = ['MI-Wayne']
    # date = '2020-05-15'
    for state in states:
        fit_SIR_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                   'JHU/CountyPopulation.csv', date)

    return


def fit_50_SEIR_SD(date='2020-05-15'):
    plt.switch_backend('agg')
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King']
    # states = ['AZ-Maricopa']
    # date = '2020-05-15'
    for state in states:
        fit_SEIR_SD_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                       'JHU/CountyPopulation.csv', date)

    return


def fit_50_SEIR(date='2020-05-15'):
    plt.switch_backend('agg')
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King']
    # states = ['AZ-Maricopa']
    # date = '2020-05-15'
    for state in states:
        fit_SEIR_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                    'JHU/CountyPopulation.csv', date)

    return


def fit_50more_SIR(date='2020-05-15'):
    plt.switch_backend('agg')
    states = ['PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside', 'PA-Delaware', 'CA-San Diego',
              'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin', 'WI-Milwaukee', 'CO-Denver',
              'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis', 'IL-Will', 'TX-Tarrant',
              'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson', 'UT-Salt Lake', 'GA-Fulton',
              'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex', 'NY-Dutchess',
              'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb',
              'IN-Lake', 'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    # date = '2020-05-15'
    for state in states:
        fit_SIR_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                   'JHU/CountyPopulation.csv', date)

    return


def fit_50more_SEIR_SD(date='2020-05-15'):
    plt.switch_backend('agg')
    states = ['PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside', 'PA-Delaware', 'CA-San Diego',
              'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin', 'WI-Milwaukee', 'CO-Denver',
              'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis', 'IL-Will', 'TX-Tarrant',
              'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson', 'UT-Salt Lake', 'GA-Fulton',
              'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex', 'NY-Dutchess',
              'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb',
              'IN-Lake', 'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    # date = '2020-05-15'
    for state in states:
        fit_SEIR_SD_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                       'JHU/CountyPopulation.csv', date)

    return


def fit_50more_SEIR(date='2020-05-15'):
    plt.switch_backend('agg')
    states = ['PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside', 'PA-Delaware', 'CA-San Diego',
              'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin', 'WI-Milwaukee', 'CO-Denver',
              'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis', 'IL-Will', 'TX-Tarrant',
              'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson', 'UT-Salt Lake', 'GA-Fulton',
              'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex', 'NY-Dutchess',
              'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb',
              'IN-Lake', 'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    # date = '2020-05-15'
    for state in states:
        fit_SEIR_MT(state, f'50Counties/init_only_{end_date}/{state}', 'JHU/JHU_Confirmed-counties.csv',
                    'JHU/CountyPopulation.csv', date)

    return


def fit_SEIR_SD_MT(state, SimFolder, ConfirmFile, PopFile, date):
    if not os.path.exists(f'50Counties/SEIR_SD_{date}/{state}'):
        os.makedirs(f'50Counties/SEIR_SD_{date}/{state}')

    print(state)

    S, I, IH, IN, R, D, G, H, days = read_sim(state, SimFolder)
    start_date = days[0]
    reopen_date = date

    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
    # print(confirmed)
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    para_best = []
    min_loss = 1000001
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        results = [executor.submit(fit_SEIR_SD, confirmed, n_0) for _ in range(num_threads)]
        threads = 0
        for f in concurrent.futures.as_completed(results):
            threads += 1
            current_loss, para = f.result()
            if current_loss < min_loss:
                print(f'updated at {threads}')
                min_loss = current_loss
                para_best = para
    # optimal = minimize(loss, [10, 0.05, 0.02], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
    #                    bounds=[beta_range, gamma_range, eta_range])

    beta = para_best[0]
    betaEI = para_best[1]
    gamma = para_best[2]
    eta = para_best[3]
    theta = para_best[4]
    Geo = para_best[5]
    c1 = para_best[6]
    size = len(confirmed)
    S = [n_0 * eta]
    E = [0]
    I = [confirmed.iloc[0]]
    R = [0]
    G = [confirmed.iloc[0]]
    for i in range(1, size):
        delta = SEIRG_sd(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        R.append(R[-1] + delta[3])
        G.append(G[-1] + delta[4])

    # print('beta:', beta_range[0], beta, beta_range[1])
    # print('gamma:', gamma_range[0], gamma, gamma_range[1])
    # print('eta:', eta_range[0], eta, eta_range[1])

    # save simulation
    c0 = ['S', 'E', 'I', 'R', 'G']
    df = pd.DataFrame([S, E, I, R, G], columns=days[:len(S)])
    df.insert(0, 'series', c0)
    df.to_csv(f'50Counties/SEIR_SD_{date}/{state}/sim.csv', index=False)

    # save parameters
    para_label = ['beta', 'betaEI', 'gamma', 'eta', 'theta', 'Geo', 'c1', 'RMSE', 'R2']
    RMSE = math.sqrt(mean_squared_error(confirmed, G))
    r2 = r2_score(confirmed, G)
    para_best = np.append(para_best, RMSE)
    para_best = np.append(para_best, r2)
    # para_best.append(RMSE)
    df = pd.DataFrame([para_best], columns=para_label)
    df.to_csv(f'50Counties/SEIR_SD_{date}/{state}/para.csv', index=False)

    days = days[:size]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(state)
    ax.plot(days, G, label='G')
    ax.plot(days, confirmed, label='confirm')
    fig.autofmt_xdate()
    ax.legend()
    fig.savefig(f'50Counties/SEIR_SD_{date}/{state}/fit.png', bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    return


def fit_SEIR_MT(state, SimFolder, ConfirmFile, PopFile, date):
    if not os.path.exists(f'50Counties/SEIR_{date}/{state}'):
        os.makedirs(f'50Counties/SEIR_{date}/{state}')

    print(state)

    S, I, IH, IN, R, D, G, H, days = read_sim(state, SimFolder)
    start_date = days[0]
    reopen_date = date

    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
    # print(confirmed)
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    para_best = []
    min_loss = 1000001
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        results = [executor.submit(fit_SEIR, confirmed, n_0) for _ in range(num_threads)]
        threads = 0
        for f in concurrent.futures.as_completed(results):
            threads += 1
            current_loss, para = f.result()
            if current_loss < min_loss:
                print(f'updated at {threads}')
                min_loss = current_loss
                para_best = para
    # optimal = minimize(loss, [10, 0.05, 0.02], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
    #                    bounds=[beta_range, gamma_range, eta_range])

    beta = para_best[0]
    betaEI = para_best[1]
    gamma = para_best[2]
    eta = para_best[3]
    theta = para_best[4]
    Geo = para_best[5]
    # c1 = para_best[6]
    size = len(confirmed)
    S = [n_0 * eta]
    E = [0]
    I = [confirmed.iloc[0]]
    R = [0]
    G = [confirmed.iloc[0]]
    for i in range(1, size):
        delta = SEIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        R.append(R[-1] + delta[3])
        G.append(G[-1] + delta[4])

    # print('beta:', beta_range[0], beta, beta_range[1])
    # print('gamma:', gamma_range[0], gamma, gamma_range[1])
    # print('eta:', eta_range[0], eta, eta_range[1])

    # save simulation
    c0 = ['S', 'E', 'I', 'R', 'G']
    df = pd.DataFrame([S, E, I, R, G], columns=days[:len(S)])
    df.insert(0, 'series', c0)
    df.to_csv(f'50Counties/SEIR_{date}/{state}/sim.csv', index=False)

    # save parameters
    para_label = ['beta', 'betaEI', 'gamma', 'eta', 'theta', 'Geo', 'RMSE', 'R2']
    RMSE = math.sqrt(mean_squared_error(confirmed, G))
    r2 = r2_score(confirmed, G)
    para_best = np.append(para_best, RMSE)
    para_best = np.append(para_best, r2)
    # para_best.append(RMSE)
    df = pd.DataFrame([para_best], columns=para_label)
    df.to_csv(f'50Counties/SEIR_{date}/{state}/para.csv', index=False)

    days = days[:size]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(state)
    ax.plot(days, G, label='G')
    ax.plot(days, confirmed, label='confirm')
    fig.autofmt_xdate()
    ax.legend()
    fig.savefig(f'50Counties/SEIR_{date}/{state}/fit.png', bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    return


def fit_SIR_MT(state, SimFolder, ConfirmFile, PopFile, date):
    if not os.path.exists(f'50Counties/SIR_{date}/{state}'):
        os.makedirs(f'50Counties/SIR_{date}/{state}')

    print(state)

    S, I, IH, IN, R, D, G, H, days = read_sim(state, SimFolder)
    start_date = days[0]
    reopen_date = date

    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    confirmed = confirmed.iloc[0].loc[start_date:reopen_date]
    # print(confirmed)
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    para_best = []
    min_loss = 1000001
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(fit_SIR, confirmed, n_0) for _ in range(num_threads)]
        threads = 0
        for f in concurrent.futures.as_completed(results):
            threads += 1
            current_loss, para = f.result()
            if current_loss < min_loss:
                print(f'updated at {threads}')
                min_loss = current_loss
                para_best = para
    # optimal = minimize(loss, [10, 0.05, 0.02], args=(confirmed, n_0, SIRG), method='L-BFGS-B',
    #                    bounds=[beta_range, gamma_range, eta_range])

    beta = para_best[0]
    gamma = para_best[1]
    eta = para_best[2]
    theta = para_best[3]
    Geo = para_best[4]
    size = len(confirmed)
    S = [n_0 * eta]
    I = [confirmed[0]]
    R = [0]
    G = [confirmed[0]]
    for i in range(1, size):
        delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
        S.append(S[-1] + delta[0])
        I.append(I[-1] + delta[1])
        R.append(R[-1] + delta[2])
        G.append(G[-1] + delta[3])

    # print('beta:', beta_range[0], beta, beta_range[1])
    # print('gamma:', gamma_range[0], gamma, gamma_range[1])
    # print('eta:', eta_range[0], eta, eta_range[1])

    # save simulation
    c0 = ['S', 'I', 'R', 'G']
    df = pd.DataFrame([S, I, R, G], columns=days[:len(S)])
    df.insert(0, 'series', c0)
    df.to_csv(f'50Counties/SIR_{date}/{state}/sim.csv', index=False)

    # save parameters
    para_label = ['beta', 'gamma', 'eta', 'theta', 'Geo', 'RMSE', 'R2']
    RMSE = math.sqrt(mean_squared_error(confirmed, G))
    r2 = r2_score(confirmed, G)
    para_best = np.append(para_best, RMSE)
    para_best = np.append(para_best, r2)
    # para_best.append(RMSE)
    df = pd.DataFrame([para_best], columns=para_label)
    df.to_csv(f'50Counties/SIR_{date}/{state}/para.csv', index=False)

    days = days[:size]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(state)
    ax.plot(days, G, label='G')
    ax.plot(days, confirmed, label='confirm')
    fig.autofmt_xdate()
    ax.legend()
    fig.savefig(f'50Counties/SIR_{date}/{state}/fit.png', bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    return


def read_sim(state, SimFolder):
    # ParaFile = f'{SimFolder}/para.csv'
    SimFile = f'{SimFolder}/sim.csv'
    df = pd.read_csv(SimFile)
    S = df[df['series'] == 'S'].iloc[0].iloc[1:]
    I = df[df['series'] == 'I'].iloc[0].iloc[1:]
    IH = df[df['series'] == 'IH'].iloc[0].iloc[1:]
    IN = df[df['series'] == 'IN'].iloc[0].iloc[1:]
    R = df[df['series'] == 'R'].iloc[0].iloc[1:]
    D = df[df['series'] == 'D'].iloc[0].iloc[1:]
    G = df[df['series'] == 'G'].iloc[0].iloc[1:]
    H = df[df['series'] == 'H'].iloc[0].iloc[1:]
    days = S.index

    return S, I, IH, IN, R, D, G, H, days


def fit_SIR(confirmed, n_0):
    # S, I, R, G, days = read_sim(state, SimFolder)
    # start_date = days[0]
    # # reopen_date = dates[0]
    np.random.seed()

    para_best = []
    min_loss = 1000001
    for Geo in Geo_range:
        for theta in theta_range:
            optimal = minimize(loss_SIR, [uni(beta_range[0], beta_range[1]),
                                          uni(gamma_range[0], gamma_range[1]),
                                          uni(eta_range[0], eta_range[1])],
                               args=(confirmed, n_0, SIRG, theta, Geo), method='L-BFGS-B',
                               bounds=[beta_range, gamma_range, eta_range])
            current_loss = loss_SIR(optimal.x, confirmed, n_0, SIRG, theta, Geo)
            if current_loss < min_loss:
                min_loss = current_loss
                [beta, gamma, eta] = optimal.x
                para_best = [beta, gamma, eta, theta, Geo]

    return current_loss, para_best


def fit_SEIR_SD(confirmed, n_0):
    # S, I, R, G, days = read_sim(state, SimFolder)
    # start_date = days[0]
    # # reopen_date = dates[0]
    np.random.seed()

    para_best = []
    min_loss = 1000001
    for c1 in np.arange(c1_range[0], c1_range[1], 0.01):
        for Geo in Geo_range:
            for theta in theta_range:
                optimal = minimize(loss_SEIR_SD, [uni(beta_SEIR_SD_range[0], beta_SEIR_SD_range[1]),
                                                  uni(betaEI_range[0], betaEI_range[1]),
                                                  uni(gamma_range[0], gamma_range[1]),
                                                  uni(eta_range[0], eta_range[1])],
                                   args=(confirmed, n_0, SEIRG_sd, theta, Geo, c1), method='L-BFGS-B',
                                   bounds=[beta_SEIR_SD_range, betaEI_range, gamma_range, eta_range])
                current_loss = loss_SEIR_SD(optimal.x, confirmed, n_0, SEIRG_sd, theta, Geo, c1)
                if current_loss < min_loss:
                    min_loss = current_loss
                    [beta, betaEI, gamma, eta] = optimal.x
                    para_best = [beta, betaEI, gamma, eta, theta, Geo, c1]

    return current_loss, para_best


def fit_SEIR(confirmed, n_0):
    # S, I, R, G, days = read_sim(state, SimFolder)
    # start_date = days[0]
    # # reopen_date = dates[0]
    np.random.seed()

    para_best = []
    min_loss = 1000001
    for Geo in Geo_range:
        for theta in theta_range:
            optimal = minimize(loss_SEIR, [uni(beta_SEIR_range[0], beta_SEIR_range[1]),
                                           uni(betaEI_range[0], betaEI_range[1]),
                                           uni(gamma_range[0], gamma_range[1]),
                                           uni(eta_range[0], eta_range[1])],
                               args=(confirmed, n_0, SEIRG_sd, theta, Geo), method='L-BFGS-B',
                               bounds=[beta_SEIR_range, betaEI_range, gamma_range, eta_range])
            current_loss = loss_SEIR(optimal.x, confirmed, n_0, SEIRG, theta, Geo)
            if current_loss < min_loss:
                min_loss = current_loss
                [beta, betaEI, gamma, eta] = optimal.x
                para_best = [beta, betaEI, gamma, eta, theta, Geo]

    return current_loss, para_best


def loss_SIR(point, confirmed, n_0, SIRG, theta, Geo):
    size = len(confirmed)
    beta = point[0]
    gamma = point[1]
    eta = point[2]
    # c1 = point[3]
    S = [n_0 * eta]
    I = [confirmed[0]]
    R = [0]
    G = [confirmed[0]]
    for i in range(1, size):
        delta = SIRG(i, [S[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, gamma, eta, n_0])
        S.append(S[-1] + delta[0])
        I.append(I[-1] + delta[1])
        R.append(R[-1] + delta[2])
        G.append(G[-1] + delta[3])
        if S[-1] < 0:
            return 1000000

    confirmed_derivative = np.diff(confirmed)
    G_derivative = np.diff(G)
    weights = [Geo ** (n - 1) for n in range(1, size)]
    weights.reverse()
    confirmed_derivative *= weights
    G_derivative *= weights

    metric0 = r2_score(confirmed, G)
    metric1 = r2_score(confirmed_derivative, G_derivative)
    return - (theta * metric0 + (1 - theta) * metric1)


def loss_SEIR_SD(point, confirmed, n_0, SIRG, theta, Geo, c1):
    size = len(confirmed)
    beta = point[0]
    betaEI = point[1]
    gamma = point[2]
    eta = point[3]
    # c1 = point[3]
    S = [n_0 * eta]
    E = [0]
    I = [confirmed.iloc[0]]
    R = [0]
    G = [confirmed.iloc[0]]
    for i in range(1, size):
        delta = SIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0, c1])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        R.append(R[-1] + delta[3])
        G.append(G[-1] + delta[4])
        if S[-1] < 0:
            return 1000000

    confirmed_derivative = np.diff(confirmed)
    G_derivative = np.diff(G)
    weights = [Geo ** (n - 1) for n in range(1, size)]
    weights.reverse()
    confirmed_derivative *= weights
    G_derivative *= weights

    metric0 = r2_score(confirmed, G)
    metric1 = r2_score(confirmed_derivative, G_derivative)
    return - (theta * metric0 + (1 - theta) * metric1)


def loss_SEIR(point, confirmed, n_0, SIRG, theta, Geo):
    size = len(confirmed)
    beta = point[0]
    betaEI = point[1]
    gamma = point[2]
    eta = point[3]
    # c1 = point[3]
    S = [n_0 * eta]
    E = [0]
    I = [confirmed.iloc[0]]
    R = [0]
    G = [confirmed.iloc[0]]
    for i in range(1, size):
        delta = SIRG(i, [S[i - 1], E[i - 1], I[i - 1], R[i - 1], G[i - 1], beta, betaEI, gamma, eta, n_0])
        S.append(S[-1] + delta[0])
        E.append(E[-1] + delta[1])
        I.append(I[-1] + delta[2])
        R.append(R[-1] + delta[3])
        G.append(G[-1] + delta[4])
        if S[-1] < 0:
            return 1000000

    confirmed_derivative = np.diff(confirmed)
    G_derivative = np.diff(G)
    weights = [Geo ** (n - 1) for n in range(1, size)]
    weights.reverse()
    confirmed_derivative *= weights
    G_derivative *= weights

    metric0 = r2_score(confirmed, G)
    metric1 = r2_score(confirmed_derivative, G_derivative)
    return - (theta * metric0 + (1 - theta) * metric1)


def compare_100():
    if not os.path.exists('50Counties/comparison'):
        os.makedirs('50Counties/comparison')
    states = ['AZ-Maricopa',
              'CA-Los Angeles',
              'CA-Orange',
              'CA-Riverside',
              'CA-San Bernardino',
              'CA-San Diego',
              'CO-Adams',
              'CO-Arapahoe',
              'CO-Denver',
              'CT-Fairfield',
              'CT-Hartford',
              'CT-New Haven',
              'DC-District of Columbia',
              'DE-New Castle',
              'DE-Sussex',
              'FL-Broward',
              'FL-Miami-Dade',
              'FL-Palm Beach',
              'GA-DeKalb',
              'GA-Fulton',
              'GA-Gwinnett',
              'IA-Polk',
              'IL-Cook',
              'IL-DuPage',
              'IL-Kane',
              'IL-Lake',
              'IL-Will',
              'IN-Lake',
              'IN-Marion',
              'LA-East Baton Rouge',
              'LA-Jefferson',
              'LA-Orleans',
              'MA-Bristol',
              'MA-Essex',
              'MA-Hampden',
              'MA-Middlesex',
              'MA-Norfolk',
              'MA-Plymouth',
              'MA-Suffolk',
              'MA-Worcester',
              'MD-Anne Arundel',
              'MD-Baltimore',
              'MD-Baltimore City',
              'MD-Montgomery',
              'MD-Prince George\'s',
              'MI-Kent',
              'MI-Macomb',
              'MI-Oakland',
              'MI-Wayne',
              'MN-Hennepin',
              'MO-St. Louis',
              'NJ-Bergen',
              'NJ-Burlington',
              'NJ-Camden',
              'NJ-Essex',
              'NJ-Hudson',
              'NJ-Mercer',
              'NJ-Middlesex',
              'NJ-Monmouth',
              'NJ-Morris',
              'NJ-Ocean',
              'NJ-Passaic',
              'NJ-Somerset',
              'NJ-Union',
              'NV-Clark',
              'NY-Bronx',
              'NY-Dutchess',
              'NY-Erie',
              'NY-Kings',
              'NY-Nassau',
              'NY-New York',
              'NY-Orange',
              'NY-Queens',
              'NY-Richmond',
              'NY-Rockland',
              'NY-Suffolk',
              'NY-Westchester',
              'OH-Cuyahoga',
              'OH-Franklin',
              'PA-Berks',
              'PA-Bucks',
              'PA-Delaware',
              'PA-Lehigh',
              'PA-Luzerne',
              'PA-Montgomery',
              'PA-Northampton',
              'PA-Philadelphia',
              'RI-Providence',
              'SD-Minnehaha',
              'TN-Davidson',
              'TN-Shelby',
              'TX-Dallas',
              'TX-Harris',
              'TX-Tarrant',
              'UT-Salt Lake',
              'VA-Fairfax',
              'VA-Prince William',
              'WA-King',
              'WA-Snohomish',
              'WI-Milwaukee']
    states.sort()
    out_df = []

    # first 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison1.png', bbox_inches="tight")
    plt.close(fig)

    # second 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25, 50):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison2.png', bbox_inches="tight")
    plt.close(fig)

    # third 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(50, 75):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison3.png', bbox_inches="tight")
    plt.close(fig)

    # forth 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(75, 100):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison4.png', bbox_inches="tight")
    plt.close(fig)

    out_df = pd.DataFrame(out_df,
                          columns=['state', 'RMSE_wtd_G', 'RMSE_wtd_G(SIR)', 'R2_G', 'R2_wtd_G', 'RMSE_G',
                                   'RMSE_G(SIR)', 'MAE_G', 'MAE_G(SIR)', 'MAPE_G', 'MAPE_G(SIR)'])
    out_df.to_csv('50Counties/comparison/RMSE.csv', index=False)
    # print(out_df)
    return


def compare_50():
    if not os.path.exists('50Counties/comparison'):
        os.makedirs('50Counties/comparison')
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King']
    out_df = []

    # first 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison1.png', bbox_inches="tight")
    plt.close(fig)

    # second 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25, 50):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison2.png', bbox_inches="tight")
    plt.close(fig)

    out_df = pd.DataFrame(out_df,
                          columns=['state', 'RMSE_G', 'RMSE_wtd_G', 'RMSE_G(SIR)', 'RMSE_wtd_G(SIR)', 'R2_G',
                                   'R2_wtd_G'])
    out_df.to_csv('50Counties/comparison/RMSE.csv', index=False)
    # print(out_df)
    return


def compare_50more():
    if not os.path.exists('50Counties/comparison'):
        os.makedirs('50Counties/comparison')
    states = ['PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside', 'PA-Delaware', 'CA-San Diego',
              'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin', 'WI-Milwaukee', 'CO-Denver',
              'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis', 'IL-Will', 'TX-Tarrant',
              'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson', 'UT-Salt Lake', 'GA-Fulton',
              'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex', 'NY-Dutchess',
              'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb',
              'IN-Lake', 'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    out_df = []

    # first 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison3.png', bbox_inches="tight")
    plt.close(fig)

    # second 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0
    row = 5
    col = 5

    for i in range(25, 50):
        num_fig += 1
        state = states[i]
        ax = fig.add_subplot(row, col, num_fig)
        compare_state(state, out_df, ax)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    # fig.autofmt_xdate()
    fig.subplots_adjust(hspace=0.4, wspace=0.25)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Cases (Thousand)")
    fig.savefig('50Counties/comparison/comparison4.png', bbox_inches="tight")
    plt.close(fig)

    out_df = pd.DataFrame(out_df,
                          columns=['state', 'RMSE_G', 'RMSE_wtd_G', 'RMSE_G(SIR)', 'RMSE_wtd_G(SIR)', 'R2_G',
                                   'R2_wtd_G'])
    out_df.to_csv('50Counties/comparison/RMSE2.csv', index=False)
    # print(out_df)
    return


# compare SD with SIR
def compare_state(state, out_df, ax):
    print(state)
    SimFolder = f'50Counties/init_only_{end_date}/{state}'
    # SimFolder = f'init_only_{end_date}/{state}'
    S, I, IH, IN, R, D, G, H, days = read_sim(state, SimFolder)
    start_date = days[0]
    end_date2 = days[-1]

    SIRFolder = f'50Counties/SIR/{state}'
    S2, I2, R2, G2, days2 = read_SIR(state, SIRFolder)

    ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    confirmed = confirmed.iloc[0].loc[start_date: end_date2]

    Geo = 0.98
    weighted_G = weighting(G, Geo)
    weighted_G2 = weighting(G2, Geo)
    weighted_confirmed = weighting(confirmed, Geo)
    out_df.append([state,
                   math.sqrt(mean_squared_error(weighted_G, weighted_confirmed)),
                   math.sqrt(mean_squared_error(weighted_G2, weighted_confirmed)),
                   r2_score(G, confirmed),
                   r2_score(weighted_G, weighted_confirmed),
                   math.sqrt(mean_squared_error(G, confirmed)),
                   math.sqrt(mean_squared_error(G2, confirmed)),
                   mean_absolute_error(G, confirmed),
                   mean_absolute_error(G2, confirmed),
                   mean_absolute_percentage_error(G, confirmed),
                   mean_absolute_percentage_error(G2, confirmed)
                   ])
    days2 = make_datetime(days[0], len(days2))
    # plt.plot(range(len(G2)), confirmed2, label='Cumulative Cases')
    # plt.plot(range(len(G2)), Gi, label='SD')
    # plt.plot(range(len(G2)), G2, label='SIR')
    ax.plot(days2, [i / 1000 for i in confirmed], linewidth=3, linestyle=':', label='Cumulative Cases')
    ax.plot(days2, [i / 1000 for i in G], label='SIR-SD')
    ax.plot(days2, [i / 1000 for i in G2], label='SIR')
    # ax.legend()
    ax.set_title(state)
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    # fig.autofmt_xdate()
    # plt.show()
    fig = plt.figure()
    ax2 = fig.add_subplot()
    ax2.plot(days2, [i / 1000 for i in confirmed], linewidth=3, linestyle=':', label='Cumulative Cases')
    ax2.plot(days2, [i / 1000 for i in G], label='SIR-SD')
    ax2.plot(days2, [i / 1000 for i in G2], label='SIR')
    ax2.set_title(state)
    ax2.set_ylabel('Cases (Thousand)')
    ax2.legend()
    fig.autofmt_xdate()
    fig.savefig(f'50Counties/comparison/comparison_{state}.png', bbox_inches="tight")
    plt.close(fig)
    return


def read_SIR(state, SimFolder):
    # ParaFile = f'{SimFolder}/para.csv'
    SimFile = f'{SimFolder}/sim.csv'
    df = pd.read_csv(SimFile)
    S = df[df['series'] == 'S'].iloc[0].iloc[1:]
    I = df[df['series'] == 'I'].iloc[0].iloc[1:]
    R = df[df['series'] == 'R'].iloc[0].iloc[1:]
    G = df[df['series'] == 'G'].iloc[0].iloc[1:]
    days = S.index

    return S, I, R, G, days


def make_datetime(start_date, size):
    dates = [datetime.datetime.strptime(start_date, '%Y-%m-%d')]
    for i in range(1, size):
        dates.append(dates[0] + datetime.timedelta(days=i))
    return dates


def bar_50():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axhline(1, linestyle='dashed', color='tab:grey')
    df = pd.read_csv('50Counties/comparison/RMSE.csv', usecols=['state', 'RMSE_wtd_G', 'RMSE_wtd_G(SIR)'])
    for index, row in df.iterrows():
        ratio = row['RMSE_wtd_G'] / row['RMSE_wtd_G(SIR)']
        ax.bar(index + 1, ratio, color='green' if ratio < 1 else 'red')
    # ax.bar(index + 1, ratio, color='green' if ratio < 1 else 'red', label=row['state'] if ratio > 1 else '')
    ax.axes.xaxis.set_ticks([])
    ax.set_title('Ratio of Weighted RMSE')
    # ax.legend()
    fig.savefig('50Counties/comparison/RMSE ratio.png', bbox_inches="tight")
    # plt.show()
    return


def bar_RMSE_100():
    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_subplot()
    df = pd.read_csv('50Counties/comparison/RMSE.csv', usecols=['state', 'RMSE_wtd_G', 'RMSE_wtd_G(SIR)'])
    ratios = []
    for index, row in df.iterrows():
        ratio = row['RMSE_wtd_G'] / row['RMSE_wtd_G(SIR)']
        ratios.append(ratio)
        ax.bar(index + 1, ratio, color='green' if ratio < 1 else 'red')
        if ratio > 1:
            print(row['state'])
    print(np.mean(ratios))
    ax.axhline(np.mean(ratios), linestyle='dashed', color='tab:grey', label=f'Average={round(np.mean(ratios), 2)}')
    # ax.bar(index + 1, ratio, color='green' if ratio < 1 else 'red', label=row['state'] if ratio > 1 else '')
    ax.axes.xaxis.set_ticks([])
    ax.set_title('Ratio of Weighted RMSE')
    ax.legend()
    fig.savefig('50Counties/comparison/RMSE ratio.png', bbox_inches="tight")
    # plt.show()
    return


def validate_all(date):
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King', 'PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside',
              'PA-Delaware', 'CA-San Diego', 'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin',
              'WI-Milwaukee', 'CO-Denver', 'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis',
              'IL-Will', 'TX-Tarrant', 'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson',
              'UT-Salt Lake', 'GA-Fulton', 'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex',
              'NY-Dutchess', 'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb', 'IN-Lake',
              'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    table = []
    for state in states:
        table.append(validate(state, date, '2020-05-22'))
    cols = ['state', 'SD 1W', 'SIR 1W', 'SD 2W', 'SIR 2W', 'SD 3W', 'SIR 3W', 'SD 4W', 'SIR 4W']
    df = pd.DataFrame(table, columns=cols)
    df.to_csv(f'50Counties/comparison_{date}/RMSE.csv', index=False)
    return


def validate(state, date, simend_date):
    if not os.path.exists(f'50Counties/comparison_{date}'):
        os.makedirs(f'50Counties/comparison_{date}')
    ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
    DeathFile = 'JHU/JHU_Death-counties.csv'
    PopFile = 'JHU/CountyPopulation.csv'

    df = pd.read_csv(f'50Counties/init_only_{date}/{state}/para.csv')
    [beta, gamma, gamma2, a1, a2, a3, h, Hiding_init, k, k2, eta, c1, metric1, metric2, r1, r2] = df.iloc[0]

    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]
    df2 = pd.read_csv(DeathFile)
    death = df2[df2.iloc[:, 0] == state]
    for start_date in confirmed.columns[1:]:
        # if confirmed.iloc[0].loc[start_date] >= I_0 and death.iloc[0].loc[start_date] > 0:
        if confirmed.iloc[0].loc[start_date] >= I_0:
            break
    days = list(confirmed.columns)
    # days = days[days.index(start_date):days.index(end_date) + 1]
    days = days[days.index(start_date):days.index(simend_date) + 1]
    days = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in days]
    confirmed = confirmed.iloc[0].loc[start_date: simend_date]
    death = death.iloc[0].loc[start_date: simend_date]
    for i in range(len(death)):
        if death.iloc[i] == 0:
            death.iloc[i] = 0.01
    death = death.tolist()

    S = [n_0 * eta]
    I = [confirmed[0]]
    IH = [I[-1] * gamma]
    IN = [I[-1] * gamma2]
    D = [death[0]]
    R = [0]
    G = [confirmed[0]]
    H = [Hiding_init * n_0 * eta]
    size = len(confirmed)
    result, [S, I, IH, IN, D, R, G, H, betas] \
        = simulate_combined(size, SIRG_combined, S, I, IH, IN, D, R, G, H, beta, gamma, gamma2, a1, a2, a3, h, k, k2,
                            eta, c1,
                            n_0, size)

    df2 = pd.read_csv(f'50Counties/SIR_{date}/{state}/para.csv')
    beta, gamma, eta, theta, Geo, RMSE, r2 = df2.iloc[0]
    S2 = [n_0 * eta]
    I2 = [confirmed[0]]
    R2 = [0]
    G2 = [confirmed[0]]
    for i in range(1, size):
        delta = SIRG(i, [S2[i - 1], I2[i - 1], R2[i - 1], G2[i - 1], beta, gamma, eta, n_0])
        S2.append(S2[-1] + delta[0])
        I2.append(I2[-1] + delta[1])
        R2.append(R2[-1] + delta[2])
        G2.append(G2[-1] + delta[3])

    fig = plt.figure()
    fig.suptitle(state)
    ax = fig.add_subplot()
    ax.plot(days, [i / 1000 for i in confirmed], linewidth=3, linestyle=':', label='Cumulative Cases')
    ax.plot(days, [i / 1000 for i in G], label='SIR-SD')
    ax.plot(days, [i / 1000 for i in G2], label='SIR')
    ax.axvline(datetime.datetime.strptime(date, '%Y-%m-%d'), linestyle='dashed', color='tab:grey')
    fig.autofmt_xdate()
    ax.legend()
    # plt.show()
    fig.savefig(f'50Counties/comparison_{date}/{state}.png', bbox_inches="tight")
    plt.close(fig)

    RMSE_start_day = days.index(datetime.datetime.strptime(date, '%Y-%m-%d'))
    return RMSE_validate(state, confirmed[RMSE_start_day:], G[RMSE_start_day:], G2[RMSE_start_day:])


def RMSE_validate(state, confirmed, G, G2):
    row = [state]
    row.append(math.sqrt(mean_squared_error(confirmed[:7], G[:7])))
    row.append(math.sqrt(mean_squared_error(confirmed[:7], G2[:7])))
    row.append(math.sqrt(mean_squared_error(confirmed[:14], G[:14])))
    row.append(math.sqrt(mean_squared_error(confirmed[:14], G2[:14])))
    row.append(math.sqrt(mean_squared_error(confirmed[:21], G[:21])))
    row.append(math.sqrt(mean_squared_error(confirmed[:21], G2[:21])))
    row.append(math.sqrt(mean_squared_error(confirmed[:28], G[:28])))
    row.append(math.sqrt(mean_squared_error(confirmed[:28], G2[:28])))
    return row


def scatter_RMSE_validate_all(date):
    PopFile = 'JHU/CountyPopulation.csv'
    Pop_df = pd.read_csv(PopFile)
    states = ['AZ-Maricopa', 'CA-Los Angeles', 'CT-Fairfield', 'CT-Hartford', 'CT-New Haven', 'DC-District of Columbia',
              'FL-Broward', 'FL-Miami-Dade', 'IL-Cook', 'IL-Lake', 'IN-Marion', 'LA-Jefferson', 'LA-Orleans',
              'MD-Montgomery', 'MD-Prince George\'s', 'MA-Bristol', 'MA-Essex', 'MA-Middlesex', 'MA-Norfolk',
              'MA-Plymouth', 'MA-Suffolk', 'MA-Worcester', 'MI-Macomb', 'MI-Oakland', 'MI-Wayne', 'NJ-Bergen',
              'NJ-Essex', 'NJ-Hudson', 'NJ-Middlesex', 'NJ-Monmouth', 'NJ-Morris', 'NJ-Ocean', 'NJ-Passaic', 'NJ-Union',
              'NY-Bronx', 'NY-Kings', 'NY-Nassau', 'NY-New York', 'NY-Orange', 'NY-Queens', 'NY-Richmond',
              'NY-Rockland', 'NY-Suffolk', 'NY-Westchester', 'PA-Philadelphia', 'RI-Providence', 'TX-Dallas',
              'TX-Harris', 'VA-Fairfax', 'WA-King', 'PA-Montgomery', 'NJ-Mercer', 'IL-DuPage', 'CA-Riverside',
              'PA-Delaware', 'CA-San Diego', 'MA-Hampden', 'NJ-Camden', 'NV-Clark', 'NY-Erie', 'MN-Hennepin',
              'WI-Milwaukee', 'CO-Denver', 'MD-Baltimore', 'FL-Palm Beach', 'OH-Franklin', 'PA-Bucks', 'MO-St. Louis',
              'IL-Will', 'TX-Tarrant', 'NJ-Somerset', 'IL-Kane', 'CA-Orange', 'NJ-Burlington', 'TN-Davidson',
              'UT-Salt Lake', 'GA-Fulton', 'MD-Baltimore City', 'TN-Shelby', 'PA-Berks', 'CO-Arapahoe', 'DE-Sussex',
              'NY-Dutchess', 'VA-Prince William', 'PA-Lehigh', 'CA-San Bernardino', 'OH-Cuyahoga', 'SD-Minnehaha',
              'LA-East Baton Rouge', 'MI-Kent', 'IA-Polk', 'WA-Snohomish', 'MD-Anne Arundel', 'GA-DeKalb', 'IN-Lake',
              'DE-New Castle', 'PA-Northampton', 'GA-Gwinnett', 'CO-Adams', 'PA-Luzerne']
    fig = plt.figure(figsize=(14, 10))
    axes = fig.subplots(2, 2)

    ax = axes[0][0]
    ax.set_title('1 week')
    df = pd.read_csv(f'50Counties/comparison_{date}/RMSE.csv')
    RMSE_SDs, RMSE_SIRs, ratios = [], [], []
    for index, row in df.iterrows():
        state = row['state']
        n_0 = Pop_df[Pop_df.iloc[:, 0] == state].iloc[0]['POP']
        RMSE_SDs.append(row['SD 1W'] / n_0)
        RMSE_SIRs.append(row['SIR 1W'] / n_0)
        ratios.append(RMSE_SDs[-1] / RMSE_SIRs[-1])
        ax.scatter(RMSE_SDs[-1], RMSE_SIRs[-1], color='green' if ratios[-1] < 1 else 'red')
    print(np.mean(ratios))
    min_RMSE = min(RMSE_SDs + RMSE_SIRs)
    max_RMSE = min(max(RMSE_SDs), max(RMSE_SIRs))
    line_range = np.arange(min_RMSE, max_RMSE + 0.0001, 0.0001)
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    ax.set_xlabel('RMSE(SIR-SD)')
    ax.set_ylabel('RMSE(SIR)')

    ax = axes[0][1]
    ax.set_title('2 weeks')
    df = pd.read_csv(f'50Counties/comparison_{date}/RMSE.csv')
    RMSE_SDs, RMSE_SIRs, ratios = [], [], []
    for index, row in df.iterrows():
        state = row['state']
        n_0 = Pop_df[Pop_df.iloc[:, 0] == state].iloc[0]['POP']
        RMSE_SDs.append(row['SD 2W'] / n_0)
        RMSE_SIRs.append(row['SIR 2W'] / n_0)
        ratios.append(RMSE_SDs[-1] / RMSE_SIRs[-1])
        ax.scatter(RMSE_SDs[-1], RMSE_SIRs[-1], color='green' if ratios[-1] < 1 else 'red')
    print(np.mean(ratios))
    min_RMSE = min(RMSE_SDs + RMSE_SIRs)
    max_RMSE = min(max(RMSE_SDs), max(RMSE_SIRs))
    line_range = np.arange(min_RMSE, max_RMSE + 0.0001, 0.0001)
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    ax.set_xlabel('RMSE(SIR-SD)')
    ax.set_ylabel('RMSE(SIR)')

    ax = axes[1][0]
    ax.set_title('3 weeks')
    df = pd.read_csv(f'50Counties/comparison_{date}/RMSE.csv')
    RMSE_SDs, RMSE_SIRs, ratios = [], [], []
    for index, row in df.iterrows():
        state = row['state']
        n_0 = Pop_df[Pop_df.iloc[:, 0] == state].iloc[0]['POP']
        RMSE_SDs.append(row['SD 3W'] / n_0)
        RMSE_SIRs.append(row['SIR 3W'] / n_0)
        ratios.append(RMSE_SDs[-1] / RMSE_SIRs[-1])
        ax.scatter(RMSE_SDs[-1], RMSE_SIRs[-1], color='green' if ratios[-1] < 1 else 'red')
    print(np.mean(ratios))
    min_RMSE = min(RMSE_SDs + RMSE_SIRs)
    max_RMSE = min(max(RMSE_SDs), max(RMSE_SIRs))
    line_range = np.arange(min_RMSE, max_RMSE + 0.0001, 0.0001)
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    ax.set_xlabel('RMSE(SIR-SD)')
    ax.set_ylabel('RMSE(SIR)')

    ax = axes[1][1]
    ax.set_title('4 weeks')
    df = pd.read_csv(f'50Counties/comparison_{date}/RMSE.csv')
    RMSE_SDs, RMSE_SIRs, ratios = [], [], []
    for index, row in df.iterrows():
        state = row['state']
        n_0 = Pop_df[Pop_df.iloc[:, 0] == state].iloc[0]['POP']
        RMSE_SDs.append(row['SD 4W'] / n_0)
        RMSE_SIRs.append(row['SIR 4W'] / n_0)
        ratios.append(RMSE_SDs[-1] / RMSE_SIRs[-1])
        ax.scatter(RMSE_SDs[-1], RMSE_SIRs[-1], color='green' if ratios[-1] < 1 else 'red')
    print(np.mean(ratios))
    min_RMSE = min(RMSE_SDs + RMSE_SIRs)
    max_RMSE = min(max(RMSE_SDs), max(RMSE_SIRs))
    line_range = np.arange(min_RMSE, max_RMSE + 0.0001, 0.0001)
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    ax.set_xlabel('RMSE(SIR-SD)')
    ax.set_ylabel('RMSE(SIR)')

    plt.subplots_adjust(hspace=0.25)
    fig.savefig(f'50Counties/comparison_{date}/scatter_{date}.png', bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    return


def plot_RMSE_MAE_MAPE():
    df = pd.read_csv('50Counties/comparison/RMSE.csv')
    PopFile = 'JHU/CountyPopulation.csv'
    Pop_df = pd.read_csv(PopFile)
    fig_RMSE = plt.figure()
    fig_RMSE.suptitle('RMSE per capita')
    ax_RMSE = fig_RMSE.add_subplot()
    fig_MAE = plt.figure()
    fig_MAE.suptitle('MAE per capita')
    ax_MAE = fig_MAE.add_subplot()
    fig_MAPE = plt.figure()
    fig_MAPE.suptitle('MAPE')
    ax_MAPE = fig_MAPE.add_subplot()
    RMSEs_SD = []
    RMSEs_SIR = []
    MAEs_SD = []
    MAEs_SIR = []
    MAPEs_SD = []
    MAPEs_SIR = []
    for index, row in df.iterrows():
        state = row['state']
        n_0 = Pop_df[Pop_df.iloc[:, 0] == state].iloc[0]['POP']
        RMSE_SD = row['RMSE_G']
        RMSEs_SD.append(RMSE_SD / n_0)
        RMSE_SIR = row['RMSE_G(SIR)']
        RMSEs_SIR.append(RMSE_SIR / n_0)
        MAE_SD = row['MAE_G']
        MAEs_SD.append(MAE_SD / n_0)
        MAE_SIR = row['MAE_G(SIR)']
        MAEs_SIR.append(MAE_SIR / n_0)
        MAPE_SD = row['MAPE_G']
        MAPEs_SD.append(MAPE_SD)
        MAPE_SIR = row['MAPE_G(SIR)']
        MAPEs_SIR.append(MAPE_SIR)
        ax_RMSE.scatter(RMSE_SD / n_0, RMSE_SIR / n_0, color='green' if RMSE_SD < RMSE_SIR else 'red')
        ax_MAE.scatter(MAE_SD / n_0, MAE_SIR / n_0, color='green' if MAE_SD < MAE_SIR else 'red')
        ax_MAPE.scatter(MAPE_SD, MAPE_SIR, color='green' if MAPE_SD < MAPE_SIR else 'red')
    ax_RMSE.set_xlabel('SIR-SD')
    ax_RMSE.set_ylabel('SIR')
    ax_MAE.set_xlabel('SIR-SD')
    ax_MAE.set_ylabel('SIR')
    ax_MAPE.set_xlabel('SIR-SD')
    ax_MAPE.set_ylabel('SIR')
    line_range = np.arange(0, min(max(RMSEs_SD), max(RMSEs_SIR)) + 0.001, 0.001)
    ax_RMSE.plot(line_range, line_range, linestyle=':', color='grey')
    line_range = np.arange(0, min(max(MAEs_SD), max(MAEs_SIR)) + 0.001, 0.001)
    ax_MAE.plot(line_range, line_range, linestyle=':', color='grey')
    line_range = np.arange(0, min(max(MAPEs_SD), max(MAPEs_SIR)) + 0.001, 0.001)
    ax_MAPE.plot(line_range, line_range, linestyle=':', color='grey')
    fig_RMSE.savefig('50Counties/comparison/scatter_RMSE.png', bbox_inches="tight")
    fig_MAE.savefig('50Counties/comparison/scatter_MAE.png', bbox_inches="tight")
    fig_MAPE.savefig('50Counties/comparison/scatter_MAPE.png', bbox_inches="tight")

    fig_hist = plt.figure(figsize=(6, 10))
    # fig_ratio = plt.figure()
    ax_RMSE = fig_hist.add_subplot(311)
    ax_RMSE.set_title('Ratio of RMSE per capita(SIR-SD/SIR)')
    ax_MAE = fig_hist.add_subplot(312)
    ax_MAE.set_title('Ratio of MAE per capita(SIR-SD/SIR)')
    ax_MAPE = fig_hist.add_subplot(313)
    ax_MAPE.set_title('Ratio of MAPE(SIR-SD/SIR)')

    ratios_RMSE = [RMSEs_SD[i] / RMSEs_SIR[i] for i in range(len(RMSEs_SD))]
    ratios_MAE = [MAEs_SD[i] / MAEs_SIR[i] for i in range(len(RMSEs_SD))]
    ratios_MAPE = [MAPEs_SD[i] / MAPEs_SIR[i] for i in range(len(RMSEs_SD))]
    ratios_RMSE.sort()
    ratios_MAE.sort()
    ratios_MAPE.sort()
    print(np.std(ratios_RMSE))
    print(np.std(ratios_MAE))
    print(np.std(ratios_MAPE))

    # for i in range(len(RMSEs_SD)):
    # 	ax_RMSE.scatter(ratios_RMSE[i], 0, alpha=0.5, color='green' if ratios_RMSE[i] < 1 else 'red')
    # 	ax_MAE.scatter(ratios_MAE[i], 0, alpha=0.5, color='green' if ratios_MAE[i] < 1 else 'red')
    # 	ax_MAPE.scatter(ratios_MAPE[i], 0, alpha=0.5, color='green' if ratios_MAPE[i] < 1 else 'red')
    ax_RMSE.hist(ratios_RMSE, edgecolor='black')
    ax_MAE.hist(ratios_MAE, edgecolor='black')
    ax_MAPE.hist(ratios_MAPE, edgecolor='black')

    ax_RMSE.axvline(np.mean(ratios_RMSE), linestyle='dashed', color='tab:grey',
                    label=f'AVG={round(np.mean(ratios_RMSE), 3)}')
    ax_MAE.axvline(np.mean(ratios_MAE), linestyle='dashed', color='tab:grey',
                   label=f'AVG={round(np.mean(ratios_MAE), 3)}')
    ax_MAPE.axvline(np.mean(ratios_MAPE), linestyle='dashed', color='tab:grey',
                    label=f'AVG={round(np.mean(ratios_MAPE), 3)}')

    ax_RMSE.axvline(1, color='red')
    ax_MAE.axvline(1, color='red')
    ax_MAPE.axvline(1, color='red')
    # ax_RMSE.axvline((ratios_RMSE[1] + ratios_RMSE[2]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'2.5%={round((ratios_RMSE[1] + ratios_RMSE[2]) / 2, 3)}')
    # ax_RMSE.axvline((ratios_RMSE[-2] + ratios_RMSE[-3]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'97.5%={round((ratios_RMSE[-2] + ratios_RMSE[-3]) / 2, 3)}')
    #
    # ax_MAE.axvline((ratios_MAE[1] + ratios_MAE[2]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'2.5%={round((ratios_MAE[1] + ratios_MAE[2]) / 2, 3)}')
    # ax_MAE.axvline((ratios_MAE[-2] + ratios_MAE[-3]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'97.5%={round((ratios_MAE[-2] + ratios_MAE[-3]) / 2, 3)}')
    #
    # ax_MAPE.axvline((ratios_MAPE[1] + ratios_MAPE[2]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'2.5%={round((ratios_MAPE[1] + ratios_MAPE[2]) / 2, 3)}')
    # ax_MAPE.axvline((ratios_MAPE[-2] + ratios_MAPE[-3]) / 2, linestyle='dashed', color='tab:grey',
    #                 label=f'97.5%={round((ratios_MAPE[-2] + ratios_MAPE[-3]) / 2, 3)}')

    fig_hist.tight_layout(pad=1)
    ax_RMSE.legend()
    ax_MAE.legend()
    ax_MAPE.legend()
    fig_hist.savefig('50Counties/comparison/ratio_hist.png', bbox_inches="tight")

    fig_box = plt.figure()
    fig_box.suptitle('Ratios of SIR-SD / SIR')
    ax_box = fig_box.add_subplot()
    ax_box.boxplot([ratios_RMSE, ratios_MAE, ratios_MAPE], showfliers=False)
    # ax_box.scatter(np.random.normal(1, 0.04, len(ratios_RMSE)), ratios_RMSE, alpha=0.4)
    for i in range(len(ratios_RMSE)):
        ax_box.scatter(np.random.normal(1, 0.04), ratios_RMSE[i], alpha=0.6,
                       c='green' if ratios_RMSE[i] <= 1 else 'red')
        ax_box.scatter(np.random.normal(2, 0.04), ratios_MAE[i], alpha=0.6,
                       c='green' if ratios_MAE[i] <= 1 else 'red')
        ax_box.scatter(np.random.normal(3, 0.04), ratios_MAPE[i], alpha=0.6,
                       c='green' if ratios_MAPE[i] <= 1 else 'red')
    ax_box.set_xticklabels(['RMSE', 'MAE', 'MAPE'])
    fig_box.savefig('50Counties/comparison/ratio_box.png', bbox_inches="tight")
    return


def main():
    # fit_50_init()
    # fit_50more_init()
    # fit_50_SIR()
    # fit_50more_SIR()
    # compare_50()
    # compare_50more()
    # compare_100()
    # bar_50()
    # bar_RMSE_100()
    # plot_RMSE_MAE_MAPE()

    # test()

    # fit until validate end date and extend to end date 5/15
    validate_end_date = '2020-04-22'
    # validate_end_date = '2020-04-15'
    # fit_50_init(validate_end_date)
    # fit_50more_init(validate_end_date)
    # fit_50_SIR(validate_end_date)
    # fit_50more_SIR(validate_end_date)

    # validate_all(validate_end_date)
    # scatter_RMSE_validate_all(validate_end_date)
    # fit_50_SEIR_SD()
    # fit_50more_SEIR_SD()
    fit_50_SEIR()
    fit_50more_SEIR()
    return


if __name__ == '__main__':
    main()
