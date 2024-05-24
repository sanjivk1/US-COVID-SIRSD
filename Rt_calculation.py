import pandas as pd
from scipy.stats import gamma
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import datetime
import numpy as np
import math
from sklearn.metrics import r2_score, mean_squared_error
import os

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
# mean = 5.21
# std = 4.32
mean = 3.99
std = 2.96
# https://www.sciencedirect.com/science/article/pii/S2468042720300634
alpha = (mean / std) ** 2
beta = mean / std ** 2


def Rt_series(dG):
    l = len(dG)
    Rt = []
    for i in range(10, l):
        Rt.append(dG[i])
        denominator = 0
        for j in range(i):
            denominator += dG[i - j - 1] * gamma.pdf(x=j, a=alpha, scale=1 / beta)
        Rt[-1] /= denominator
    return Rt


def state_Rt(state, ax):
    ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]

    PopFile = 'JHU/CountyPopulation.csv'
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    sd_folder = f'50Counties/init_only_2020-08-31'
    SIR_folder = f'50Counties/SIR'
    df1 = pd.read_csv(f'{sd_folder}/{state}/sim.csv')
    G1 = df1[df1['series'] == 'G'].iloc[0, 1:]
    df2 = pd.read_csv(f'{SIR_folder}/{state}/sim.csv')
    G2 = df2[df2['series'] == 'G'].iloc[0, 1:]
    dates_str = G1.index
    confirmed = confirmed.iloc[0][dates_str[0]:dates_str[-1]]

    dG1 = [G1[i] - G1[i - 1] for i in range(1, len(G1))]
    dG2 = [G2[i] - G2[i - 1] for i in range(1, len(G2))]
    dConfirmed = [confirmed[i] - confirmed[i - 1] for i in range(1, len(confirmed))]

    dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

    Rt_c = Rt_series(dConfirmed)
    Rt1 = Rt_series(dG1)
    Rt2 = Rt_series(dG2)

    ax.set_title(state)
    ax.plot(dates[len(dates) - len(Rt_c):], Rt_c, label='Reported', linewidth=3, linestyle=':')
    ax.plot(dates[len(dates) - len(Rt_c):], Rt1, label='SIR-SD')
    ax.plot(dates[len(dates) - len(Rt_c):], Rt2, label='SIR')
    date_form = DateFormatter("%m-%d")
    ax.xaxis.set_major_formatter(date_form)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig = plt.figure()
    ax2 = fig.add_subplot()
    ax2.set_title(state)
    ax2.set_ylabel('Reproduction Number')
    ax2.plot(dates[len(dates) - len(Rt_c):], Rt_c, label='Reported', linewidth=3, linestyle=':')
    ax2.plot(dates[len(dates) - len(Rt_c):], Rt1, label='SIR-SD')
    ax2.plot(dates[len(dates) - len(Rt_c):], Rt2, label='SIR')
    ax2.legend()
    fig.autofmt_xdate()
    folder = '50Counties/comparison/Rt'
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(f'{folder}/Rt_{state}.png', bbox_inches="tight")
    plt.close(fig)
    # plt.show()

    RMSE_sd = math.sqrt(mean_squared_error(Rt_c, Rt1))
    RMSE_SIR = math.sqrt(mean_squared_error(Rt_c, Rt2))
    # print(f'{state}:{round(r2_sd, 5)} VS {round(r2_SIR, 5)}')
    return RMSE_sd, RMSE_SIR


def gamma_plotter():
    t_range = np.arange(0, 20, 0.01)
    p = []
    m = 5
    s = 1
    a = (m / s) ** 2
    b = m / (s ** 2)
    for t in t_range:
        p.append(gamma.pdf(t, a=a, scale=1 / b))
    # p.append(gamma.pdf(t, a=5, scale=1))
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t_range, p)
    plt.show()
    return


def RMSE_all():
    folder = '50Counties/comparison/Rt'
    if not os.path.exists(folder):
        os.makedirs(folder)
    RMSEs_sd = []
    RMSEs_SIR = []

    row = 5
    col = 5

    # first 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0

    for state in states[:25]:
        num_fig += 1
        ax = fig.add_subplot(row, col, num_fig)
        RMSE_sd, RMSE_SIR = state_Rt(state, ax)
        RMSEs_sd.append(RMSE_sd)
        RMSEs_SIR.append(RMSE_SIR)
        print(state)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Reproduction Number")
    fig.savefig('50Counties/comparison/Rt/grid_Rt1.png', bbox_inches="tight")
    plt.close(fig)

    # second 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0

    for state in states[25:50]:
        num_fig += 1
        ax = fig.add_subplot(row, col, num_fig)
        RMSE_sd, RMSE_SIR = state_Rt(state, ax)
        RMSEs_sd.append(RMSE_sd)
        RMSEs_SIR.append(RMSE_SIR)
        print(state)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Reproduction Number")
    fig.savefig('50Counties/comparison/Rt/grid_Rt2.png', bbox_inches="tight")
    plt.close(fig)

    # third 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0

    for state in states[50:75]:
        num_fig += 1
        ax = fig.add_subplot(row, col, num_fig)
        RMSE_sd, RMSE_SIR = state_Rt(state, ax)
        RMSEs_sd.append(RMSE_sd)
        RMSEs_SIR.append(RMSE_SIR)
        print(state)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Reproduction Number")
    fig.savefig('50Counties/comparison/Rt/grid_Rt3.png', bbox_inches="tight")
    plt.close(fig)

    # last 25 counties
    fig = plt.figure(figsize=[16, 18])
    num_fig = 0

    for state in states[75:]:
        num_fig += 1
        ax = fig.add_subplot(row, col, num_fig)
        RMSE_sd, RMSE_SIR = state_Rt(state, ax)
        RMSEs_sd.append(RMSE_sd)
        RMSEs_SIR.append(RMSE_SIR)
        print(state)

    ax.legend(loc='upper left', bbox_to_anchor=(0, -0.3))
    fig.subplots_adjust(hspace=0.4, wspace=0.25)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Reproduction Number")
    fig.savefig('50Counties/comparison/Rt/grid_Rt4.png', bbox_inches="tight")
    plt.close(fig)

    cols = ['state', 'RMSE_sd', 'RMSE_SIR']
    df = pd.DataFrame(columns=cols)
    df['state'] = states
    df['RMSE_sd'] = RMSEs_sd
    df['RMSE_SIR'] = RMSEs_SIR
    df.to_csv(f'{folder}/Rt_RMSE.csv', index=False)
    # print(df)
    return


def RMSE_plot():
    folder = '50Counties/comparison/Rt'
    if not os.path.exists(folder):
        os.makedirs(folder)
    df = pd.read_csv(f'{folder}/Rt_RMSE.csv')
    df = df[df['state'] != 'RI-Providence']
    RMSEs_sd = list(df['RMSE_sd'])
    RMSEs_SIR = list(df['RMSE_SIR'])
    fig = plt.figure()
    fig.suptitle('RMSE of reproduction numbers')
    ax = fig.add_subplot()
    # ax.scatter([r2s_sd[i], r2s_SIR[i] for i in range(len(r2s_sd))])
    for i in range(len(RMSEs_sd)):
        ax.scatter(RMSEs_sd[i], RMSEs_SIR[i], color='red' if RMSEs_sd[i] > RMSEs_SIR[i] else 'green')
    ax.set_xlabel('RMSE(SIR-SD)')
    ax.set_ylabel('RMSE(SIR)')
    max_lim = min(max(RMSEs_sd), max(RMSEs_SIR))
    min_lim = max(min(RMSEs_sd), min(RMSEs_SIR))
    line_range = np.arange(min_lim, max_lim + 0.001, 0.001)
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    fig.savefig(f'{folder}/Rt_RMSE_scatter.png', bbox_inches="tight")

    ratios_RMSE = [RMSEs_sd[i] / RMSEs_SIR[i] for i in range(len(RMSEs_sd))]

    fig_hist = plt.figure()
    fig_hist.suptitle('Ratio of RMSE(SIR-SD/SIR)')
    ax = fig_hist.add_subplot()
    # ax.set_title('Ratio of RMSE(SIR-SD/SIR)')
    ax.hist(ratios_RMSE, edgecolor='black')
    ax.axvline(np.mean(ratios_RMSE), linestyle='dashed', color='tab:grey',
               label=f'AVG={round(np.mean(ratios_RMSE), 3)}')
    ax.axvline(1, color='red')
    ax.legend()
    fig_hist.savefig(f'{folder}/Rt_RMSE_hist.png', bbox_inches="tight")

    fig_box = plt.figure(figsize=(3, 6))
    fig_box.suptitle('Ratio of RMSE(SIR-SD/SIR)')
    ax = fig_box.add_subplot()
    # ax.set_title('Ratio of RMSE(SIR-SD/SIR)')
    ax.boxplot(ratios_RMSE, showfliers=False)
    for i in range(len(ratios_RMSE)):
        ax.scatter(np.random.normal(1, 0.04), ratios_RMSE[i], alpha=0.6, c='green' if ratios_RMSE[i] <= 1 else 'red')
    ax.set_xticklabels([''])
    fig_box.savefig(f'{folder}/Rt_RMSE_box.png', bbox_inches="tight")

    return


def R0_state(state):
    ConfirmFile = 'JHU/JHU_Confirmed-counties.csv'
    df = pd.read_csv(ConfirmFile)
    confirmed = df[df.iloc[:, 0] == state]

    PopFile = 'JHU/CountyPopulation.csv'
    df = pd.read_csv(PopFile)
    n_0 = df[df.iloc[:, 0] == state].iloc[0]['POP']

    sd_folder = f'50Counties/init_only_2020-08-31'
    SIR_folder = f'50Counties/SIR'
    df1 = pd.read_csv(f'{sd_folder}/{state}/sim.csv')
    G1 = df1[df1['series'] == 'G'].iloc[0, 1:]
    df2 = pd.read_csv(f'{SIR_folder}/{state}/sim.csv')
    G2 = df2[df2['series'] == 'G'].iloc[0, 1:]
    dates_str = G1.index
    confirmed = confirmed.iloc[0][dates_str[0]:dates_str[-1]]

    dG1 = [G1.iloc[i] - G1.iloc[i - 1] for i in range(1, len(G1))]
    dG2 = [G2.iloc[i] - G2.iloc[i - 1] for i in range(1, len(G2))]
    dConfirmed = [confirmed.iloc[i] - confirmed.iloc[i - 1] for i in range(1, len(confirmed))]

    dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d in dates_str]

    Rt_c = Rt_series(dConfirmed)
    Rt1 = Rt_series(dG1)
    Rt2 = Rt_series(dG2)
    return Rt1[0], Rt2[0]


def R0_all():
    folder = '50Counties/comparison/R0'
    if not os.path.exists(folder):
        os.makedirs(folder)
    R0s_sd = []
    R0s_SIR = []
    for state in states:
        R0_sd, R0_SIR = R0_state(state)
        R0s_sd.append(R0_sd)
        R0s_SIR.append(R0_SIR)
    df = pd.DataFrame(columns=['state', 'R0_SD', 'R0_SIR'])
    df['state'] = states
    df['R0_SD'] = R0s_sd
    df['R0_SIR'] = R0s_SIR
    df.to_csv(f'{folder}/R0.csv', index=False)
    return


def R0_para_all():
    folder = '50Counties/comparison/R0'
    sd_folder = f'50Counties/init_only_2020-08-31'
    SIR_folder = f'50Counties/SIR'
    SEIR_folder = '50Counties/SEIR_2020-05-15'
    if not os.path.exists(folder):
        os.makedirs(folder)
    R0s_sd = []
    R0s_SIR = []
    R0s_SEIR = []
    for state in states:
        # SIR-sd
        df = pd.read_csv(f'{sd_folder}/{state}/para.csv')
        b = df['beta'].iloc[0]
        g = df['gamma'].iloc[0]
        g2 = df['gamma2'].iloc[0]
        e = df['eta'].iloc[0]
        R0s_sd.append(b * e / (g + g2))

        # SIR
        df = pd.read_csv(f'{SIR_folder}/{state}/para.csv')
        b = df['beta'].iloc[0]
        g = df['gamma'].iloc[0]
        e = df['eta'].iloc[0]
        R0s_SIR.append(b * e / g)

        #SEIR
        df = pd.read_csv(f'{SEIR_folder}/{state}/para.csv')
        b = df['beta'].iloc[0]
        bEI = df['betaEI'].iloc[0]
        g = df['gamma'].iloc[0]
        e = df['eta'].iloc[0]
        R0s_SEIR.append(b * e / (g + bEI))

    df = pd.DataFrame(columns=['state', 'R0_SD', 'R0_SIR', 'R0_SEIR'])
    df['state'] = states
    df['R0_SD'] = R0s_sd
    df['R0_SIR'] = R0s_SIR
    df['R0_SEIR'] = R0s_SEIR
    df.to_csv(f'{folder}/R0_para.csv', index=False)
    return


def main():
    # gamma_plotter()
    # RMSE_all()
    # RMSE_plot()
    # R0_all()
    R0_para_all()
    return


if __name__ == '__main__':
    main()
