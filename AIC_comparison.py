import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def AIC_compare():
    # folder = '50Counties/comparison'
    df = pd.read_csv('50Counties/comparison/RMSE.csv')[['state', 'RMSE_G', 'RMSE_G(SIR)']]
    para_SIR, para_SD = 3, 8
    out_df = pd.DataFrame(columns=['state', 'AIC_SD', 'AIC_SIR', 'MSE_SD', 'MSE_SIR', 'sample size'])
    for index, row in df.iterrows():
        state, RMSE_SD, RMSE_SIR = row
        n = pd.read_csv(f'50Counties/SIR/{state}/sim.csv').shape[1] - 1
        # print(n, state, RMSE_SD, RMSE_SIR)
        MSE_SD = RMSE_SD ** 2
        MSE_SIR = RMSE_SIR ** 2
        AIC_SIR = n * np.log(MSE_SIR) + 2 * para_SIR
        AIC_SD = n * np.log(MSE_SD) + 2 * para_SD
        out_df.loc[len(out_df)] = [state, AIC_SD, AIC_SIR, MSE_SD, MSE_SIR, n]
    # print(out_df.columns)
    out_df.to_csv('50Counties/comparison/AIC.csv', index=False)


def AIC_boxplot():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig_box = plt.figure(figsize=(3, 6))
    fig_box.suptitle('Ratios of AIC(SIR-SD/SIR)')
    ax = fig_box.add_subplot()
    ax.boxplot(AIC_ratio,
               showfliers=False
               )
    ax.scatter(np.random.normal(1, 0.04, len(AIC_ratio)), AIC_ratio, color=cmap)
    ax.set_xticklabels([])
    # fig_box.savefig('50Counties/comparison/AIC_box.png', bbox_inches="tight")
    plt.show()


def AIC_scatter():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig = plt.figure()
    fig.suptitle('AIC')
    ax = fig.add_subplot()
    ax.scatter(df['AIC_SD'], df['AIC_SIR'], color=cmap)
    ax.set_xlabel('SIR-SD')
    ax.set_ylabel('SIR')
    line_range = [max(min(df['AIC_SD']), min(df['AIC_SIR'])), min(max(df['AIC_SD']), max(df['AIC_SIR']))]
    ax.plot(line_range, line_range, linestyle=':', color='grey')
    fig.savefig('50Counties/comparison/AIC_scatter.png', bbox_inches="tight")


def AIC_hist():
    df = pd.read_csv('50Counties/comparison/AIC.csv')
    cmap = ['red' if AIC_SD >= AIC_SIR else 'green' for AIC_SD, AIC_SIR in zip(df['AIC_SD'], df['AIC_SIR'])]
    AIC_ratio = list(df['AIC_SD'] / df['AIC_SIR'])
    fig = plt.figure()
    fig.suptitle('Ratio of AIC(SIR-SD/SIR)')
    ax = fig.add_subplot()
    ax.hist(AIC_ratio, edgecolor='black')
    ax.axvline(np.mean(AIC_ratio), linestyle='dashed', color='tab:grey',
               label=f'AVG={round(np.mean(AIC_ratio), 3)}')
    ax.axvline(1, color='red')
    ax.legend()
    fig.savefig('50Counties/comparison/AIC_hist.png', bbox_inches="tight")


if __name__ == '__main__':
    # AIC_compare()
    # AIC_boxplot()
    # AIC_scatter()
    AIC_hist()
