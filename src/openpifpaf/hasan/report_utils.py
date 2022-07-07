from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

now = datetime.now()

sns.set_theme(style="whitegrid")

save_dir = 'report_assets/graphs/'
ap = 'Average Precision (AP)'

"""

MTL section

"""


def main2():
    # rs = np.random.RandomState(365)
    # values = rs.randn(365, 4).cumsum(axis=0)
    # dates = pd.date_range("1 1 2016", periods=365, freq="D")
    # data = pd.DataFrame(values, dates, columns=["A", "B", "C", "D"])
    # data = data.rolling(7).mean()

    # cp_comparison = dict(
    #     data=pd.DataFrame(np.array([
    #         [27.5, 27.9, 27.8, 28.0, 27.8, 26, 26],
    #         [27.5, 27.9, 27.8, 28.0, 27.8, 35, 35],
    #     ]).T.tolist(), [60], columns=['Copy Paste', 'Standard']),
    #     id='cp_aug', title='Copy Paste Augmentation vs Standard Data Augmentation', xlabel='Epochs', ylabel=ap,
    # )

    frozen = dict(
        data=pd.DataFrame(np.array([
            [27.5, 27.9, 27.8, 28.0, 27.8],
            [5.3, 6.9, 6.9, 6.9, 6.9],
            [10.0, 12.3, 12.3, 12.4, 12.5],

        ]).T.tolist(), list(range(100, 150, 10)), columns=['Det', 'Pose AP', 'Pose AR']),
        id='mtl_frozen', title='MTL Frozen Detection Backbone', xlabel='Epochs', ylabel=ap
    )

    # dataset weights
    weights1 = dict(
        data=pd.DataFrame(np.array([
            [27.4, 27.6, 27.5, 27.5, 27.5],
            [2.4, 2.4, 2.5, 2.5, 2.5],
        ]).T.tolist(), [70, 90, 110, 130, 150], columns=['Det', 'Pose']),
        id='mtl_weights_0.3_1', title='MTL Dataset Weights 0.3 1', xlabel='Epochs', ylabel=ap
    )

    graphs = [frozen, weights1]

    for graph in graphs:
        ax = sns.lineplot(data=graph['data'], palette="tab10", linewidth=2.5)
        ax.set(title=graph.get('title'), xlabel=graph.get('xlabel'), ylabel=graph.get('ylabel'))
        plt.savefig(save_dir + f'{graph.get("id")}_plot_{now.strftime("%d_%m_%H:%M:%S")}.jpeg')
        plt.clf()


def main():
    main2()
    # cp_main()
    # detkp_main()


if __name__ == '__main__':
    main()