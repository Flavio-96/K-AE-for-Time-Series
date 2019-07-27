import pandas as pd
import numpy as np
import sklearn.preprocessing as pp


def load_data(direc, dataset, perm=True, ratio_train=0.8):
    datadir = direc + '/' + dataset + '/' + dataset
    data_train = np.genfromtxt(datadir + '_TRAIN.tsv', delimiter='\t')
    data_test_val = np.genfromtxt(datadir + '_TEST.tsv', delimiter='\t')[:-1]
    data = np.concatenate((data_train, data_test_val), axis=0)

    N, D = data.shape
    ind_cut = int(ratio_train * N)
    if perm:
        ind = np.random.permutation(N)
    else:
        ind = range(0, N)
    return data[ind[:ind_cut], 1:], data[ind[ind_cut:], 1:], data[ind[:ind_cut], 0], data[ind[ind_cut:], 0]


def rebuild_data(data_train, data_test, labels_train, labels_test):
    all_data = np.concatenate((data_train, data_test))
    all_labels = np.concatenate((labels_train, labels_test)).reshape(-1, 1)
    return np.concatenate((all_labels, all_data), axis=1)


def remove_outlier(dataset):
    df = pd.DataFrame(dataset)
    for col in df:
        low_threshold = df[col].quantile(0.03)
        high_threshold = df[col].quantile(0.97)
        df.loc[df[col] < low_threshold, col] = low_threshold
        df.loc[df[col] > high_threshold, col] = high_threshold
    return df.to_numpy()


def scale_data(dataset):
    min_max_scaler = pp.MinMaxScaler()
    dataset_scaled = min_max_scaler.fit_transform(dataset)
    return dataset_scaled


def plot_dataframe(dataset, title):
    pd.DataFrame(dataset).plot(legend=False, title=title)
