import h5py
import re
import torch
import numpy as np
from sklearn.metrics import r2_score
from nilearn.connectome import ConnectivityMeasure

from envs.base import EnvBase
from bots.static.fmri_autoreg import fMRI_Bot

class Env(EnvBase):

    def __init__(self, args, rank, size):
        self.X, self.Y, self.conn_mat = self.load_data(**args.additional_arguments)
        self.X = torch.Tensor(self.X)
        super().__init__(args, rank, size)


    def initialize_bots(self, args, rank, nb_populations):
        n_ROIs = self.X.shape[1]
        input_length = self.X.shape[-1]
        d_input = n_ROIs * input_length
        d_output = n_ROIs
        self.bots = [fMRI_Bot(d_input, d_output, args, rank)]


    @staticmethod
    def load_data(path, dataset, task_filter, lag, length, stride=1, standardize=False, shuffle=False):
        """
        Load pre-processed data from HDF5 file.

        :param path: path to the HDF5 file
        :type path: str
        :param dataset: dataset to use (e.g. 'friends')
        :type dataset: str
        :param task_filter: regular expression to apply on run names
        :type task_filter: str
        :param lag: lag between last input time point and prediction time point
        :type lag: int
        :param length: length of input time sequence (i.e. number of input time points)
        :type length: int
        :param stride: stride between consecutive input sequences (default=1)
        :type stride: int
        :param standardize: bool (default=False)
        :type standardize: bool
        :param shuffle: wether to shuffle the data (default=False)
        :type shuffle: bool
        """
        data_list = []
        with h5py.File(path, "r") as h5file:
            for key in list(h5file[dataset].keys()):
                if task_filter is None or re.search(task_filter, key):
                    data_list.append(h5file[dataset][key][:])
        if not data_list:
            raise RuntimeError(
                f"Couldn't load data, it might be due to the task filter: {task_filter}"
                )
        if standardize and data_list:
            means = np.concatenate(data_list, axis=0).mean(axis=0)
            stds = np.concatenate(data_list, axis=0).std(axis=0)
            data_list = [(data - means) / stds for data in data_list]
        if shuffle and data_list:
            rng = np.random.default_rng()
            data_list = [rng.shuffle(np.concatenate(data_list, axis=0))]

        correlation_measure = ConnectivityMeasure(kind='correlation')
        conn_mat = correlation_measure.fit_transform([np.concatenate(data_list)])[0]

        X_tot, Y_tot = [], []
        delta = lag - 1
        for data in data_list:
            X = []
            Y = []
            for i in range(0, data.shape[0] - length - delta, stride):
                X.append(np.moveaxis(data[i : i + length], 0, 1))
                Y.append(data[i + length + delta])
            X_tot.append(np.array(X))
            Y_tot.append(np.array(Y))
        X_tot = np.concatenate(X_tot)
        Y_tot = np.concatenate(Y_tot)

        return X_tot, Y_tot, conn_mat


    def run(self, gen_nb):
        [bot] = self.bots
        prediction = bot(self.X).detach().numpy()
        bot_fitness = r2_score(self.Y, prediction)
        return [bot_fitness]
