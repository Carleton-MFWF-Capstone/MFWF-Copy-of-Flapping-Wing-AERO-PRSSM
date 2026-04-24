import scipy.io
import numpy as np
from pathlib import Path
from .base_ds import BaseDS


class DataManager(BaseDS):
    def __init__(self, seq_len, seq_stride, in_dir):
        super(DataManager, self).__init__(seq_len, seq_stride, in_dir)

    @staticmethod
    def load_ds(filename, print_title=True, title='No Title', dtype=np.float64):
        ds = scipy.io.loadmat(filename)
        if print_title:
            print('Loaded Dataset ' + '' + title)
        u = ds['ds_u'].astype(dtype)
        y = ds['ds_y'].astype(dtype)
        u_raw = ds['ds_u_raw'].astype(dtype)
        y_raw = ds['ds_y_raw'].astype(dtype)
        pos = ds['ds_pos'].astype(dtype)
        mean_u = ds['ds_mean_u'].astype(dtype)
        mean_y = ds['ds_mean_y'].astype(dtype)
        std_u = ds['ds_std_u'].astype(dtype)
        std_y = ds['ds_std_y'].astype(dtype)

        return u, y, u_raw, y_raw, pos, mean_u, mean_y, std_u, std_y


class FlappingWingAerodynamics(DataManager):
    def __init__(self, seq_len, seq_stride, in_dir, validation_fraction=0.1):
        '''
        The input/output pairs are already normalized. Predictions will be for the normalized outputs. Can be
        denormalized in postprocessing.
        '''

        super(FlappingWingAerodynamics, self).__init__(seq_len, seq_stride, in_dir)
        self.title = 'Aerodynamics with GPSSM'
        Ntrain = 512
        path = Path(self.data_path) / 'flapping_wing_aerodynamics.mat'
        u, y, u_raw, y_raw, pos, mean_u, mean_y, std_u, std_y = self.load_ds(str(path), title=self.title)

        u_train_full = np.stack(u[:Ntrain], axis=0)
        y_train_full = np.stack(y[:Ntrain], axis=0)
        pos_train_full = np.stack(pos[:Ntrain], axis=0)
        u_test = np.stack(u[Ntrain:], axis=0)
        y_test = np.stack(y[Ntrain:], axis=0)
        pos_test = np.stack(pos[Ntrain:], axis=0)
        self.dim_u = u_train_full.shape[2]
        self.dim_y = y_train_full.shape[2]

        val_count = int(round(u_train_full.shape[0] * float(validation_fraction)))
        val_count = min(max(val_count, 1), max(u_train_full.shape[0] - 1, 1))
        train_count = u_train_full.shape[0] - val_count

        u_train = u_train_full[:train_count]
        y_train = y_train_full[:train_count]
        pos_train = pos_train_full[:train_count]
        u_val = u_train_full[train_count:]
        y_val = y_train_full[train_count:]
        pos_val = pos_train_full[train_count:]

        self.train_in = u_train
        self.train_out = y_train
        self.train_pos = pos_train
        self.val_in = u_val
        self.val_out = y_val
        self.val_pos = pos_val
        self.test_in = u_test
        self.test_out = y_test
        self.test_pos = pos_test
        self.mean_in = mean_u
        self.mean_out = mean_y
        self.std_in = std_u
        self.std_out = std_y
        self.create_batches()
