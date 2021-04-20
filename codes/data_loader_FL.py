from base import BaseDataGenerator
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pyplot import savefig


class DataGenerator(BaseDataGenerator):
    def __init__(self, config, num_client):
        super(DataGenerator, self).__init__(config)
        # load data here: generate 3 state variables: train_set, val_set and test_set
        self.num_client = num_client # sua dong nay
        self.load_NAB_dataset(self.config['dataset'], self.config['y_scale'])

    def load_NAB_dataset(self, dataset, y_scale=6):
        if dataset == 'ecg':
            data_dir = '../datasets/NAB-known-anomaly/'
            data = np.load(data_dir + dataset + '_{}.npz'.format(self.num_client))

            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            stride_ori = data['training'].reshape((-1, self.config['n_channel'])).strides
            strides = np.insert(stride_ori, 0, stride_ori[0], axis = 0)
            print("N_TRAIN_SAMPLE: ", n_train_sample)
            # dataclient = data['training']
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            shape = [n_train_vae, self.config['l_win'], self.config['n_channel']]

            rolling_windows = np.lib.stride_tricks.as_strided(data['training'],shape, strides, writeable = False)
            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            # if self.config['n_channel']==1:
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            # else: #them dong nay
            #     self.train_set_vae = dict(data=rolling_windows[idx_train])
            #     self.val_set_vae = dict(data=rolling_windows[idx_val])
            #     self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
                n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
                n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
                shape = [n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel']]
                strides = np.insert(stride_ori, 0, [stride_ori[0]*self.config['l_win'], stride_ori[0]*self.config['l_win']], axis = 0)
                cur_lstm_seq = np.lib.stride_tricks.as_strided(data['training'][k:],shape, strides, writeable = False)
                if k == 0:
                    lstm_seq = cur_lstm_seq
                else:
                    lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            # if self.config['n_channel'] == 1: #them dong nay
            #self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
            #self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
            # else:
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])
            # print("shape of train, val set lstm:",self.train_set_lstm['data'].shape,self.val_set_lstm['data'].shape)
        elif 'scada' in dataset:
            data_dir = '../datasets/NAB-known-anomaly/{}/'.format(self.config['dataset'])
            data = np.load(data_dir + dataset + '_{}.npz'.format(self.num_client))

            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            stride_ori = data['training'].reshape((-1, self.config['n_channel'])).strides
            strides = np.insert(stride_ori, 0, stride_ori[0], axis = 0)
            print("N_TRAIN_SAMPLE: ", n_train_sample)
            # dataclient = data['training']
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            shape = [n_train_vae, self.config['l_win'], self.config['n_channel']]

            rolling_windows = np.lib.stride_tricks.as_strided(data['training'],shape, strides, writeable = False)
            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            # if self.config['n_channel']==1:
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            # else: #them dong nay
            #     self.train_set_vae = dict(data=rolling_windows[idx_train])
            #     self.val_set_vae = dict(data=rolling_windows[idx_val])
            #     self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
                n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
                n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
                shape = [n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel']]
                strides = np.insert(stride_ori, 0, [stride_ori[0]*self.config['l_win'], stride_ori[0]*self.config['l_win']], axis = 0)
                cur_lstm_seq = np.lib.stride_tricks.as_strided(data['training'][k:],shape, strides, writeable = False)
                if k == 0:
                    lstm_seq = cur_lstm_seq
                else:
                    lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            # if self.config['n_channel'] == 1: #them dong nay
            #self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
            #self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
            # else:
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])
        else:
            data_dir = '../datasets/NAB-known-anomaly/'
            data = np.load(data_dir + dataset + '.npz')
            # slice training set into rolling windows
            n_train_sample = len(data['training'])
            stride_ori = data['training'].reshape((-1, self.config['n_channel'])).strides
            strides = np.insert(stride_ori, 0, stride_ori[0], axis = 0)
            print("N_TRAIN_SAMPLE: ", n_train_sample)
            # dataclient = data['training']
            n_train_vae = n_train_sample - self.config['l_win'] + 1
            shape = [n_train_vae, self.config['l_win'], self.config['n_channel']]

            rolling_windows = np.lib.stride_tricks.as_strided(data['training'],shape, strides, writeable = False)
            # create VAE training and validation set
            idx_train, idx_val, self.n_train_vae, self.n_val_vae = self.separate_train_and_val_set(n_train_vae)
            # if self.config['n_channel']==1:
            self.train_set_vae = dict(data=np.reshape(rolling_windows[idx_train],(-1, self.config['l_win'], self.config['n_channel'])))
            self.val_set_vae = dict(data=np.reshape(rolling_windows[idx_val],(-1, self.config['l_win'], self.config['n_channel'])))
            self.test_set_vae = dict(data=np.reshape(rolling_windows[idx_val[:self.config['batch_size']]], (-1, self.config['l_win'], self.config['n_channel'])))
            # else: #them dong nay
            #     self.train_set_vae = dict(data=rolling_windows[idx_train])
            #     self.val_set_vae = dict(data=rolling_windows[idx_val])
            #     self.test_set_vae = dict(data=rolling_windows[idx_val[:self.config['batch_size']]])
            print("shape of train,val,test set vae:",self.train_set_vae['data'].shape,\
                self.val_set_vae['data'].shape,self.test_set_vae['data'].shape)

            # create LSTM training and validation set
            for k in range(self.config['l_win']):
                n_not_overlap_wins = (n_train_sample - k) // self.config['l_win']
                n_train_lstm = n_not_overlap_wins - self.config['l_seq'] + 1
                shape = [n_train_lstm, self.config['l_seq'], self.config['l_win'], self.config['n_channel']]
                strides = np.insert(stride_ori, 0, [stride_ori[0]*self.config['l_win'], stride_ori[0]*self.config['l_win']], axis = 0)
                cur_lstm_seq = np.lib.stride_tricks.as_strided(data['training'][k:],shape, strides, writeable = False)
                if k == 0:
                    lstm_seq = cur_lstm_seq
                else:
                    lstm_seq = np.concatenate((lstm_seq, cur_lstm_seq), axis=0)
            n_train_lstm = lstm_seq.shape[0]
            idx_train, idx_val, self.n_train_lstm, self.n_val_lstm = self.separate_train_and_val_set(n_train_lstm)
            # if self.config['n_channel'] == 1: #them dong nay
            #self.train_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_train], -1))
            #self.val_set_lstm = dict(data=np.expand_dims(lstm_seq[idx_val], -1))
            # else:
            self.train_set_lstm = dict(data=lstm_seq[idx_train])
            self.val_set_lstm = dict(data=lstm_seq[idx_val])



    def plot_time_series(self, data, time, data_list):
        fig, axs = plt.subplots(1, 4, figsize=(18, 2.5), edgecolor='k')
        fig.subplots_adjust(hspace=.8, wspace=.4)
        axs = axs.ravel()
        for i in range(4):
            axs[i].plot(time / 60., data[:, i])
            axs[i].set_title(data_list[i])
            axs[i].set_xlabel('time (h)')
            axs[i].set_xlim((np.amin(time) / 60., np.amax(time) / 60.))
        savefig(self.config['result_dir'] + '/raw_training_set_normalised.pdf')
