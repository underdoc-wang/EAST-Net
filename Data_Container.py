import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class DataInput(object):
    def __init__(self, norm_bool:bool, graph_bool:bool, norm_mode:str='minmax'):
        self.norm_bool = norm_bool
        self.norm_mode = norm_mode
        self.norm_list = list()     # initialize a list to store norm stats
        self.graph_bool = graph_bool    # whether the data is graph or grid based

    def load_data(self, data_dir:str):
        npz_data = np.load(data_dir)
        print('Dataset contents:', list(npz_data.keys()))

        if not self.graph_bool:
            modes =[]
            for mode in ['taxi', 'bike', 'scooter']:
                if mode in list(npz_data.keys()):
                    modes.append(npz_data[mode])
            mob = np.concatenate(modes, axis=-1)
        else:
            mob = npz_data['poi']
        if not self.norm_bool:
            pass
        else:
            mob = self.channel_wise_normalize(mob)

        # store data in a dictionary
        dataset = dict()
        dataset['mob'] = mob
        dataset['tcov'] = npz_data['meta_onehot']
        dataset['mask'] = npz_data['mask']

        return dataset

    def normalize(self, x:np.array):
        if self.norm_mode == 'minmax':
            self.x_max, self.x_min = x.max(axis=0), x.min(axis=0)
            x = (x - self.x_min) / (self.x_max - self.x_min)
        elif self.norm_mode == 'z_score':
            self.x_mean, self.x_std = x.mean(axis=0), x.std(axis=0)
            x = (x - self.x_mean) / self.x_std
        else:
            raise ValueError
        return x

    def denormalize(self, x:np.array):
        if self.norm_mode == 'minmax':
            x = (self.x_max - self.x_min) * x + self.x_min
        elif self.norm_mode == 'z_score':
            x = self.x_std * x + self.x_mean
        else:
            raise ValueError
        return x

    def channel_wise_normalize(self, x:np.array):
        assert len(x.shape) in [3, 4]
        # x = x.swapaxes(-1, -2)

        x_norm = []
        for c in range(x.shape[-1]):
            if self.norm_mode == 'minmax':
                flow_c, c_min, c_max = self.minmax_normalize(x[...,c])
                x_norm.append(flow_c)
                self.norm_list.append({'min':c_min, 'max':c_max})
                print(f'channel {c}, min: {c_min}, max: {c_max}')
            elif self.norm_mode == 'z_score':
                flow_c, c_mean, c_std = self.z_score_normalize(x[...,c])
                x_norm.append(flow_c)
                self.norm_list.append({'mean':c_mean, 'std':c_std})
                print(f'channel {c}, mean: {c_mean}, std: {c_std}')
            else:
                raise ValueError

        x_norm = np.stack(x_norm, axis=-1)
        return x_norm # .swapaxes(-1, -2)

    def channel_wise_denormalize(self, y:np.array):
        # y = y.swapaxes(-1, -2)
        assert y.shape[-1] == len(self.norm_list)

        y_denorm = []
        for c in range(y.shape[-1]):
            if (self.norm_bool)&(self.norm_mode=='minmax'):
                y_denorm.append(self.minmax_denormalize(y[...,c], c))
            elif (self.norm_bool)&(self.norm_mode=='z_score'):
                y_denorm.append(self.z_score_denormalize(y[...,c], c))
            else:
                raise ValueError

        y_denorm = np.stack(y_denorm, axis=-1)
        return y_denorm # .swapaxes(-1, -2)

    def minmax_normalize(self, x:np.array):     # normalize to [-1, 1]
        x_max, x_min = x.max(), x.min()
        x = (x - x_min) / (x_max - x_min)
        #x = x * 2 - 1
        return x, x_min, x_max

    def minmax_denormalize(self, x:np.array, c:int):
        #x = (x + 1) / 2
        x = (self.norm_list[c]['max'] - self.norm_list[c]['min']) * x + self.norm_list[c]['min']
        return x

    def z_score_normalize(self, x:np.array):        # normalize to N(0, 1)
        x_mean, x_std = x.mean(), x.std()
        x = (x - x_mean) / x_std
        return x, x_mean, x_std

    def z_score_denormalize(self, x:np.array, c:int):
        x = x * self.norm_list[c]['std'] + self.norm_list[c]['mean']
        return x



class MobDataset(Dataset):
    def __init__(self, inputs:dict, output:torch.Tensor, mode:str, mode_len:dict):
        self.mode = mode
        self.mode_len = mode_len
        self.inputs, self.output = self.prepare_xy(inputs, output)

    def __len__(self):
        return self.mode_len[self.mode]

    def __getitem__(self, item):
        return self.inputs['x_seq'][item], self.inputs['t_x'][item], self.inputs['t_y'][item], self.output[item]

    def prepare_xy(self, inputs:dict, output:torch.Tensor):
        if self.mode == 'train':
            start_idx = 0
        elif self.mode == 'validate':
            start_idx = self.mode_len['train']
        else:       # test
            start_idx = self.mode_len['train']+self.mode_len['validate']

        x = dict()
        x['x_seq'] = inputs['x_seq'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['t_x'] = inputs['t_x'][start_idx : (start_idx + self.mode_len[self.mode])]
        x['t_y'] = inputs['t_y'][start_idx: (start_idx + self.mode_len[self.mode])]
        y = output[start_idx : start_idx + self.mode_len[self.mode]]
        return x, y



class DataGenerator(object):
    def __init__(self, obs_len:int, pred_len, data_split_ratio:tuple):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.data_split_ratio = data_split_ratio

    def split2len(self, data_len:int):
        mode_len = dict()
        mode_len['validate'] = int(self.data_split_ratio[1]/sum(self.data_split_ratio) * data_len)
        mode_len['test'] = int(self.data_split_ratio[2]/sum(self.data_split_ratio) * data_len)
        mode_len['train'] = data_len - mode_len['validate'] - mode_len['test']
        return mode_len

    def get_data_loader(self, data:dict, params:dict):
        feat_dict = dict()
        # flows
        x_seq, y_seq = self.get_feats(data['mob'].reshape(data['mob'].shape[0], params['H']*params['W'], params['C']))
        feat_dict['x_seq'] = torch.from_numpy(np.asarray(x_seq)).float().to(params['device']) if params['device'].startswith('cuda') else torch.from_numpy(np.asarray(x_seq)).float()
        y_seq = torch.from_numpy(np.asarray(y_seq)).float().to(params['device']) if params['device'].startswith('cuda') else torch.from_numpy(np.asarray(y_seq)).float()
        # time covariates
        t_x, t_y = self.get_feats(data['tcov'])
        feat_dict['t_x'] = torch.from_numpy(np.asarray(t_x)).float().to(params['device']) if params['device'].startswith('cuda') else torch.from_numpy(np.asarray(t_x)).float()
        feat_dict['t_y'] = torch.from_numpy(np.asarray(t_y)).float().to(params['device']) if params['device'].startswith('cuda') else torch.from_numpy(np.asarray(t_y)).float()

        mode_len = self.split2len(data_len=y_seq.shape[0])
        data_loader = dict()        # data_loader for [train, validate, test]
        for mode in ['train', 'validate', 'test']:
            dataset = MobDataset(inputs=feat_dict, output=y_seq,
                                 mode=mode, mode_len=mode_len)
            data_loader[mode] = DataLoader(dataset=dataset, batch_size=params['batch_size'], shuffle=False, drop_last=True)     # drop incomplete batch
        # data loading default: single-processing | for multi-processing: num_workers=pos_int or pin_memory=True (GPU)
        return data_loader

    def get_feats(self, data:np.array):
        x, y = [], []
        for i in range(self.obs_len, data.shape[0]-self.pred_len):
            x.append(data[i-self.obs_len : i])
            y.append(data[i : i+self.pred_len])
        return x, y

