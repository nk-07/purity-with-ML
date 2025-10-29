import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        self.mode = mode
        self.data_dir = data_dir
        
        with open(data_dir + 'augmented_dataset.pickle', 'rb') as handle:
            self.dict_data = pickle.load(handle)

        self.df_X = pd.read_csv("./external_data/input_dataset.csv",header=None)
        self.df_y = pd.read_csv("./external_data/label_dataset.csv",header=None)
        
        if self.mode == 'train':
            self.X = np.asarray(pd.DataFrame(self.dict_data['X_train_1']).iloc[:, :-2]).astype('float32').reshape(-1, 1, 600)
            self.y = np.asarray(pd.DataFrame(self.dict_data['y_train_1']).iloc[:,:-6]).astype('float32')
            self.y_p2 = np.asarray(pd.DataFrame(self.dict_data[f'y_train_1']).iloc[:, 16]).astype('float32')
        elif self.mode == 'val':
            self.X = np.asarray(pd.DataFrame(self.dict_data['X_val_imbalanced_1']).iloc[:, :-2]).astype('float32').reshape(-1, 1, 600)
            self.y = np.asarray(pd.DataFrame(self.dict_data['y_val_imbalanced_1']).iloc[:,:-6]).astype('float32')
            self.y_p2 = np.asarray(pd.DataFrame(self.dict_data[f'y_val_imbalanced_1']).iloc[:, 16]).astype('float32')
        elif self.mode == 'test':
            self.X = np.asarray(pd.DataFrame(self.dict_data['X_test_1']).iloc[:, :-2]).astype('float32').reshape(-1, 1, 600)
            self.y = np.asarray(pd.DataFrame(self.dict_data['y_test_1']).iloc[:,:-6]).astype('float32')
            self.y_p2 = np.asarray(pd.DataFrame(self.dict_data[f'y_test_1']).iloc[:, 16]).astype('float32')
        elif self.mode == 'test_real':
            self.X = np.asarray(self.df_X.iloc[:, :-2]).astype('float32').reshape(-1, 1, 600)
            self.y = np.asarray(self.df_y.iloc[:,:-6]).astype('float32')
            self.y_p2 = np.asarray(self.df_y.iloc[:, 16]).astype('float32')
        elif self.mode == 'test_all':
            self.X = np.asarray(pd.DataFrame(self.dict_data['X_test_imbalanced_1']).iloc[:, :-2]).astype('float32').reshape(-1, 1, 600)
            self.y = np.asarray(pd.DataFrame(self.dict_data['y_test_imbalanced_1']).iloc[:,:-6]).astype('float32')
            self.y_p2 = np.asarray(pd.DataFrame(self.dict_data[f'y_test_imbalanced_1']).iloc[:, 16]).astype('float32')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y_fg = torch.from_numpy(self.y[idx]) #multi-label classification target
        y_pur = torch.tensor(self.y_p2[idx], dtype=torch.long)  #multi-class classification target
        
        return x, {
            'fg': y_fg,
            'purity': y_pur
        }

