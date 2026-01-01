import pandas as pd
import numpy as np
import pickle
import random
from scipy import interpolate
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

data_dir = './DB'

def horizontal_aug(y, points,direction='both'):
    """Shift spectrum left or right by a few wavenumbers."""
    if direction == 'left':
        shift = list(range(-3, 0))  # Only negative shifts: [-3, ..., -1]
    elif direction == 'right':
        shift = list(range(1, 4))  # Only positive shifts: [1, ..., 3]
    elif direction == 'both':
        shift = list(range(-3, 4))  # Both directions: [-3, ..., 3]
        shift.remove(0)
    x = np.linspace(4000, 400, points)
    x_new = np.linspace(4000, 400, 3600)
    f = interpolate.interp1d(x, y, kind='slinear')
    y = f(x_new)
    select = random.choice(shift)
    if select > 0:
        # Delete last few.
        y_r = y[:-select]
        for i in range(select):
            y_r = np.insert(y_r, 1, y[0])
        y = y_r
    elif select < 0:
        y_l = y
        for i in range(select*-1):
            y_l = np.delete(y_l, 0)
        last = y_l[len(y_l)-1]
        for i in range(select*-1):
            y_l = np.append(y_l, last)
        y = y_l
    f = interpolate.interp1d(x_new, y, kind='slinear')
    y = f(x)
    return y

def sampling(X_train, y_train, size, fgs=16):
    """Samples a specified proportion of the training data using only first fgs columns for stratification."""
    y_strat = y_train.iloc[:, :fgs]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size)
    for train_index, sample_index in msss.split(X_train, y_strat):
        X_sample = X_train.iloc[sample_index]
        y_sample = y_train.iloc[sample_index]
    return X_sample, y_sample

def augment_data(direction):
    def wrapper(row):
        return horizontal_aug(row, points=600, direction=direction)

    return wrapper

with open(data_dir + '/processed_dataset.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)

data_dictionary = {}
for fold_num in range(1, 7):
    X_train = pd.DataFrame(dict_data[f'X_train_{fold_num}'])
    y_train = pd.DataFrame(dict_data[f'y_train_{fold_num}'])

    unique_values = y_train.iloc[:, 16].unique()

    mask = ~y_train.iloc[:, 16].isin([0, 3, 14])
    X_train_2 = X_train.loc[mask].reset_index(drop=True)
    y_train_2 = y_train.loc[mask].reset_index(drop=True)

    spectra = X_train_2.iloc[:, :-2].values.astype(float)
    meta = X_train_2.iloc[:, -2:]

    augmented_left = np.array([augment_data('left')(row) for row in spectra])
    augmented_right = np.array([augment_data('right')(row) for row in spectra])

    X_aug_left = pd.concat([pd.DataFrame(augmented_left), meta], axis=1)
    X_aug_right = pd.concat([pd.DataFrame(augmented_right), meta], axis=1)

    def process_class(class_value, sample_ratio, direction):
        """augmentation is only for a percentage of the entire class"""
        mask = (y_train.iloc[:, 16] == class_value)
        X_class = X_train.loc[mask].reset_index(drop=True)
        y_class = y_train.loc[mask].reset_index(drop=True)

        X_sampled, y_sampled = sampling(
            X_class, y_class,
            size=sample_ratio,
            fgs=16)

        X_sampled = X_sampled.reset_index(drop=True)
        y_sampled = y_sampled.reset_index(drop=True)

        spectra = X_sampled.iloc[:, :-2].values.astype(float)
        meta = X_sampled.iloc[:, -2:]

        augmented = np.array([augment_data(direction)(row) for row in spectra])
        return pd.concat([pd.DataFrame(augmented), meta], axis=1), y_sampled

    X_aug_0, y_sampled_0 = process_class(0, 0.5, 'both')
    X_aug_14, y_sampled_14 = process_class(14, 0.22, 'both')
    X_aug_3, y_sampled_3 = process_class(3, 0.3, 'both')

    X_combined = pd.concat([
        X_train,
        X_aug_left,
        X_aug_right,
        X_aug_0,
        X_aug_14,
        X_aug_3
    ], ignore_index=True)

    y_combined = pd.concat([
        y_train,
        y_train_2,  
        y_train_2, 
        y_sampled_0,
        y_sampled_14,
        y_sampled_3
    ], ignore_index=True)

    #save results
    data_dictionary[f'X_val_{fold_num}'] = np.asarray(dict_data[f'X_val_{fold_num}'])
    data_dictionary[f'y_val_{fold_num}'] = np.asarray(dict_data[f'y_val_{fold_num}'])

    data_dictionary[f'X_test_{fold_num}'] = np.asarray(dict_data[f'X_test_{fold_num}'])
    data_dictionary[f'y_test_{fold_num}'] = np.asarray(dict_data[f'y_test_{fold_num}'])

    data_dictionary[f'X_test_imbalanced_{fold_num}'] = np.asarray(dict_data[f'X_test_imbalanced_{fold_num}'])
    data_dictionary[f'y_test_imbalanced_{fold_num}'] = np.asarray(dict_data[f'y_test_imbalanced_{fold_num}'])

    data_dictionary[f'X_val_imbalanced_{fold_num}'] = np.asarray(dict_data[f'X_val_imbalanced_{fold_num}'])
    data_dictionary[f'y_val_imbalanced_{fold_num}'] = np.asarray(dict_data[f'y_val_imbalanced_{fold_num}'])

    data_dictionary[f'X_train_{fold_num}'] = np.asarray(X_combined)
    data_dictionary[f'y_train_{fold_num}'] = np.asarray(y_combined)

with open(data_dir + '/augmented_dataset.pickle', 'wb') as handle:
    pickle.dump(data_dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

