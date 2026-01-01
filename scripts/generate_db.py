import numpy as np
import pandas as pd
import random
import pickle
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from src.LK import linear_comb
from math import ceil, floor
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold

data_dir = './DB'

os.makedirs(f"{data_dir}/test", exist_ok=True)
os.makedirs(f"{data_dir}/train", exist_ok=True)
os.makedirs(f"{data_dir}/val", exist_ok=True)
os.makedirs(f"{data_dir}/val_imbalanced", exist_ok=True)
os.makedirs(f"{data_dir}/test_imbalanced", exist_ok=True)

df_X = pd.read_csv(data_dir + '/input_dataset_gt.csv',header=None)
df_y = pd.read_csv(data_dir + '/label_dataset_gt.csv',header=None)
#df_X_columns = [A_4000, ... A_400, 'InChI', 'DB_type']
#df_y_columns = ['Alkane','Alkene','Arene', 'Halide', 'Alcohol','Aldehyde','Ketone', 'Carboxylic acid', 'Acyl halide', 'Ester', 'Ether', 'Amine', 'Amide', 'Nitrile', 'Phenol', 'Nitro', 'Class_of_purity','Conc','InChI','State','DB', 'ID']

X = df_X.iloc[:,0:600]
y = df_y.iloc[:,0:22]

all_results = []

def to_smiles(mol_tuple):
    return tuple(Chem.MolToSmiles(mol) for mol in mol_tuple)

def from_smiles(smiles_tuple):
    return tuple(Chem.MolFromSmiles(smiles) for smiles in smiles_tuple)

def find_matches(row_idx,reaction_smarts, mix):
    """
    Function for searching for substance-impurity pairs: the function accepted all examples with characteristic label
    of the main substance (e.g., all acids) as input, and the corresponding impurity was determined based on the reaction_smarts.
    If the search gave a complete match between the InChI string of the impurity and the substance in the DB, the pair was added to the matches.

    Args:
        row_idx (list): A list of column numbers corresponding to functional groups. For example, if we need all the acids, we refer to column 7.
        reaction_smarts (str): Type of reaction in SMARTS string notation.
        mix (bool): A parameter that controls that the main substance index comes first in a pair of indexes.

    Returns:
        matches (list): A list of index pairs in DB of molecules like (idx_main_compound, idx_impurity).
    """
    y_n = y.copy()
    y_n['supposed_impurity'] = None
    for index, row in y_n.iterrows():
        if all(row[i] == 1 for i in row_idx):
            molecule = Chem.MolFromInchi(row.iloc[18])
            reaction = AllChem.ReactionFromSmarts(reaction_smarts)
            products = reaction.RunReactants((molecule,))
            # convert to smiles and back to keep only unique products
            products_smiles = [to_smiles(product_tuple) for product_tuple in products]
            unique_smiles = {tuple(sorted(smiles_tuple)) for smiles_tuple in products_smiles}
            unique_products = [from_smiles(smiles_tuple) for smiles_tuple in unique_smiles]
            if len(unique_products) == 0:
                continue
            success = False
            first_item = unique_products[0]
            mol1, *rest = first_item
            for mol in [mol1, *rest]:
                if mol is None:
                    continue
                try:
                    modified_inchi = str(Chem.MolToInchi(mol))
                    success = True
                    break
                except Exception as e:
                    print(f"Problem: {str(e)}")
            if not success:
                continue
            if modified_inchi == row.iloc[18]:
                continue
            y_n.at[index, 'supposed_impurity'] = modified_inchi

    matches = []

    for idx, row in y_n.iterrows():
        if all(row[i] == 1 for i in row_idx):
            for i, supp_row in y_n[18].items():
                if row['supposed_impurity'] == supp_row:
                    pair = (i, idx) if mix else (idx, i)
                    matches.append(pair)

    #print(f"Found {len(matches)} matches: {matches}")
    return matches

def OneImpurity(row_idx,impurity_inchi):
    """
    Function for searching for substance-impurity pairs: the function returned matches of all examples with characteristic
    labels of the main substance with all conditions (e.g., all aromatic acids) as input, and the corresponding impurity
    was determined based on the impurity_inchi.
    """

    y_n = y.copy()
    index2 = y_n[y_n.iloc[:, 18] == impurity_inchi].index[0]

    matches = []

    for index1, row in y_n.iterrows():
        if all(row[i] == 1 for i in row_idx):
            if index2 != index1:
                pair = (index1, index2)
                matches.append(pair)

    #print(f"Found {len(matches)} matches: {matches}")
    return matches


def OneImpurity_for_any_conditions(row_idx,impurity_inchi):
    """
    Function for searching for substance-impurity pairs: the function returned matches of all examples with characteristic
    labels of the main substance with any conditions (e.g., acids and alcohols and esters) as input, and the corresponding
    impurity was determined based on the impurity_inchi.
    """
    y_n = y.copy()
    index2 = y_n[y_n.iloc[:, 18] == impurity_inchi].index[0]

    matches = []

    for index1, row in y_n.iterrows():
        if any(row[i] == 1 for i in row_idx):
            if index2 != index1:
                pair = (index1, index2)
                matches.append(pair)

    #print(f"Found {len(matches)} matches: {matches}")
    return matches

def split_matches(matches, external_test=None, use_swap=False):
    """
    Splits the substance-impurity pairs in a ratio of 72:13:15 for training, validation and test sets.

    Args:
        matches (list):  A list of index of substance-impurity pairs in SDBS+NIST DB.
        external_test (list): A list of index of substance-impurity pairs in external test set.
        use_swap (bool): A parameter that moving pairs (elements of the list of matches) that are present in the
        external test set from the training set (or validation) to the test set, if necessary

    Returns:
        result_dict (dict): Dictionary containing matches for train, val, test sets.
    """
    result_dict = {}
    n = len(matches)

    # Calculate test size (15% of total)
    test_size = ceil(0.15 * n)

    # Calculate validation size (15% of train_val)
    val_size = floor(0.15 * n)
    val_size = 1 if val_size == 0 else val_size

    for i in range(1, 7):
        train_val, test = train_test_split(matches, test_size=test_size)

        if use_swap:
            if external_test is None:
                raise ValueError("external_test required when use_swap=True")
            train_val, test = swap_elements(train_val, test, external_test)
        train_set, val_set = train_test_split(train_val, test_size=val_size)

        result_dict[f'X_train_matches_{i}'] = train_set.copy()
        result_dict[f'X_val_matches_{i}'] = val_set.copy()
        result_dict[f'X_test_matches_{i}'] = test.copy()

    return result_dict

def swap_elements(train, test, external_test):
    """
    Moves the elements common to train and external_test from train (or val) to test. In their place in train, "clean"
    elements from test are inserted (those that are not in external_test).

    Returns:
        new_train, new_test (list): Updated train and test datasets.
    """
    external_test_set = set(external_test)
    leaked_from_train = [elem for elem in train if elem in external_test_set]

    if not leaked_from_train:
        return train, test

    clean_train = [elem for elem in train if elem not in external_test_set]
    clean_candidates_from_test = [elem for elem in test if elem not in external_test_set]
    num_to_swap = len(leaked_from_train)
    if len(clean_candidates_from_test) < num_to_swap:
        return train, test
    to_move_from_test = clean_candidates_from_test[:num_to_swap]
    new_train = to_move_from_test + clean_train
    new_test = list(test) 
    try:
        first_elem_to_replace = to_move_from_test[0]
        idx = new_test.index(first_elem_to_replace)
        new_test = new_test[:idx] + leaked_from_train + new_test[idx + 1:]
    except ValueError:
        return train, test
    return new_train, new_test

def sampling(X_train, y_train, size, fgs=16):
    """Samples a specified proportion of the training data using only functional group labels for stratification."""
    if not hasattr(X_train, 'iloc'):
        X_train = pd.DataFrame(X_train)
    if not hasattr(y_train, 'iloc'):
        y_train = pd.DataFrame(y_train)
    y_strat = y_train.iloc[:, :fgs]

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=size)#, random_state=42)
    for train_index, sample_index in msss.split(X_train, y_strat):
        X_sample = X_train.iloc[sample_index]
        y_sample = y_train.iloc[sample_index]
    return X_sample, y_sample


def generate_data(df_X, df_y, pairs, repeats, classes, noise):
    """Generate data from pairs of index of molecules using linear_comb"""
    X_total, y_total = [], []
    for index1, index2 in pairs:
        X_sample, y_sample = linear_comb(df_X, df_y, index1, index2, repeats, classes, noise)
        X_total.append(X_sample)
        y_total.append(y_sample)
    return np.vstack(X_total), np.concatenate(y_total)


def write_pairs_to_file(path, pairs, df_y):
    """Write identifiers of molecules used for linear combination to file"""
    with open(path, 'a') as f:
        for x, y in pairs:
            value_x = df_y.iloc[x, 21]
            value_y = df_y.iloc[y, 21]
            f.write(f"{value_x} + {value_y}\n")

def write_list_to_file(path, data):
    with open(path, 'a') as f:
        f.writelines('\n'.join(str(x) for x in data))

def train_val_test(matches, repeats, classes, n_noise, test_repeats=3, test_noise=1, test_repeats2=1,
                   external_test=None, use_swap=False):
    """
    Generate data (linear combinations of spectra) for train, validation, and test sets with balanced and imbalanced
    configurations: split -> generation with linear_comb -> sampling for balanced configurations set (~15 examples per class) ->
    save identifiers of pairs to file -> save generated data to dict.

    Args:
        matches (list): List of match pairs for splitting.
        repeats (int): Number of linear combinations for training data generation.
        classes (list): List with purity class for main compound.
        n_noise (int): Noise level for training data.
        test_repeats (int): Number of repeats for test and validation data (default: 3).
        test_noise (int): Noise level for test data (default: 1, no noise).
        test_repeats2 (int): Number of repeats for balanced test data to get approximately 15 examples per class.
        external_test (list): Optional external test set (default: None).
        use_swap (bool): Whether to use swap in splitting (default: False).

    Returns:
        all_folds (dict): Dictionary containing all six folds data (train, val, val_imbalanced, test, test_imbalanced sets).
    """
    all_folds = split_matches(matches, external_test, use_swap)

    for fold_num in range(1, 7):
        # Get sets for current fold
        test = all_folds[f'X_test_matches_{fold_num}']
        train = all_folds[f'X_train_matches_{fold_num}']
        val = all_folds[f'X_val_matches_{fold_num}']

        # Generate and save imbalanced test set
        all_folds[f'X_test_imbalanced_{fold_num}'], all_folds[f'y_test_imbalanced_{fold_num}'] = generate_data(df_X, df_y, test, test_repeats, classes, test_noise)
        write_pairs_to_file(f"{data_dir}/test_imbalanced/test{classes[0]}_{fold_num}.txt", test, df_y)

        # Generate balanced test set
        X_test_total, y_test_total = generate_data(df_X, df_y, test, test_repeats2, classes, test_noise)

        # Apply sampling based on classes
        size = 16 if classes == [3] else 0.2 if classes == [14] else 0.85 if classes == [4] else None
        if size is not None:
            X_test_selected, y_test_selected = sampling(X_test_total, y_test_total, size)
            test_data = y_test_selected.iloc[:, 21].tolist()
            write_list_to_file(f"{data_dir}/test/test_{classes[0]}_{fold_num}.txt", test_data)
            all_folds[f'X_test_{fold_num}'] = X_test_selected
            all_folds[f'y_test_{fold_num}'] = y_test_selected
        else:
            all_folds[f'X_test_{fold_num}'] = X_test_total
            all_folds[f'y_test_{fold_num}'] = y_test_total
            write_pairs_to_file(f"{data_dir}/test/test_{classes[0]}_{fold_num}.txt", test, df_y)

        #print(f"\nFold {fold_num}:")
        #print(f"Train set ({len(train)} matches): {train}")
        #print(f"Validation set ({len(val)} matches): {val}")
        #print(f"Test set ({len(test)} matches): {test}")

        # Save train and validation set indices
        for path, data in [
            (f"{data_dir}/train/train{classes[0]}_{fold_num}.txt", train),
            (f"{data_dir}/val_imbalanced/val{classes[0]}_{fold_num}.txt", val)
        ]: write_pairs_to_file(path, data, df_y)

        # Generate and save training and validation data
        all_folds[f'X_train_{fold_num}'], all_folds[f'y_train_{fold_num}'] = generate_data(df_X, df_y, train, repeats, classes, n_noise)
        all_folds[f'X_val_imbalanced_{fold_num}'], all_folds[f'y_val_imbalanced_{fold_num}'] = generate_data(df_X, df_y, val, test_repeats, classes, test_noise)

        X_val_total, y_val_total = generate_data(df_X, df_y, test, test_repeats2, classes, test_noise)
        # Apply sampling based on classes
        size = 16 if classes == [3] else 0.2 if classes == [14] else 0.85 if classes == [4] else None
        if size is not None:
            X_val_selected, y_val_selected = sampling(X_val_total, y_val_total, size)
            val_data = y_val_selected.iloc[:, 21].tolist()
            write_list_to_file(f"{data_dir}/val/val_{classes[0]}_{fold_num}.txt", val_data)
            all_folds[f'X_val_{fold_num}'] = X_val_selected
            all_folds[f'y_val_{fold_num}'] = y_val_selected
        else:
            all_folds[f'X_val_{fold_num}'] = X_val_total
            all_folds[f'y_val_{fold_num}'] = y_val_total
            write_pairs_to_file(f"{data_dir}/val/val_{classes[0]}_{fold_num}.txt", test, df_y)

    return all_folds

def split_data(x_df,y_df, fgs, external_test, use_swap):
    """
    Split data of pure substances spectra for train, validation, and test sets with balanced and imbalanced configurations.

    Args:
        x_df, y_df (pd.DataFrame): Input data (spectra) and label data.
        fgs (int): Number of functional groups.
        external_test (list): A list of index of pure substances in external test set.
        use_swap (bool): A parameter that controls that all substances from the external test set, if they were in the train
        or val set, were transferred to the test set.

    Returns:
        data_dictionary (dict): Dictionary containing all six folds data (train, val, val_imbalanced, test, test_imbalanced
        sets) only for pure class.
    """
    data_dictionary = {}
    X = x_df.to_numpy()
    y = y_df.to_numpy()
    y_strat = y_df.iloc[:, :fgs].to_numpy()

    msss = MultilabelStratifiedShuffleSplit(n_splits=2, test_size=0.15)
    for train_val_index, test_index in msss.split(X, y_strat):
        X_train_val, X_test = X[train_val_index], X[test_index]
        y_train_val, y_test = y[train_val_index], y[test_index]

        data_dictionary['X_test_matches'] = x_df.index[test_index].tolist()

    if use_swap:
        if external_test is None:
            raise ValueError("external_test required when use_swap=True")
        train_val_index, test_index = swap_elements(train_val_index, test_index, external_test)

    for fold_num in range(1, 7):
        data_dictionary[f'X_test_matches_{fold_num}'] = x_df.index[test_index].tolist()
        test_index_id = y_df.iloc[test_index, 21].tolist()

        data_dictionary[f'X_test_imbalanced_{fold_num}'] = X_test
        data_dictionary[f'y_test_imbalanced_{fold_num}'] = y_test

        write_list_to_file(f"{data_dir}/test_imbalanced/test0_{fold_num}.txt", test_index_id)

        size = 15
        X_test_total2, y_test_total2 = sampling(X_test, y_test, size)
        test_index_id2 = y_test_total2.iloc[:, 21].tolist()
        write_list_to_file(f"{data_dir}/test/test0_{fold_num}.txt", test_index_id2)

        data_dictionary[f'X_test_{fold_num}'] = X_test_total2
        data_dictionary[f'y_test_{fold_num}'] = y_test_total2

    mskf = MultilabelStratifiedKFold(n_splits=6, shuffle=True)
    for fold_num, (train_index, val_index) in enumerate(mskf.split(X_train_val, y_train_val), 1):
        # Get original indices for current fold
        original_train_index = y_df.iloc[train_val_index[train_index], 21].tolist()
        original_val_index = y_df.iloc[train_val_index[val_index], 21].tolist()

        data_dictionary[f'X_train_matches_{fold_num}'] = original_train_index
        data_dictionary[f'X_val_matches_{fold_num}'] = original_val_index

        X_train, X_val = X_train_val[train_index], X_train_val[val_index]
        y_train, y_val = y_train_val[train_index], y_train_val[val_index]

        data_dictionary[f'X_train_{fold_num}'] = X_train
        data_dictionary[f'y_train_{fold_num}'] = y_train
        data_dictionary[f'X_val_imbalanced_{fold_num}'] = X_val
        data_dictionary[f'y_val_imbalanced_{fold_num}'] = y_val

        size = 15
        X_val_2, y_val_2 = sampling(X_val, y_val, size)
        val_index_id2 = y_val_2.iloc[:, 21].tolist()
        write_list_to_file(f"{data_dir}/val/val0_{fold_num}.txt", val_index_id2)

        data_dictionary[f'X_val_{fold_num}'] = X_val_2
        data_dictionary[f'y_val_{fold_num}'] = y_val_2

        paths = [
            (f"{data_dir}/train/train0_{fold_num}.txt", original_train_index),
            (f"{data_dir}/val_imbalanced/val0_{fold_num}.txt", original_val_index)]

        for path, data in paths:
            with open(path, 'w') as f:
                f.writelines(f"{x}\n" for x in data)

    return data_dictionary

#Clean spectra
test = [83, 125, 275, 1081, 1156, 1158]
dict_0 = split_data(df_X,df_y,16,external_test=test, use_swap=True)
all_results.append(dict_0)

#aldehyde / carboxylic acid
aldehyde2acid = '[CX3H1:1](=O)>>[CX3:1](=O)[OX2H]'
matches_1 = find_matches([5], aldehyde2acid, mix = False)
dict_1 = train_val_test(matches_1, 21, [1],n_noise = 5, test_repeats = 3,test_noise = 1,test_repeats2 =7)
all_results.append(dict_1)

#aldehydes / primary alcohol
alcohol2aldehydes = '[CH2:1][OH]>>[CH:1]=O'
matches_2 = find_matches([4], alcohol2aldehydes, mix = True)
dict_2 = train_val_test(matches_2, 23, [2],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 =5)
all_results.append(dict_2)

# Molecule / water
water_inchi = 'InChI=1S/H2O/h1H2'
test_3 = [(1156,2370)]
matches_3 = OneImpurity_for_any_conditions([7,4,5,9,10,6,14],water_inchi)
dict_3 = train_val_test(matches_3, 2, [3],n_noise = 1, test_repeats = 3,test_noise = 1,test_repeats2 = 1,external_test=test_3, use_swap=True)
all_results.append(dict_3)

# Ether/alcohol
ether2alcohol = '[O:1]([$([C,c]);!$(C=O):2])[$([C,c]);!$(C=O):3] >> [$([C,c]);!$(C=O):3][OH]' #decomposition of ether
test_4 = [(1158,522)]
matches_4 = find_matches([10], ether2alcohol, mix = False)
dict_4 = train_val_test(matches_4, 5, [4],n_noise = 2, test_repeats = 3,test_noise = 1,test_repeats2 = 1, external_test=test_4, use_swap=True)
all_results.append(dict_4)

# Amide/ester
ester2amide = '[O:1]=[C:2][O:3][C,c:4]>>[O:1]=[C:2][NH2]' #ester aminolysis
matches_5 = find_matches([9], ester2amide, mix = True)
dict_5 = train_val_test(matches_5, 15, [5],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 = 5)
all_results.append(dict_5)

#amide/acid
acid2amide = '[CX3:1](=O)[OX2H]>>[CX3:1](=O)[NH2]'
#rdkit has a problem with amide-iminol tautomerism, so this list of pairs (amide_matches) was compiled manually.
amide_matches = [(245,402), (282,1156), (795,1161), (957,2307),(1094,402), (1312,1156), (1418,402),(1437,402),(1446,402),(1606,402), (1690, 345),(1914,402), (2379,2307),  (1028, 5),  (866, 127),  (688, 345)]
dict_6 = train_val_test(amide_matches, 19, [6],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 = 5)
all_results.append(dict_6)

#Molecule / phenol
phenol_inchi = "InChI=1S/C6H6O/c7-6-4-2-1-3-5-6/h1-5,7H"
matches_7 = OneImpurity([15,2],phenol_inchi)
dict_7 = train_val_test(matches_7, 13, [7],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 = 4)
all_results.append(dict_7)

#amine/haloalkane
amine2haloalkane = ['[NH2:1][CX4:2]>>[Cl:1][CX4:2]', '[NH2:1][CX4:2]>>[Br:1][CX4:2]', '[NH2:1][CX4:2]>>[F:1][CX4:2]', '[NH2:1][CX4:2]>>[I:1][CX4:2]']
matches_8 = []
for halogen_type_reaction in amine2haloalkane:
    result = find_matches([11], halogen_type_reaction,mix=False)
    matches_8.extend(result)
dict_8 = train_val_test(matches_8, 12, [8],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 = 4)
all_results.append(dict_8)

#acyl halide/Carboxylic acid
halide2acid = ['[CX3:1](=[OX1])[F]>>[CX3:1](=O)[OX2H]', '[CX3:1](=[OX1])[Cl]>>[CX3:1](=O)[OX2H]',
               '[CX3:1](=[OX1])[Br]>>[CX3:1](=O)[OX2H]', '[CX3:1](=[OX1])[I]>>[CX3:1](=O)[OX2H]']
matches_9 = []
for halogen_type_reaction in halide2acid:
    result = find_matches([8], halogen_type_reaction, mix=False)
    matches_9.extend(result)
dict_9 = train_val_test(matches_9, 23, [9],n_noise = 4, test_repeats = 3,test_noise = 1,test_repeats2 = 7)
all_results.append(dict_9)

#haloalkane/alcohol
haloalkane2alcohol = '[Cl,F,Br,IX1:1][CX4:2]>>[OH:1][CX4:2]'
test_10 = [(83,749)]
matches_10 = find_matches([3], haloalkane2alcohol, False)
dict_10 = train_val_test(matches_10, 8, [10],n_noise = 3, test_repeats = 3,test_noise = 1,test_repeats2 = 2,external_test=test_10, use_swap=True)
all_results.append(dict_10)

#haloalkane/amine
haloalkane2amine = '[Cl,F,Br,IX1:1][CX4:2]>>[NH2:1][CX4:2]'
matches_11 = find_matches([3], haloalkane2amine, False)
dict_11 = train_val_test(matches_11, 12, [11],n_noise = 4, test_repeats = 3,test_noise = 1,  test_repeats2 = 4)
all_results.append(dict_11)

#Primary alcohol / aldehydes
alcohol2aldehydes = '[CH2:1][OH]>>[CH:1]=O'
matches_12 = find_matches([4], alcohol2aldehydes, mix = False)
dict_12 = train_val_test(matches_12, 23, [12],n_noise = 4, test_repeats = 3,test_noise = 1, test_repeats2 = 5)
all_results.append(dict_12)

# Alcohol / ketone
ketone2alcohol = '[C,c:1][C:2](=[O:3])[C,c:4]>>[C,c:1][C:2]([O:3])[C,c:4]' #ketone reduction
test_13 = [(1023,1081)]
matches_13 = find_matches([6], ketone2alcohol,True)
dict_13 = train_val_test(matches_13, 15, [13],n_noise = 4, test_repeats = 3,test_noise = 1, test_repeats2 = 4,external_test=test_13, use_swap=True)
all_results.append(dict_13)

# Ester/alcohol
ester2alcohol = '[O:1]=[C:2][O:3][C,c:4]>>[OH:3][C,c:4]' #ester hydrolysis
test_14 = [(125,2404)]
matches_14 = find_matches([9], ester2alcohol, mix = False)
dict_14 = train_val_test(matches_14, 3, [14],n_noise = 2, test_repeats = 3,test_noise = 1, test_repeats2 = 1,external_test=test_14, use_swap=True)
all_results.append(dict_14)

# Ester/Carboxylic acid
ester2acid = '[O:1]=[C:2][O:3][C,c:4]>>[O:1]=[C:2][OH]' #ester hydrolysis
matches_14_2 = find_matches([9], ester2acid, mix = False)
dict_14_2 = train_val_test(matches_14_2, 3, [14],n_noise = 2, test_repeats = 3,test_noise = 1, test_repeats2 = 1)
all_results.append(dict_14_2)

#nitrile / acid
nitrile2acid = '[C,c:1][CX2]#[NX1]>>[C,c:1][CX3](=O)[OX2H]'
matches_15 = find_matches([13], nitrile2acid, mix = False)
dict_15 = train_val_test(matches_15, 42, [15],n_noise = 5, test_repeats = 3,test_noise = 1, test_repeats2 = 7)
all_results.append(dict_15)

combined_all_folds = {}

for fold_num in range(1, 7):
    x_test = [res[f'X_test_{fold_num}'] for res in all_results]
    y_test = [res[f'y_test_{fold_num}'] for res in all_results]

    x_test2 = [res[f'X_test_imbalanced_{fold_num}'] for res in all_results]
    y_test2 = [res[f'y_test_imbalanced_{fold_num}'] for res in all_results]

    x_trains = [res[f'X_train_{fold_num}'] for res in all_results]
    y_trains = [res[f'y_train_{fold_num}'] for res in all_results]

    x_vals = [res[f'X_val_{fold_num}'] for res in all_results]
    y_vals = [res[f'y_val_{fold_num}'] for res in all_results]

    x_val2 = [res[f'X_val_imbalanced_{fold_num}'] for res in all_results]
    y_val2 = [res[f'y_val_imbalanced_{fold_num}'] for res in all_results]

    combined_all_folds[f'X_train_{fold_num}'] = np.vstack(x_trains)
    combined_all_folds[f'y_train_{fold_num}'] = np.concatenate(y_trains)

    combined_all_folds[f'X_test_{fold_num}'] = np.vstack(x_test)
    combined_all_folds[f'y_test_{fold_num}'] = np.concatenate(y_test)

    combined_all_folds[f'X_test_imbalanced_{fold_num}'] = np.vstack(x_test2)
    combined_all_folds[f'y_test_imbalanced_{fold_num}'] = np.concatenate(y_test2)

    combined_all_folds[f'X_val_{fold_num}'] = np.vstack(x_vals)
    combined_all_folds[f'y_val_{fold_num}'] = np.concatenate(y_vals)

    combined_all_folds[f'X_val_imbalanced_{fold_num}'] = np.vstack(x_val2)
    combined_all_folds[f'y_val_imbalanced_{fold_num}'] = np.concatenate(y_val2)

with open(data_dir + '/processed_dataset.pickle', 'wb') as handle:
    pickle.dump(combined_all_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
