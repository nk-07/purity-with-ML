import pandas as pd
import numpy as np

def linear_comb(X_train, y_train, index1, index2, n, classes, n_noise):
    """
    Creates linear combinations of two spectra with added noise according to work: Digital Discovery, 2022,1, 35-44 [10.1039/D1DD00027F]

    Args:
        X_train, y_train (pd.DataFrame): Input data (spectra) and label data.
        index1 (int): Index of the first pure component in X_train.
        index2 (int): Index of the second pure component in X_train.
        classes (list): Purity class of the first component.
        n (int): Number of unique combinations to generate.
        n_noise (int): Number of noise levels to apply to each combination.

    Returns:
        X (np.array): An array with the generated spectra with noise and metadata.
        y (np.array): An array with the corresponding labels.
    """
    X_train = X_train.iloc[:, :-2]
    nf = np.linspace(0, 1, n_noise)  #noise factors

    pure_components = X_train.iloc[[index1, index2], :].values.T  #shape: (num_wavelengths, 2)

    conc_c2 = np.random.uniform(0.05, 0.25, n)
    concentrations = np.stack((1 - conc_c2, conc_c2), axis=1)  # shape: n x 2: [conc1, conc2]

    base_spectra = concentrations @ pure_components.T #shape: (n x num_wavelengths)
    all_spectra = np.tile(base_spectra, (n_noise, 1))  #shape: (n * n_noise) x num_wavelengths

    max_peaks = np.max(base_spectra, axis=1)
    full_max_peaks = np.tile(max_peaks, n_noise)  # shape: (n * n_noise,)

    full_nf = np.repeat(nf, n)  # shape: (n * n_noise,)

    # Generate noise: base spectra + (random_coeff * max peak * noise factor)
    noise = np.random.uniform(-0.02, 0.02, size=all_spectra.shape) * full_max_peaks[:, np.newaxis] * full_nf[:, np.newaxis]

    # Add noise and clip to the range [0, 1]
    all_spectra += noise
    all_spectra = np.clip(all_spectra, 0, 1)

    result_df = pd.DataFrame(all_spectra)
    generated_data = result_df.assign(
        NF = full_nf.flatten(),
        Conc = np.tile((1 - conc_c2).flatten(), n_noise),
        Inchi = 0,
        DB_Type = 0)

    y_sample_total = pd.DataFrame()
    for index, row in generated_data.iterrows():
        conc = row['Conc']
        comp_1 = y_train.iloc[index1]
        comp_2 = y_train.iloc[index2]

        combined_row = []

        num_cls = len(comp_1) - 6
        #disjunction for functional group set
        for i in range(num_cls):
            combined_row.append(max(comp_1[i], comp_2[i]))
        combined_row.append(classes[0])
        combined_row.append(np.round(conc,4))
        combined_row.append(comp_1.iloc[-4] + ' + ' + comp_2.iloc[-4])
        combined_row.append(comp_1.iloc[-3] + ' + ' + comp_2.iloc[-3])
        combined_row.append(comp_1.iloc[-2] + ' + ' + comp_2.iloc[-2])
        combined_row.append(comp_1.iloc[-1] + ' + ' + comp_2.iloc[-1])

        y_sample_temp = pd.DataFrame([combined_row])
        y_sample_total = pd.concat([y_sample_total, y_sample_temp], ignore_index=True)

        generated_data['Inchi'] = comp_1.iloc[-4] + ' + ' + comp_2.iloc[-4]
        generated_data['DB_Type'] = comp_1.iloc[-2] + ' + ' + comp_2.iloc[-2]

    X_sample_total = generated_data.drop(columns=['Conc', 'NF'])
  
    X = X_sample_total.to_numpy()
    y = y_sample_total.to_numpy()
    
    return X, y
