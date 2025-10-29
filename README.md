This repository contains the scripts and data for reproducing the work "IR-assisted predicting of impurities in chemicals using machine learning: towards smart self-driving laboratory". 

## Repository Structure

The repository is organized as follows:

- `data/` - Contains identifiers of molecules used for create linear-combination-data for train, validation and test.
- `data/input_index/` - Contains identifiers of molecules from NIST and SDBS databases.
- `external_data/` - Contains input and label data for 11 spectra of mixtures and 6 spectra of pure compound.
- `scripts/` - Contains python scripts to reproduce this work.
- `scripts/src/` - Contains utils.

## Installation

To set up the project environment, please follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nk-07/purity-with-ML.git
   cd purity-with-ML
   ```

2. **Install dependencies:**

   Create the conda environment from the corresponding YAML configuration file:

   - generate data -> `project_env.yml`
   - train model -> `libmtl_env.yml`

   ```bash
   conda env create -f [name]_env.yml
   ```

## Steps

1. **Data collection:**

   Extraction of spectral and molecular data from the NIST and SDBS databases by unique identifiers from nist.csv and sdbs.csv files. The "data/input_index/" directory in github repository contains filetered identifiers of molecules.

2. **Data preprocessing:**

   Please refer to [1] for preprocessing.py script. We rewrite it as follows: extraction of intensities -> bringing the intensities to the 4000-400 cm-1 range to get a series of 600 points -> convert from transmission to absorption -> baseline correction.


3. **Data generating:**

   Generate the linear combinations with incorporated noise for substance-impurity pairs, which were found in database of pure compound by SMARTS reaction notation.

   ```bash
   python generate_db.py
   ```

4. **Data augmentation:**

   Applied horizontal shifting to balance the train set on purity prediction task.

   ```bash
   python data_augmentation.py
   ```

5. **Hyperparametr optimization:**

   Please refer to [1] for hyperparametr_optimization.py script.


6. **Train the model:**

   Train the model with implementation of multitask learning from libmtl[2]. 

   ```bash
   python main.py --weighting EW --arch MMoE --num_experts 4 --lr 1e-4 --img_size 600  --save_path ./model/
   ```

7. **Evaluate the model:**

   Select mode='test' when prepare dataloader in the main.py file for testing on a balanced test set, mode='test_all' for testing on an imbalanced test set, mode='test_real' for testing on an external test set.

   ```bash
   python main.py --weighting EW --arch MMoE --num_experts 4 --img_size 600  --load_path ./model/ --mode test
   ```

## References

[1] https://github.com/gj475/irchracterizationcnn

[2] https://github.com/median-research-group/LibMTL

