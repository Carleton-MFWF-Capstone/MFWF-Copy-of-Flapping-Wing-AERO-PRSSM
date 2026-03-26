# Modelling Flapping Wing Aerodynamics with PR-SSM
This is the companion code for the state-space modelling of flapping wing aerodynamics with Probabilistic Recurrent State-Space Models (PR-SSM) (Doerr et al., ICML 2018). The results of this study is reported in the manuscript "State-space aerodynamic model reveals high force control authority and predictability in flapping flight" (**https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0222**). The majority of the code for modelling the aerodynamics with PR-SSM was modified from the source code (**https://github.com/boschresearch/PR-SSM**).

## Prerequisites
The PR-SSM code implemented here depends on Tensorflow 1.14 (Tensorflow-gpu is prefered). The datasets used in this study is available at **https://doi.org/10.5061/dryad.zgmsbccbs**.

To use the code or replicate the results in the manuscript the data sets must be placed in the correct paths. The input files (**flapping_wing_aerodynamics.mat** and **flapping_wing_aerodynamics_lasso.mat**) should be stored in *../Data/Input/* folder. 

The PR-SSM code will generate the new model logs and the data files in the *../Data/Output/**dir_name*** (replace ***dir_name*** with the desired directory name).

The results and the model logs for the model used in the manuscript can be found in the *../Data/Output/20_08_15_IN_LK7Normalized_OUT_FMwoFa5_Dimx12_Epochs1000_512Train/* directory. To perform the postprocessing code provided in *../Matlab_Postprocess/* path, the output data set **predict_train_n_test.mat** should be stored in the *../matfiles/* output directory.

## Getting Started
This section is intended as a handoff guide for the next team member who needs to continue training, postprocessing, or visualization work on this project.

### 1. Clone the repository
Clone or download the repository to a local machine, ideally in a path that is not excessively deep because some output folders and MATLAB files can approach Windows path-length limits.

### 2. Set up the Python environment
The codebase depends on TensorFlow 1.14 and uses TensorFlow 1.x constructs such as `tf.contrib`, so it must be run in a compatible environment rather than a modern TensorFlow 2-only environment.

Recommended core Python packages:
- `tensorflow==1.14` or `tensorflow-gpu==1.14`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tqdm`
- `openpyxl`

Typical workflow:
```bash
conda create -n prssm-fw python=3.7
conda activate prssm-fw
pip install tensorflow-gpu==1.14 numpy scipy matplotlib pandas tqdm openpyxl
```

If GPU support is unavailable, install the CPU build of TensorFlow 1.14 instead.

### 3. Place the required input data in the correct folder
The following files are expected in [Data/Input](Data/Input):
- `flapping_wing_aerodynamics.mat`
- `flapping_wing_aerodynamics_lasso_fit.mat`

The data descriptions are summarized in [Data/Input/README.md](Data/Input/README.md).

### 4. Confirm the repository structure
Before running anything, verify that these folders exist:
- `Data/Input/`
- `Data/Output/`
- `Python_PRSSM/`
- `Python_PRSSM/model/`
- `Python_PRSSM/training/`
- `Python_PRSSM/outputs/`

### 5. Update the training configuration
The main run configuration lives in [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py).

Before launching a new run, check and update:
- `out_dir`: name of the output folder for the new experiment
- `epochs`
- `batch_size`
- `learning_rate`
- model settings such as `dim_x`, `ind_pnt_num`, and `samples`
- any recognition-model or architecture settings in [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py)

Important note:
- The current optimizer in [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py) uses a fixed learning rate taken from `model_config['learning_rate']`.
- Postprocessing is triggered automatically at the end of [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py).

### 6. Run training
From the repository root:
```bash
python Python_PRSSM/DataInformed.py
```

The script will ask:
- whether to start a new simulation
- whether to retrain an old simulation

During training, the code writes:
- `training_log.txt`
- checkpoints such as `best.ckpt`, `best_test.ckpt`, and `model.ckpt`
- MATLAB output files inside `matfiles/`

### 7. Check the generated output folder
Each run writes to a folder under `Data/Output/`.

The most important generated files are:
- `training_config.json`
- `training_log.txt`
- `best.ckpt*`
- `best_test.ckpt*`
- `model.ckpt*`
- `matfiles/training_loss.mat`
- `matfiles/predict_train_n_test.mat`

### 8. Review training behavior
Use the training log and postprocessing outputs to judge whether the run was successful.

Useful indicators:
- final train and test loss trends
- early stopping behavior
- logged `Training diagnosis` from [Python_PRSSM/training/trainer.py](Python_PRSSM/training/trainer.py)
- parity plots and RMSE / R² metrics from postprocessing

### 9. Run postprocessing manually if needed
If you want to rerun postprocessing for an existing output folder, use:
```bash
python Python_PRSSM/outputs/postprocess_model_outputs.py
```

Update the `OUT_DIR` value in that script first, or call its helper from Python for a specific output directory.

### 10. Run the interactive wing animation
To inspect a trained model interactively:
```bash
python Python_PRSSM/outputs/animate_wing_trajectory.py
```

The script will:
- list available output folders with usable MATLAB prediction files
- let you choose the output directory
- let you choose a `.mat` file inside `matfiles/`

### 11. Use the hyperparameter recommendation tool
To analyze the experiment summary workbook and suggest the next run:
```bash
python Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py
```

This script expects the Excel comparison workbook to exist at the configured path and uses:
- average test RMSE
- average test R²
- training classification

to recommend the next hyperparameter set.

### 12. Update the experiment tracking workbook
After each run, add the new experiment to the comparison workbook with:
- architecture settings
- training settings
- average RMSE and R²
- training classification (`Overtraining`, `Undertraining`, `Well-balanced`, `Inconclusive`, or `Unavailable`)
- any notes worth preserving

### 13. Common issues
- `openpyxl` import errors: install `openpyxl` in the same Python environment that is running the script.
- Windows path-length errors: keep the repository close to the drive root when possible and avoid extremely long output folder names.
- TensorFlow compatibility errors: make sure the environment uses TensorFlow 1.14.
- Missing `.mat` files: confirm the input data is in `Data/Input/` and the output `matfiles/` folder was created successfully.

### 14. Recommended first steps for a new team member
1. Read this README and [Data/Input/README.md](Data/Input/README.md).
2. Create and test the Python environment.
3. Verify the input `.mat` files are present.
4. Open [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py) and [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py) to understand the current training configuration.
5. Review one previous output folder in `Data/Output/` to see the expected artifacts.
6. Run the animation and postprocessing tools on an existing output before launching a new training job.

## Purpose of Script
- `Python_PRSSM`
  - [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py): Main run script that sets the training configuration, builds the dataset/model/output objects, launches training, and runs postprocessing.
  - `database`
    - [Python_PRSSM/database/base_ds.py](Python_PRSSM/database/base_ds.py): Defines the base dataset interface and shared dataset behavior used by the training pipeline.
    - [Python_PRSSM/database/data_manager.py](Python_PRSSM/database/data_manager.py): Loads, preprocesses, normalizes, and batches the flapping-wing aerodynamic dataset for train/test use.
  - `model`
    - [Python_PRSSM/model/base_model.py](Python_PRSSM/model/base_model.py): Provides the shared base class structure for TensorFlow models, including graph setup and common model utilities.
    - [Python_PRSSM/model/gp_tf.py](Python_PRSSM/model/gp_tf.py): Implements the TensorFlow Gaussian-process kernel and conditional prediction routines used inside PR-SSM.
    - [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py): Defines the PR-SSM model graph, latent-state propagation, likelihood, loss, recognition model, and optimizer.
    - [Python_PRSSM/model/tf_transform.py](Python_PRSSM/model/tf_transform.py): Contains forward and inverse parameter transforms used to keep constrained model variables numerically valid.
  - `outputs`
    - [Python_PRSSM/outputs/animate_wing_trajectory.py](Python_PRSSM/outputs/animate_wing_trajectory.py): Opens an interactive visualization that animates the wing motion and compares measured versus predicted aerodynamic outputs over a trajectory.
    - [Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py](Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py): Reads the Excel comparison table and recommends new hyperparameter settings by balancing predicted accuracy against training-classification risk.
    - [Python_PRSSM/outputs/outputs.py](Python_PRSSM/outputs/outputs.py): Handles model evaluation outputs by generating predictions, saving MATLAB-compatible result files, and managing output artifacts.
    - [Python_PRSSM/outputs/postprocess_model_outputs.py](Python_PRSSM/outputs/postprocess_model_outputs.py): Postprocesses a trained model output folder into summary metrics and figures such as loss curves, parity plots, and prediction comparisons.
  - `training`
    - [Python_PRSSM/training/trainer.py](Python_PRSSM/training/trainer.py): Runs the training loop, logs train/test loss and timing, saves checkpoints, applies early stopping, and classifies the run as overtraining, undertraining, well-balanced, inconclusive, or unavailable.
- `Data/Input`
  - [Data/Input/inspect_data.py](Data/Input/inspect_data.py): Inspects the raw MATLAB input dataset so you can verify contents, structure, and variables before training.
