# Modelling Flapping Wing Aerodynamics with PR-SSM

This is the companion code for state-space modelling of flapping wing aerodynamics with Probabilistic Recurrent State-Space Models (PR-SSM) (Doerr et al., ICML 2018). The results of the original study are reported in "State-space aerodynamic model reveals high force control authority and predictability in flapping flight" (**https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0222**). Much of the PR-SSM modelling code was adapted from **https://github.com/boschresearch/PR-SSM**.

## Prerequisites

The PR-SSM code depends on TensorFlow 1.14 and uses TensorFlow 1.x constructs such as `tf.contrib`, so it should be run in a compatible Python 3.7 environment.

Recommended packages:
- `tensorflow==1.14` or `tensorflow-gpu==1.14`
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `tqdm`
- `openpyxl`
- `scikit-learn`

Typical setup:

```bash
conda create -n prssm-fw python=3.7
conda activate prssm-fw
pip install tensorflow-gpu==1.14 numpy scipy matplotlib pandas tqdm openpyxl scikit-learn
```

If GPU support is unavailable, install the CPU build of TensorFlow 1.14 instead.

## Project Layout

Keep the project folder and data folder as siblings. In the current shared layout, this means:

```text
C:\
  MFWF-Copy-of-Flapping-Wing-AERO-PRSSM\
  Data\
```

The code resolves the data folder as `../Data` relative to the project folder. With the current layout, that is `C:\Data`.

Expected data files:
- `..\Data\Input\flapping_wing_aerodynamics.mat`
- `..\Data\Input\flapping_wing_aerodynamics_lasso_fit.mat`
- `..\Data\Comparison of Model Accuracy - Fixed Seed.xlsx`

Generated runs are written under:

```text
..\Data\Output\<run_name>\
```

## Getting Started

1. Open a terminal in the project folder.
2. Activate the TensorFlow 1.14 environment.
3. Confirm `..\Data\Input\` contains the required `.mat` files.
4. Confirm `..\Data\Output\` exists for generated runs.
5. Review [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py) and [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py) before launching a new run.

Run training:

```bash
python Python_PRSSM/DataInformed.py
```

Run postprocessing for the configured default output:

```bash
python Python_PRSSM/outputs/postprocess_model_outputs.py
```

Run the interactive wing animation:

```bash
python Python_PRSSM/outputs/animate_wing_trajectory.py
```

Run the hyperparameter recommendation tool:

```bash
python Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py
```

Run the automation loop:

```bash
python Python_PRSSM/automation_runner.py --max-runs 1
```

## Output Files

Each completed run under `..\Data\Output\` normally contains:
- `training_config.json`
- `training_log.txt`
- checkpoint files such as `best.ckpt*`, `best_test.ckpt*`, and `model.ckpt*`
- `matfiles/training_loss.mat`
- `matfiles/predict_train_n_test.mat`
- postprocessing figures under `figures/`
- `metrics_summary.csv`

Use the training log, loss plots, parity plots, RMSE, R2, and the logged training classification to judge whether a run is overtraining, undertraining, well-balanced, inconclusive, or unavailable.

## Common Issues

- `openpyxl` import errors: install `openpyxl` in the same Python environment used to run the scripts.
- TensorFlow compatibility errors: make sure the environment uses TensorFlow 1.14.
- Missing `.mat` files: confirm the input data is in `..\Data\Input\`.
- Windows path-length errors: keep the project and data folders close to the drive root and avoid very long output folder names.

## Script Map

- [Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py): Main run script for training, retraining, prediction, and postprocessing.
- [Python_PRSSM/automation_runner.py](Python_PRSSM/automation_runner.py): Runs surrogate-guided tuning cycles and logs results to the comparison workbook.
- [Python_PRSSM/experiment_runner.py](Python_PRSSM/experiment_runner.py): Shared experiment configuration, run execution, logging, and workbook integration.
- [Python_PRSSM/database/data_manager.py](Python_PRSSM/database/data_manager.py): Loads, preprocesses, normalizes, and batches the flapping-wing dataset.
- [Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py): Defines the PR-SSM TensorFlow model graph, loss, recognition model, and optimizer.
- [Python_PRSSM/training/trainer.py](Python_PRSSM/training/trainer.py): Runs training, early stopping, checkpointing, and training classification.
- [Python_PRSSM/outputs/outputs.py](Python_PRSSM/outputs/outputs.py): Saves predictions and MATLAB-compatible output artifacts.
- [Python_PRSSM/outputs/postprocess_model_outputs.py](Python_PRSSM/outputs/postprocess_model_outputs.py): Generates summary metrics and figures for a trained model output folder.
- [Python_PRSSM/outputs/animate_wing_trajectory.py](Python_PRSSM/outputs/animate_wing_trajectory.py): Opens the interactive wing-motion visualization.
- [Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py](Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py): Reads the comparison workbook and recommends next hyperparameter settings.
- [Data/Input/README.md](Data/Input/README.md): Describes the required input data files and key variables.
