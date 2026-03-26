# Modelling Flapping Wing Aerodynamics with PR-SSM
This is the companion code for the state-space modelling of flapping wing aerodynamics with Probabilistic Recurrent State-Space Models (PR-SSM) (Doerr et al., ICML 2018). The results of this study is reported in the manuscript "State-space aerodynamic model reveals high force control authority and predictability in flapping flight" (**https://royalsocietypublishing.org/doi/10.1098/rsif.2021.0222**). The majority of the code for modelling the aerodynamics with PR-SSM was modified from the source code (**https://github.com/boschresearch/PR-SSM**).

## Prerequisites
The PR-SSM code implemented here depends on Tensorflow 1.14 (Tensorflow-gpu is prefered). The datasets used in this study is available at **https://doi.org/10.5061/dryad.zgmsbccbs**.

To use the code or replicate the results in the manuscript the data sets must be placed in the correct paths. The input files (**flapping_wing_aerodynamics.mat** and **flapping_wing_aerodynamics_lasso.mat**) should be stored in *../Data/Input/* folder. 

The PR-SSM code will generate the new model logs and the data files in the *../Data/Output/**dir_name*** (replace ***dir_name*** with the desired directory name).

The results and the model logs for the model used in the manuscript can be found in the *../Data/Output/20_08_15_IN_LK7Normalized_OUT_FMwoFa5_Dimx12_Epochs1000_512Train/* directory. To perform the postprocessing code provided in *../Matlab_Postprocess/* path, the output data set **predict_train_n_test.mat** should be stored in the *../matfiles/* output directory.

## Purpose of Script
[Python_PRSSM/DataInformed.py](Python_PRSSM/DataInformed.py): Main run script that sets the training configuration, builds the dataset/model/output objects, launches training, and runs postprocessing.
[Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py](Python_PRSSM/outputs/optimize_hyperparameters_from_comparison.py): Reads the Excel comparison table and recommends new hyperparameter settings by balancing predicted accuracy against training-classification risk.
[Python_PRSSM/outputs/animate_wing_trajectory.py](Python_PRSSM/outputs/animate_wing_trajectory.py): Opens an interactive visualization that animates the wing motion and compares measured versus predicted aerodynamic outputs over a trajectory.
[Python_PRSSM/outputs/postprocess_model_outputs.py](Python_PRSSM/outputs/postprocess_model_outputs.py): Postprocesses a trained model output folder into summary metrics and figures such as loss curves, parity plots, and prediction comparisons.
[Python_PRSSM/outputs/outputs.py](Python_PRSSM/outputs/outputs.py): Handles model evaluation outputs by generating predictions, saving MATLAB-compatible result files, and managing output artifacts.
[Python_PRSSM/training/trainer.py](Python_PRSSM/training/trainer.py): Runs the training loop, logs train/test loss and timing, saves checkpoints, applies early stopping, and classifies the run as overtraining, undertraining, well-balanced, inconclusive, or unavailable.
[Python_PRSSM/database/base_ds.py](Python_PRSSM/database/base_ds.py): Defines the base dataset interface and shared dataset behavior used by the training pipeline.
[Python_PRSSM/database/data_manager.py](Python_PRSSM/database/data_manager.py): Loads, preprocesses, normalizes, and batches the flapping-wing aerodynamic dataset for train/test use.
[Python_PRSSM/model/base_model.py](Python_PRSSM/model/base_model.py): Provides the shared base class structure for TensorFlow models, including graph setup and common model utilities.
[Python_PRSSM/model/gp_tf.py](Python_PRSSM/model/gp_tf.py): Implements the TensorFlow Gaussian-process kernel and conditional prediction routines used inside PR-SSM.
[Python_PRSSM/model/prssm.py](Python_PRSSM/model/prssm.py): Defines the PR-SSM model graph, latent-state propagation, likelihood, loss, recognition model, and optimizer.
[Python_PRSSM/model/tf_transform.py](Python_PRSSM/model/tf_transform.py): Contains forward and inverse parameter transforms used to keep constrained model variables numerically valid.
[Data/Input/inspect_data.py](Data/Input/inspect_data.py): Inspects the raw MATLAB input dataset so you can verify contents, structure, and variables before training.
