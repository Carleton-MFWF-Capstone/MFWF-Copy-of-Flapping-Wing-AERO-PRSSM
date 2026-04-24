Store the flapping wing aerodynamics input data here.

Expected files:
- `flapping_wing_aerodynamics.mat`
- `flapping_wing_aerodynamics_lasso_fit.mat`

`flapping_wing_aerodynamics.mat` includes the time history of:
- Euler angles of the wing: `ds_pos` (stroke, deviation, rotation)
- Kinematic variables derived from `ds_pos`: `ds_u_raw` (7 variables; see the paper for definitions and order)
- Aerodynamic forces and moments: `ds_y_raw` (5 variables; see the paper for definitions and order)
- Standardized versions of `ds_u_raw` and `ds_y_raw`: `ds_u` and `ds_y`, with means `ds_mean_u` and `ds_mean_y`, and standard deviations `ds_std_u` and `ds_std_y`

`flapping_wing_aerodynamics_lasso_fit.mat` includes:
- Kinematic features derived from `ds_u_raw`: `ds_uLR_raw` (11 variables; see the paper for definitions and order)
- Aerodynamic forces and moments: `ds_y`
- Lasso model fitted to `ds_y` with `ds_uLR_raw` as the input: `LRmodel`
- Lasso predictions for all trajectories: `lr_y`
