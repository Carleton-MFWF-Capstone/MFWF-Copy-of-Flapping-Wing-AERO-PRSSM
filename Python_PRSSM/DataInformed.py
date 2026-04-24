import json
import time
from pathlib import Path

from experiment_runner import (
    build_experiment_config,
    get_default_output_root,
    load_training_config,
    resolve_output_dir,
    run_experiment,
)


DEFAULT_OUTPUT_NAME = "Output_folder_earlystopX"


def main():
    default_overrides = {"output_name": DEFAULT_OUTPUT_NAME}
    default_config = build_experiment_config(default_overrides)
    default_out_dir = resolve_output_dir(default_config, output_name=DEFAULT_OUTPUT_NAME)
    config_path = default_out_dir / "training_config.json"

    answer = input('Start a new simulation? [Y/N]\n').strip().upper()
    if answer in {"Y", "YES"}:
        print('Starting a new simulation. The config files are updated...')
        result = run_experiment(
            overrides=default_overrides,
            output_name=DEFAULT_OUTPUT_NAME,
            retrain=False,
            train=True,
            run_postprocess=True,
        )
        print('Number of epochs:' + str(default_config['epochs']))
        print('Latent state dim:' + str(default_config['dim_x']))
        print('Outputs saved to:' + result['out_dir'])
        time.sleep(2)
        return

    if not config_path.exists():
        raise FileNotFoundError(
            "Could not find an existing training_config.json at {}".format(config_path)
        )

    model_config = load_training_config(config_path)
    print('Loaded existing configuration from {}'.format(config_path))
    print('Number of epochs:' + str(model_config['epochs']))
    print('Latent state dim:' + str(model_config['dim_x']))
    time.sleep(2)

    answer2 = input('Retrain? [Y/N]\n').strip().upper()
    if answer2 in {"Y", "YES"}:
        print('Continuing an old simulation. The configuration is uploaded...')
        retrain_output_name = Path(model_config['out_dir']).name.rstrip("/\\") + '_RT'
        overrides = dict(model_config)
        overrides["output_root"] = str(get_default_output_root())
        overrides["output_name"] = retrain_output_name
        overrides["in_dir"] = model_config["in_dir"]
        result = run_experiment(
            overrides=overrides,
            output_name=retrain_output_name,
            retrain=True,
            train=True,
            run_postprocess=True,
        )
        print('Outputs saved to:' + result['out_dir'])
        time.sleep(2)
        return

    if answer2 in {"N", "NO"}:
        print('Predictions from an old simulation. The configuration is uploaded...')
        existing_output_name = Path(model_config['out_dir']).name.rstrip("/\\")
        overrides = dict(model_config)
        overrides["output_root"] = str(get_default_output_root())
        overrides["output_name"] = existing_output_name
        overrides["in_dir"] = model_config["in_dir"]
        result = run_experiment(
            overrides=overrides,
            output_name=existing_output_name,
            retrain=False,
            train=False,
            run_postprocess=True,
        )
        print('Outputs saved to:' + result['out_dir'])
        time.sleep(2)
        return

    raise ValueError("Invalid response for retrain prompt.")


if __name__ == "__main__":
    main()
