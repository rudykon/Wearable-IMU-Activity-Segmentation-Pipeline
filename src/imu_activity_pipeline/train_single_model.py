"""Single-model training entry point for a selected scale and seed.

Purpose:
    Retrains one configured model without launching the full multi-scale, multi-
    seed training suite.
Inputs:
    CLI arguments select the window suffix (`3s`, `5s`, or `8s`), random seed,
    and optional epoch/early-stopping overrides.
Outputs:
    Writes the selected checkpoint and any required normalization/cache files to
    the configured model directory.
"""
import os, sys, pickle, argparse
import numpy as np
from sklearn.model_selection import train_test_split
from . import config as cfg
from .sensor_data_processing import normalize_imu
from . import train as train_module

def main():
    """Train one model for a selected window suffix and random seed."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', required=True, choices=['3s', '5s', '8s'])
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--min-epochs-before-early-stop', type=int, default=None)
    args = parser.parse_args()

    if args.epochs is not None:
        os.environ['NUM_EPOCHS_STAGE2'] = str(args.epochs)
        cfg.NUM_EPOCHS_STAGE2 = args.epochs
        train_module.NUM_EPOCHS_STAGE2 = args.epochs
    if args.patience is not None:
        os.environ['EARLY_STOPPING_PATIENCE'] = str(args.patience)
        cfg.EARLY_STOPPING_PATIENCE = args.patience
        train_module.EARLY_STOPPING_PATIENCE = args.patience
    if args.min_epochs_before_early_stop is not None:
        os.environ['MIN_EPOCHS_BEFORE_EARLY_STOP'] = str(args.min_epochs_before_early_stop)
        cfg.MIN_EPOCHS_BEFORE_EARLY_STOP = args.min_epochs_before_early_stop
        train_module.MIN_EPOCHS_BEFORE_EARLY_STOP = args.min_epochs_before_early_stop

    # Find window config
    for ws_sec, ws_samples, ws_step, ws_suffix in cfg.WINDOW_CONFIGS:
        if ws_suffix == args.suffix:
            break
    else:
        print(f"Unknown suffix: {args.suffix}")
        sys.exit(1)

    model_name = f'combined_model_{args.suffix}_seed{args.seed}'
    cache_file = os.path.join(cfg.MODEL_DIR, f'_cache_{args.suffix}.npz')

    # Load or prepare data
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}...")
        d = np.load(cache_file)
        X_train, y_train, X_val, y_val = d['X_train'], d['y_train'], d['X_val'], d['y_val']
    else:
        print(f"Preparing {args.suffix} data...")
        windows, labels, user_ids = train_module.prepare_training_data(
            window_size=ws_samples, window_step=ws_step
        )
        print("Normalizing...")
        norm_windows, mean, std = normalize_imu(windows)
        norm_params = {'mean': mean, 'std': std}
        with open(os.path.join(cfg.MODEL_DIR, f'norm_params_{args.suffix}.pkl'), 'wb') as f:
            pickle.dump(norm_params, f)
        if args.suffix == "3s":
            with open(os.path.join(cfg.MODEL_DIR, 'norm_params.pkl'), 'wb') as f:
                pickle.dump(norm_params, f)
        del windows

        unique_users = list(set(user_ids))
        train_users, val_users = train_test_split(unique_users, test_size=cfg.VAL_SPLIT, random_state=42)
        user_array = np.array(user_ids)
        X_train = norm_windows[np.isin(user_array, train_users)]
        y_train = labels[np.isin(user_array, train_users)]
        X_val = norm_windows[np.isin(user_array, val_users)]
        y_val = labels[np.isin(user_array, val_users)]

        print(f"Caching to {cache_file}...")
        np.savez(cache_file, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}")

    val_f1 = train_module.train_single_model(
        X_train, y_train, X_val, y_val, args.seed, model_name, window_size=ws_samples
    )
    print(f"\nFINAL: {model_name} val_f1={val_f1:.4f}")

if __name__ == "__main__":
    main()
