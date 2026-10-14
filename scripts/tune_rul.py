import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import mlflow

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir, load_yaml, calculate_metrics
from src.common.splits import get_persistent_split
from src.models.model_service import ModelService

def tune_subset(subset: str = "FD001", n_trials: int = 30, timeout: int = 300):
    root_dir = get_root_dir()
    train_path = root_dir / "data" / "processed" / subset / "train.parquet"
    if not train_path.exists():
        from src.features.pipeline import DataPipeline
        DataPipeline(subset).run()

    train_df = pd.read_parquet(train_path)
    ms = ModelService(subset=subset)
    feature_cols = ms._select_features(train_df)

    # Persistent split: TRAIN engines ONLY
    all_units = [int(u) for u in np.unique(train_df["unit"].values)]
    split = get_persistent_split(subset, all_units=all_units)
    train_units = split["train_units"]
    
    # Assert units used for tuning are strictly a subset of train_units
    assert set(train_units).issubset(set(all_units)), "Train units must come from dataset"
    assert len(set(train_units).intersection(set(split["val_units"]))) == 0, "No val units allowed in tuning"
    assert len(set(train_units).intersection(set(split["lifecycle_holdout_units"]))) == 0, "No holdout units allowed in tuning"

    df_train_only = train_df[train_df["unit"].isin(train_units)].copy().reset_index(drop=True)
    units_arr = df_train_only["unit"].values
    X_all = df_train_only[feature_cols].values
    y_all = df_train_only["rul_capped"].values

    # GroupKFold(k=5, groups=engine) on TRAIN engines only
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X_all, y_all, groups=units_arr))

    # Pre-select one fixed-seed random truncation point per engine (seed=42)
    rng = np.random.RandomState(42)
    truncation_indices = []
    for u in train_units:
        u_indices = df_train_only.index[df_train_only["unit"] == u].tolist()
        if u_indices:
            # pick random truncation cycle index for this engine
            idx = rng.choice(u_indices)
            truncation_indices.append(idx)
    truncation_set = set(truncation_indices)

    # 1. Evaluate baseline default model
    t0_default = time.time()
    default_rmses, default_nasas = [], []
    for tr_idx, val_idx in splits:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        model.fit(X_all[tr_idx], y_all[tr_idx])
        pred_val = np.maximum(0.0, model.predict(X_all[val_idx]))
        m = calculate_metrics(y_all[val_idx], pred_val)
        default_rmses.append(m["rmse"])
        default_nasas.append(m["nasa_score"])
    default_cv_rmse = float(np.mean(default_rmses))
    default_cv_nasa = float(np.mean(default_nasas))
    default_train_sec = round(time.time() - t0_default, 2)

    # 2. Optuna study
    mlflow.set_experiment(f"AETURNUS-Tuning-{subset}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name=f"Optuna_Parent_{subset}") as parent_run:
        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 80, 220, step=20)
            learning_rate = trial.suggest_float("learning_rate", 0.02, 0.12, log=True)
            num_leaves = trial.suggest_int("num_leaves", 15, 63)
            min_child_samples = trial.suggest_int("min_child_samples", 10, 50)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True)

            fold_trunc_rmses = []
            fold_all_rmses = []

            for tr_idx, val_idx in splits:
                model = lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    min_child_samples=min_child_samples,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_all[tr_idx], y_all[tr_idx])
                pred_all = np.maximum(0.0, model.predict(X_all[val_idx]))
                
                # All-window RMSE
                all_rmse = np.sqrt(np.mean((pred_all - y_all[val_idx]) ** 2))
                fold_all_rmses.append(all_rmse)

                # Truncation-point RMSE
                val_trunc_mask = [i for i, original_idx in enumerate(val_idx) if original_idx in truncation_set]
                if val_trunc_mask:
                    trunc_pred = pred_all[val_trunc_mask]
                    trunc_true = y_all[val_idx][val_trunc_mask]
                    trunc_rmse = np.sqrt(np.mean((trunc_pred - trunc_true) ** 2))
                    fold_trunc_rmses.append(trunc_rmse)
                else:
                    fold_trunc_rmses.append(all_rmse)

            mean_trunc_rmse = float(np.mean(fold_trunc_rmses))
            mean_all_rmse = float(np.mean(fold_all_rmses))

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                mlflow.log_metric("cv_truncation_rmse", mean_trunc_rmse)
                mlflow.log_metric("cv_all_window_rmse", mean_all_rmse)

            return mean_trunc_rmse

        t0_tune = time.time()
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        tune_duration_sec = round(time.time() - t0_tune, 2)

    best_params = study.best_params
    best_params["random_state"] = 42

    # Save best parameters to configs/best_params_{subset}.json
    best_params_path = root_dir / "configs" / f"best_params_{subset}.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Evaluate tuned model over 5 folds
    tuned_rmses, tuned_nasas = [], []
    for tr_idx, val_idx in splits:
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=best_params["n_estimators"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            reg_alpha=best_params["reg_alpha"],
            reg_lambda=best_params["reg_lambda"],
            random_state=42,
            verbose=-1
        )
        model.fit(X_all[tr_idx], y_all[tr_idx])
        pred_val = np.maximum(0.0, model.predict(X_all[val_idx]))
        m = calculate_metrics(y_all[val_idx], pred_val)
        tuned_rmses.append(m["rmse"])
        tuned_nasas.append(m["nasa_score"])

    tuned_cv_rmse = float(np.mean(tuned_rmses))
    tuned_cv_nasa = float(np.mean(tuned_nasas))

    tuning_report = {
        "subset": subset,
        "n_trials": n_trials,
        "default_cv_rmse": round(default_cv_rmse, 3),
        "default_cv_nasa": round(default_cv_nasa, 1),
        "tuned_cv_rmse": round(tuned_cv_rmse, 3),
        "tuned_cv_nasa": round(tuned_cv_nasa, 1),
        "best_params": best_params,
        "train_time_sec": tune_duration_sec
    }

    report_path = root_dir / "artifacts" / "reports" / f"tuning_{subset}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(tuning_report, f, indent=2)

    print(f"[{subset}] Optuna tuning complete! Default CV RMSE: {default_cv_rmse:.2f} -> Tuned: {tuned_cv_rmse:.2f}")
    return tuning_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="all")
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    if args.subset.lower() == "all":
        for sub in ["FD001", "FD002", "FD003", "FD004"]:
            print(f"\n================ Tuning {sub} ({args.trials} trials) ================")
            tune_subset(sub, n_trials=args.trials)
    else:
        tune_subset(args.subset, n_trials=args.trials)
