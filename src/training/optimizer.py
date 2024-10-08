import optuna
import json
import os
from sklearn.model_selection import train_test_split
from src.models.dnn_model import DNNModel
from src.training.trainer import train_model, evaluate_model
from src.utils.logger import get_logger
import pandas as pd

logger = get_logger(__name__)


def save_best_hyperparams(best_hyperparams: dict, filepath: str) -> None:
    """Save the best hyperparameters to a JSON file."""
    try:
        with open(filepath, "w") as f:
            json.dump(best_hyperparams, f, indent=4)
        logger.info(f"Best hyperparameters saved to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save hyperparameters: {e}")
        raise


def load_best_hyperparams(filepath: str) -> dict:
    """Load the best hyperparameters from a JSON file."""
    try:
        with open(filepath, "r") as f:
            best_hyperparams = json.load(f)
        logger.info(f"Best hyperparameters loaded from {filepath}")
        return best_hyperparams
    except Exception as e:
        logger.error(f"Failed to load hyperparameters: {e}")
        raise


def objective(trial, X, y, config, device=None):
    """
    Objective function for Optuna hyperparameter optimization.

    Args:
        trial: Optuna trial object for hyperparameter suggestions.
        X: Training features.
        y: Training labels.
        device: The device (CPU/GPU) to run the training on.

    Returns:
        The validation loss after training the model with the suggested hyperparameters.
    """

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_int("hidden_size", 32, 256)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5) if num_layers > 1 else 0.0

    logger.info(
        f"Trial {trial.number}: Hyperparameters: "
        f"learning_rate={learning_rate:.6f}, batch_size={batch_size}, "
        f"hidden_size={hidden_size}, num_layers={num_layers}, "
        f"weight_decay={weight_decay:.6f}, dropout={dropout:.4f}, optimizer={config.optimizer}"
    )

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model
    model = DNNModel(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
        activation_function=config.activation_function,
    )

    # Train the model
    model_state = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        optimizer_name=config.optimizer,
        device=device,
    )

    # Evaluate the model
    model.load_state_dict(model_state)
    val_mse, val_rmse, val_r2 = evaluate_model(model, X_val, y_val, device)

    logger.info(
        f"Trial {trial.number} completed with validation MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}"
    )

    # Return the validation loss as the objective value
    return val_mse


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    study_name: str = "DNN Hyperparameter Optimization",
    n_trials=50,
    config: dict = None,
    device=None,
    hyperparams_path: str = "best_hyperparams.json",
    storage_path: str = "optuna_study.db",
) -> dict:
    """
    Optimize hyperparameters using Optuna.

    Args:
        X: Features dataframe.
        y: Target series.
        study_name: Name for the Optuna study.
        n_trials: Number of trials to run for optimization.
        device: The device (CPU/GPU) to run the training on.
        hyperparams_path: Path to save the best hyperparameters.
        storage_path: Path to the SQLite database for saving the study.

    Returns:
        The best set of hyperparameters found during optimization.
    """

    # Create or load an Optuna study
    if os.path.exists(storage_path):
        logger.info(f"Resuming existing study from {storage_path}")
        storage = f"sqlite:///{storage_path}"
        study = optuna.load_study(study_name=study_name, storage=storage)
    else:
        logger.info(f"Starting a new study: {study_name}")
        storage = f"sqlite:///{storage_path}"
        study = optuna.create_study(
            direction="minimize", study_name=study_name, storage=storage
        )

    logger.info(f"Study {study_name} using storage: {storage_path}")

    # Optimize the objective function
    study.optimize(
        lambda trial: objective(trial, X, y, config, device), n_trials=n_trials
    )

    best_hyperparams = study.best_params
    logger.info(f"Best hyperparameters: {json.dumps(best_hyperparams, indent=4)}")
    logger.info(f"Best validation MSE: {study.best_value:.4f}")

    # Save the best hyperparameters
    save_best_hyperparams(best_hyperparams, hyperparams_path)

    return best_hyperparams
