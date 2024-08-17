import optuna
import json
from sklearn.model_selection import train_test_split
from src.models.lstm_model import LSTMModel
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


def objective(trial, X, y, device=None):
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
        f"learning_rate={learning_rate}, batch_size={batch_size}, "
        f"hidden_size={hidden_size}, num_layers={num_layers}, weight_decay={weight_decay}, dropout={dropout}"
    )

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define the model
    model = LSTMModel(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
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
        device=device,
    )

    # Evaluate the model
    model.load_state_dict(model_state)
    val_mse, val_rmse, val_r2 = evaluate_model(model, X_val, y_val, device)

    logger.info(f"Trial {trial.number} completed with validation MSE: {val_mse:.4f}")

    # Return the validation loss as the objective value
    return val_mse


def optimize_hyperparameters(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials=50,
    device=None,
    hyperparams_path: str = "best_hyperparams.json",
) -> dict:
    """
    Optimize hyperparameters using Optuna.

    Args:
        X: Features dataframe.
        y: Target series.
        n_trials: Number of trials to run for optimization.
        device: The device (CPU/GPU) to run the training on.
        hyperparams_path: Path to save the best hyperparameters.

    Returns:
        The best set of hyperparameters found during optimization.
    """

    # Define Optuna study
    study = optuna.create_study(direction="minimize")

    # Optimize the objective function
    study.optimize(lambda trial: objective(trial, X, y, device), n_trials=n_trials)

    best_hyperparams = study.best_params
    logger.info(f"Best hyperparameters: {best_hyperparams}")
    logger.info(f"Best validation MSE: {study.best_value:.4f}")

    save_best_hyperparams(best_hyperparams, hyperparams_path)

    return study.best_params
