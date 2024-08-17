import os
import subprocess
import zipfile
from typing import Tuple, Optional
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.model_selection import train_test_split
from functools import lru_cache

from src.data.data_loader import load_data
from src.processing.preprocessor import preprocess_data
from src.models.lstm_model import LSTMModel
from src.training.optimizer import optimize_hyperparameters
from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model
from src.utils.logger import get_logger
from src.config.config import Config

logger = get_logger(__name__)


def download_kaggle_dataset(dataset_name: str, output_path: str) -> None:
    """
    Download a dataset from Kaggle and extract its contents.

    Args:
        dataset_name (str): Kaggle dataset name
        output_path (str): Path to save the dataset

    Raises:
        Exception: If an error occurs during the download or extraction
    """
    try:
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Downloading Kaggle dataset: {dataset_name}")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", output_path],
            check=True,
        )

        zip_file = next(f for f in os.listdir(output_path) if f.endswith(".zip"))
        zip_path = os.path.join(output_path, zip_file)

        logger.info(f"Extracting dataset from: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_path)

        os.remove(zip_path)
        logger.info("Dataset downloaded and extracted successfully")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


@lru_cache(maxsize=None)
def load_processed_data(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load processed data if it exists, with caching.

    Args:
        cfg (Config): Configuration object

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed feature matrix and target vector
    """
    try:
        x_parquet_path = cfg.processed_x_path.replace(".csv", ".parquet")
        y_parquet_path = cfg.processed_y_path.replace(".csv", ".parquet")

        if os.path.exists(x_parquet_path) and os.path.exists(y_parquet_path):
            logger.info(
                f"Loading processed data from Parquet files: {x_parquet_path}, {y_parquet_path}"
            )
            X = pq.read_table(x_parquet_path).to_pandas()
            y = pq.read_table(y_parquet_path).to_pandas().squeeze("columns")
        else:
            logger.info(
                f"Loading processed data from CSV files: {cfg.processed_x_path}, {cfg.processed_y_path}"
            )
            X = pd.read_csv(cfg.processed_x_path, low_memory=False)
            y = pd.read_csv(cfg.processed_y_path).squeeze("columns")

            logger.info("Saving data as Parquet for future use.")
            X.to_parquet(x_parquet_path, index=False)
            y.to_frame().to_parquet(y_parquet_path, index=False)

        logger.info("Processed data loaded successfully.")
        return X, y
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        raise


def save_processed_data(X: pd.DataFrame, y: pd.Series, cfg: Config) -> None:
    """
    Save processed data to disk in both CSV and Parquet formats.

    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        cfg (Config): Configuration object
    """
    try:
        logger.info(
            f"Saving processed data to CSV files: {cfg.processed_x_path}, {cfg.processed_y_path}"
        )
        X.to_csv(cfg.processed_x_path, index=False)
        y.to_csv(cfg.processed_y_path, index=False)

        x_parquet_path = cfg.processed_x_path.replace(".csv", ".parquet")
        y_parquet_path = cfg.processed_y_path.replace(".csv", ".parquet")

        logger.info(
            f"Saving processed data to Parquet files: {x_parquet_path}, {y_parquet_path}"
        )
        X.to_parquet(x_parquet_path, index=False)
        y.to_frame().to_parquet(y_parquet_path, index=False)

        logger.info("Processed data saved successfully in CSV and Parquet formats.")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise


def train_and_save_model(
    X: pd.DataFrame, y: pd.Series, cfg: Config, hyperparams: dict
) -> None:
    """Train a model using the provided data and save it to disk with optimized hyperparameters."""
    logger.info(
        f"Starting model training with data shapes - X: {X.shape}, y: {y.shape}"
    )
    try:
        model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=hyperparams.get("hidden_size", cfg.hidden_size),
            num_layers=hyperparams.get("num_layers", cfg.num_layers),
            output_size=cfg.output_size,
            dropout=hyperparams.get("dropout", 0.0),
        )
        logger.info(
            f"Training model with configuration: {cfg} and hyperparameters: {hyperparams}"
        )
        trained_model = train_model(
            model,
            X,
            y,
            epochs=cfg.epochs,
            batch_size=hyperparams.get("batch_size", cfg.batch_size),
            learning_rate=hyperparams.get("learning_rate", cfg.learning_rate),
            weight_decay=hyperparams.get("weight_decay", 0.0),
        )
        logger.info(f"Saving trained model to: {cfg.model_path}")
        torch.save(trained_model.state_dict(), cfg.model_path)
        logger.info("Model training completed and saved successfully.")
    except Exception as e:
        logger.error(f"Error in model training pipeline: {e}")
        raise


def evaluate_saved_model(
    X: pd.DataFrame, y: pd.Series, cfg: Config
) -> Tuple[float, float, float]:
    """Evaluate a saved model using the provided data."""
    logger.info(
        f"Starting model evaluation with data shapes - X: {X.shape}, y: {y.shape}"
    )
    try:
        model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            output_size=cfg.output_size,
        )
        logger.info(f"Loading model state from: {cfg.model_path}")
        model.load_state_dict(torch.load(cfg.model_path))
        model.eval()
        mse, rmse, r2 = evaluate_model(model, X, y)
        logger.info(
            f"Model evaluation completed - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}"
        )
        return mse, rmse, r2
    except Exception as e:
        logger.error(f"Error in model evaluation pipeline: {e}")
        raise


def process_data(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """Process raw data into feature matrix and target vector."""
    try:
        logger.info("Loading raw data for processing.")
        (
            appearances_df,
            games_df,
            players_df,
            clubs_df,
            club_games_df,
            game_events_df,
            game_lineups_df,
            transfers_df,
        ) = load_data(cfg)

        logger.info("Preprocessing data to extract features and target.")
        X, y = preprocess_data(
            appearances_df,
            games_df,
            players_df,
            transfers_df,
            clubs_df,
            cfg,
        )

        logger.info(
            f"Data processing completed - Features shape: {X.shape}, Target shape: {y.shape}"
        )
        return X, y
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise


def main(cfg: Config) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
    """Main function to orchestrate data processing, hyperparameter optimization, model training, and evaluation."""
    logger.info("Starting main pipeline.")

    try:
        if not os.path.exists(cfg.data_path) or not os.listdir(cfg.data_path):
            logger.info(
                f"Raw data not found in {cfg.data_path}. Downloading from Kaggle."
            )
            download_kaggle_dataset("davidcariboo/player-scores", cfg.data_path)

        if os.path.exists(cfg.processed_x_path) and os.path.exists(
            cfg.processed_y_path
        ):
            logger.info("Processed data found. Loading from files.")
            X, y = load_processed_data(cfg)
        else:
            logger.info("Processed data not found. Starting full data processing.")
            X, y = process_data(cfg)
            save_processed_data(X, y, cfg)

        logger.info("Splitting data into training and testing sets.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state
        )

        best_hyperparams = None
        if cfg.optimize_hyperparameters:
            logger.info("Optimizing hyperparameters with Optuna.")
            best_hyperparams = optimize_hyperparameters(
                X_train,
                y_train,
                n_trials=cfg.n_trials,  # Make n_trials configurable in cfg
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        if not best_hyperparams:
            logger.info("Using default hyperparameters from config.")
            best_hyperparams = {
                "hidden_size": cfg.hidden_size,
                "num_layers": cfg.num_layers,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "dropout": 0.0,
                "weight_decay": 0.0,
            }

        logger.info("Training and saving the model with selected hyperparameters.")
        train_and_save_model(X_train, y_train, cfg, best_hyperparams)

        logger.info("Evaluating the trained model.")
        mse, rmse, r2 = evaluate_saved_model(X_test, y_test, cfg)

        logger.info(
            f"Model Performance - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}"
        )
        logger.info("Main pipeline completed successfully.")
        return X, y
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        return None


if __name__ == "__main__":
    cfg = Config()
    result = main(cfg)
    if result:
        X, y = result
        logger.info(f"Final dataset shape - X: {X.shape}, y: {y.shape}")
    else:
        logger.error("Main pipeline failed to complete successfully.")
