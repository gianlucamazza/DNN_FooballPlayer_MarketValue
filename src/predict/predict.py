import torch
import pandas as pd
from src.models.dnn_model import DNNModel
from src.processing.preprocessor import preprocess_data
from src.utils.logger import get_logger
from src.config.config import Config
from src.predict.predict_utils import download_player_data

logger = get_logger(__name__)


def load_model(cfg: Config) -> DNNModel:
    """Load the trained DNN model from disk."""
    logger.info(f"Loading model from {cfg.model_path}")
    try:
        model = DNNModel(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            output_size=cfg.output_size,
            dropout=cfg.dropout,
        )
        model.load_state_dict(
            torch.load(cfg.model_path, map_location=torch.device("cpu"))
        )
        model.eval()  # Set model to evaluation mode
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_new_data(player_id: int, cfg: Config) -> pd.DataFrame:
    """Download and preprocess the new input data for prediction."""
    logger.info(f"Downloading and preprocessing data for player ID: {player_id}")
    try:
        download_player_data(player_id, cfg.data_path)
        input_data_path = f"{cfg.data_path}/{player_id}.json"
        input_data = pd.read_json(input_data_path)

        # Assuming preprocess_data returns the features and target, but only features are needed for prediction
        X, _ = preprocess_data(
            appearances=input_data["appearances"],
            games=input_data["games"],
            players=input_data["players"],
            transfers=input_data["transfers"],
            clubs=input_data["clubs"],
            config=cfg,
        )
        logger.info(f"Data preprocessed successfully with shape: {X.shape}")
        return X
    except Exception as e:
        logger.error(f"Failed to preprocess data: {e}")
        raise


def predict(model: DNNModel, X: pd.DataFrame) -> pd.Series:
    """Make predictions using the trained model."""
    logger.info("Making predictions on input data")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        X_tensor = torch.FloatTensor(X.values).to(device)

        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()

        logger.info("Predictions made successfully.")
        return pd.Series(predictions, index=X.index)
    except Exception as e:
        logger.error(f"Failed to make predictions: {e}")
        raise


def main(player_id: int, output_data_path: str, cfg: Config):
    """Main function to load the model, preprocess the data, and make predictions."""
    logger.info("Starting prediction process")
    try:
        # Preprocess the new data
        X = preprocess_new_data(player_id, cfg)

        # Load the model
        model = load_model(cfg)

        # Make predictions
        predictions = predict(model, X)

        # Save predictions
        logger.info(f"Saving predictions to {output_data_path}")
        predictions.to_csv(output_data_path, index=False)
        logger.info("Predictions saved successfully.")
    except Exception as e:
        logger.error(f"Prediction process failed: {e}")
