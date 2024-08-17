import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_model(model, X_test, y_test):
    logger.info("Starting model evaluation")

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test.values)
        y_pred = model(X_test_tensor).numpy()
        y_true = y_test.values

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Root Mean Squared Error: {rmse:.4f}")
        logger.info(f"R2 Score: {r2:.4f}")

    logger.info("Evaluation completed")
    return mse, rmse, r2
