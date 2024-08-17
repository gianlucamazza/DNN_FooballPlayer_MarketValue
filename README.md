# Player Market Value Prediction

## Overview

The Player Market Value Prediction project predicts football players' market values using various features derived from their game performances, transfers, and other relevant data. This project employs machine learning techniques, specifically an LSTM (Long Short-Term Memory) model, to analyze historical data and forecast player values. These predictions can aid clubs, agents, and analysts in making informed decisions within the football market.

## Features

- **Data Preprocessing**: Merges and cleans data from multiple sources, including player appearances, game events, transfers, and club statistics.
- **Feature Engineering**: Generates additional features such as age, goals per game, and other metrics to enhance the model’s predictive accuracy.
- **Model Training**: Trains an LSTM model to predict player market values.
- **Evaluation**: Assesses model performance using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

## Data Sources

This project utilizes several datasets to construct the feature set for model training:

- **Appearances Data**: Records of player appearances in matches, including metrics like minutes played, goals, and assists.
- **Games Data**: Information about matches, such as date, venue, teams, and outcomes.
- **Players Data**: Personal and career details of players, including position, height, and market values.
- **Transfers Data**: Details about player transfers between clubs, including transfer fees and market values at the time of the transfer.
- **Clubs Data**: Information about football clubs, including squad size, average age, and total market value.

## Configuration and Kaggle Setup

If the dataset is not available locally, the pipeline will automatically download it from [Kaggle](https://www.kaggle.com/). To enable this, you need to configure your Kaggle API key.

### Setting Up Kaggle

1. **Create a Kaggle Account**:
   - If you don’t have a Kaggle account, [sign up here](https://www.kaggle.com/account/login?phase=startSignup).

2. **Generate a Kaggle API Key**:
   - Log in to your Kaggle account.
   - Navigate to the "Account" section in your profile settings.
   - Scroll to the "API" section and click "Create New API Token" to download the `kaggle.json` file.

3. **Place the API Key**:
   - Move the `kaggle.json` file to the appropriate directory:
     - **Windows**: `C:\Users\<Your-Username>\.kaggle\`
     - **Mac/Linux**: `/Users/<Your-Username>/.kaggle/`
   - Ensure correct file permissions:
     - **Mac/Linux**: Run `chmod 600 ~/.kaggle/kaggle.json`.

4. **Install the Kaggle API Package**:
   - Install the required package via pip:
     ```bash
     pip install kaggle
     ```

## Project Structure

```plaintext
.
├── README.md
├── data
│   ├── processed
│   └── raw
├── logs
├── models
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── config
    │   ├── __init__.py
    │   └── config.py
    ├── data
    │   ├── __init__.py
    │   └── data_loader.py
    ├── evaluation
    │   ├── __init__.py
    │   └── evaluator.py
    ├── main.py
    ├── models
    │   ├── __init__.py
    │   └── lstm_model.py
    ├── processing
    │   ├── __init__.py
    │   └── preprocessor.py
    ├── training
    │   ├── __init__.py
    │   ├── optimizer.py
    │   └── trainer.py
    └── utils
        ├── __init__.py
        └── logger.py
```

- **`data/`**: Stores raw and processed datasets.
- **`logs/`**: Contains logs generated during data processing and model training.
- **`models/`**: Directory for saving trained models.
- **`src/`**: The main codebase, including modules for data processing, model training, and evaluation.
  - **`config/`**: Configuration files managing paths, model parameters, and settings.
  - **`data/`**: Modules for data loading and preprocessing.
  - **`evaluation/`**: Scripts for evaluating model performance.
  - **`models/`**: Contains model definitions and training scripts.
  - **`processing/`**: Includes preprocessing logic and feature engineering.
  - **`training/`**: Logic for training models, including hyperparameter optimization.
  - **`utils/`**: Utility functions like logging and configuration management.

## How to Use

### Prerequisites

- Python 3.8 or higher
- Required Python libraries listed in `requirements.txt`

### Setup

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/gianlucamazza/player-market-value-prediction.git
   cd player-market-value-prediction
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

You can run the entire pipeline—data processing, model training, and evaluation—with the following command:

```bash
python -m src.main
```

If the data isn't found locally, the pipeline will automatically download the necessary datasets from Kaggle.

### Configuration

The project uses a configuration file (`config.py`) to manage paths, feature selection, model parameters, and other settings. You can customize the configuration to fit your specific needs.

## Future Work

- **Model Optimization**: Explore different models and hyperparameters to enhance prediction accuracy.
- **Feature Expansion**: Integrate additional features like player injuries, contract details, and match importance.
- **Real-time Predictions**: Develop a system for real-time player market value predictions based on live match data.
