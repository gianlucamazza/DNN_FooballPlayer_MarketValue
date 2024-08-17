# Player Market Value Prediction

## Overview

The Player Market Value Prediction project aims to predict the market value of football players based on various features extracted from their game appearances, performance metrics, transfers, and other relevant data. This project leverages machine learning techniques, specifically an LSTM (Long Short-Term Memory) model, to analyze historical data and forecast player values. The predictions can be valuable for clubs, agents, and analysts in making informed decisions in the football market.

## Features

- **Data Preprocessing**: Merging and cleaning data from various sources, including player appearances, game events, transfers, and club statistics.
- **Feature Engineering**: Creating additional features such as age, goals per game, and more to improve the predictive power of the model.
- **Model Training**: Training a machine learning model, specifically an LSTM model, to predict player market values.
- **Evaluation**: Assessing the model's performance using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

## Data Sources

The project utilizes multiple datasets to construct the feature set for model training:

- **Appearances Data**: Records of player appearances in matches, including minutes played, goals, assists, and other performance metrics.
- **Games Data**: Information about the matches, such as date, venue, competing teams, and match outcomes.
- **Players Data**: Personal and career details of the players, including position, height, foot preference, and previous market values.
- **Transfers Data**: Details about player transfers between clubs, including transfer fees and market values at the time of transfer.
- **Clubs Data**: Information about football clubs, including squad size, average age, total market value, and other club-specific metrics.

## Configuration and Kaggle Setup

This project is configured to work with datasets available on [Kaggle](https://www.kaggle.com/). Kaggle is a popular platform for data science competitions, datasets, and notebooks. To use Kaggle datasets in this project, you need to configure your Kaggle API key and place it in the appropriate directory.

### Setting Up Kaggle

1. **Create a Kaggle Account**:
   - If you don't already have a Kaggle account, [sign up here](https://www.kaggle.com/account/login?phase=startSignup).

2. **Generate Kaggle API Key**:
   - Log in to your Kaggle account.
   - Go to the "Account" tab in your profile settings.
   - Scroll down to the "API" section and click "Create New API Token". This will download a `kaggle.json` file containing your API key.

3. **Place API Key in Correct Directory**:
   - Place the downloaded `kaggle.json` file in the following directory:
     - **Windows**: `C:\Users\<Your-Username>\.kaggle\`
     - **Mac/Linux**: `/Users/<Your-Username>/.kaggle/`

   - Ensure that the file has the correct permissions:
     - **Mac/Linux**: Run `chmod 600 ~/.kaggle/kaggle.json` to set the appropriate permissions.

4. **Install the Kaggle API Python Package**:
   - You can install the Kaggle API using pip:
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
    │   └── trainer.py
    └── utils
        ├── __init__.py
        └── logger.py
```

- **`data/`**: Contains raw and processed data.
- **`logs/`**: Directory for storing logs generated during processing and training.
- **`models/`**: Directory for saving trained models.
- **`src/`**: Contains the main codebase, including data processing, model training, and evaluation scripts.
  - **`config/`**: Configuration files for managing paths, model parameters, and other settings.
  - **`data/`**: Modules for loading and preprocessing data.
  - **`evaluation/`**: Scripts for evaluating the performance of the trained models.
  - **`models/`**: Definition and training of the machine learning models.
  - **`processing/`**: Preprocessing logic and feature engineering.
  - **`training/`**: Training logic for the models.
  - **`utils/`**: Utility functions such as logging and configuration management.

## How to Use

### Prerequisites

- Python 3.8 or higher
- Required Python libraries listed in `requirements.txt`

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/gianlucamazza/player-market-value-prediction.git
   cd player-market-value-prediction
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ``` 

3. Ensure your data is placed in the correct directories as specified in the configuration files.

### Running the Pipeline

You can run the entire data processing, model training, and evaluation pipeline using the following command:

```bash
python -m src.main
```

This will load the data, preprocess it, train the model, and evaluate the results.

### Configuration

The project uses a configuration file (`config.py`) to manage paths, feature selection, model parameters, and other settings. You can modify the configuration to suit your specific needs.

## Future Work

- **Model Optimization**: Experiment with different machine learning models and hyperparameters to improve prediction accuracy.
- **Feature Expansion**: Incorporate additional features such as player injuries, contract details, and match importance.
- **Real-time Predictions**: Develop a system for making real-time market value predictions based on live match data.
