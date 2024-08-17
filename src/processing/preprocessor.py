import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.logger import get_logger
from src.config.config import Config
from typing import Tuple

logger = get_logger(__name__)


def merge_datasets(appearances, games, players, clubs, transfers) -> pd.DataFrame:
    logger.info("Merging datasets: appearances, games, players, clubs, transfers")

    player_games = pd.merge(appearances, games, on="game_id", how="left")
    logger.info(f"Merged appearances and games: {player_games.shape}")

    player_data = pd.merge(player_games, players, on="player_id", how="left")
    logger.info(f"Merged with players: {player_data.shape}")

    player_club_data = pd.merge(
        player_data, clubs, left_on="player_club_id", right_on="club_id", how="left"
    )
    logger.info(f"Merged with clubs: {player_club_data.shape}")

    final_dataset = pd.merge(player_club_data, transfers, on="player_id", how="left")
    logger.info(f"Final merged dataset: {final_dataset.shape}")

    # Fill specific NaNs
    fill_values = {
        "height_in_cm": (
            final_dataset["height_in_cm"].median()
            if "height_in_cm" in final_dataset
            else None
        ),
        "position": "Unknown" if "position" in final_dataset else None,
        "foot": "Unknown" if "foot" in final_dataset else None,
        "transfer_fee": 0 if "transfer_fee" in final_dataset else None,
    }
    final_dataset.fillna(fill_values, inplace=True)
    logger.info(
        f"Filled NaN values for the following columns: {list(fill_values.keys())}"
    )

    # Fill any remaining NaNs
    fill_remaining_nans(final_dataset)

    return final_dataset


def fill_remaining_nans(final_dataset: pd.DataFrame) -> None:
    logger.info("Filling remaining NaN values in the dataset")
    for column in final_dataset.columns:
        if final_dataset[column].isna().any():
            if final_dataset[column].dtype == "object":
                final_dataset.loc[:, column] = final_dataset[column].fillna("Unknown")
            else:
                # Check if the column has any non-NaN values to calculate the median
                if final_dataset[column].notna().any():
                    final_dataset.loc[:, column] = final_dataset[column].fillna(
                        final_dataset[column].median()
                    )
                else:
                    logger.warning(
                        f"Column '{column}' is entirely NaN; filling with an appropriate default value."
                    )
                    if (
                        final_dataset[column].dtype == "float64"
                        or final_dataset[column].dtype == "int64"
                    ):
                        final_dataset.loc[:, column] = final_dataset[column].fillna(0)
                    else:
                        final_dataset.loc[:, column] = final_dataset[column].fillna(
                            "Unknown"
                        )
    logger.info("Remaining NaN values filled.")


def handle_name_columns(final_dataset: pd.DataFrame) -> None:
    logger.info("Handling name columns")
    if "name_x" in final_dataset.columns:
        final_dataset["player_name"] = final_dataset["name_x"]
        final_dataset.drop(columns=["name_x", "name_y"], inplace=True, errors="ignore")
        logger.info("Dropped redundant name columns")


def handle_market_value_columns(final_dataset: pd.DataFrame) -> None:
    logger.info("Handling market value columns")
    if (
        "market_value_in_eur_x" in final_dataset.columns
        or "market_value_in_eur_y" in final_dataset.columns
    ):
        if (
            "market_value_in_eur_x" in final_dataset.columns
            and "market_value_in_eur_y" in final_dataset.columns
        ):
            final_dataset["market_value_in_eur"] = final_dataset[
                "market_value_in_eur_x"
            ].combine_first(final_dataset["market_value_in_eur_y"])
            final_dataset.drop(
                ["market_value_in_eur_x", "market_value_in_eur_y"], axis=1, inplace=True
            )
            logger.info("Combined and cleaned market value columns")
        elif "market_value_in_eur_x" in final_dataset.columns:
            final_dataset.rename(
                columns={"market_value_in_eur_x": "market_value_in_eur"}, inplace=True
            )
            logger.info("Renamed market_value_in_eur_x to market_value_in_eur")
        elif "market_value_in_eur_y" in final_dataset.columns:
            final_dataset.rename(
                columns={"market_value_in_eur_y": "market_value_in_eur"}, inplace=True
            )
            logger.info("Renamed market_value_in_eur_y to market_value_in_eur")

    if "market_value_in_eur" not in final_dataset.columns:
        logger.error("'market_value_in_eur' column not found after processing.")
        raise KeyError("'market_value_in_eur' column not found in dataset.")


def calculate_age(final_dataset: pd.DataFrame) -> None:
    logger.info("Calculating player age")
    if "date_of_birth" in final_dataset.columns and "date_x" in final_dataset.columns:
        # Replace any non-date strings like "Unknown" with NaT (Not a Time) before conversion
        final_dataset["date_of_birth"] = pd.to_datetime(
            final_dataset["date_of_birth"], errors="coerce"
        )
        final_dataset["date_x"] = pd.to_datetime(
            final_dataset["date_x"], errors="coerce"
        )

        final_dataset["age"] = (
            final_dataset["date_x"].dt.year - final_dataset["date_of_birth"].dt.year
        )

        # Fill any remaining NaNs in the age column with the median age
        final_dataset["age"] = final_dataset["age"].fillna(
            final_dataset["age"].median()
        )

        logger.info("Player age calculated")
    else:
        logger.warning(
            "Missing columns for age calculation: 'date_of_birth' or 'date_x'"
        )


def calculate_goals_per_game(final_dataset: pd.DataFrame) -> None:
    logger.info("Calculating goals per game")
    if "goals" in final_dataset.columns and "minutes_played" in final_dataset.columns:
        final_dataset["goals_per_game"] = (
            final_dataset["goals"] / final_dataset["minutes_played"]
        ).fillna(0)
        logger.info("Goals per game calculated")
    else:
        logger.warning(
            "Missing columns for goals_per_game calculation: 'goals' or 'minutes_played'"
        )


def select_and_scale_features(
    final_dataset: pd.DataFrame, config: Config
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Selecting and scaling features")

    if "market_value_in_eur" in final_dataset.columns:
        target = final_dataset["market_value_in_eur"]
        features = final_dataset.drop(columns=["market_value_in_eur"])
    else:
        logger.error(
            "'market_value_in_eur' column not found in final dataset for prediction."
        )
        raise KeyError("'market_value_in_eur' column missing in dataset.")

    selected_features = [
        feature for feature in config.selected_features if feature in features.columns
    ]
    features = features[selected_features].copy()

    logger.info(f"Selected features: {selected_features}")

    non_numeric_columns = features.select_dtypes(include=["object"]).columns.tolist()

    if non_numeric_columns:
        logger.info(f"Non-numeric columns found: {non_numeric_columns}")
        for column in non_numeric_columns:
            logger.info(f"Encoding non-numeric column: {column}")
            le = LabelEncoder()
            features[column] = le.fit_transform(features[column].astype(str))

    if features.select_dtypes(include=["object"]).empty:
        logger.info("All columns are now numeric.")
    else:
        logger.error("Some columns are still non-numeric after encoding.")
        raise ValueError("Non-numeric columns remain in the dataset after encoding.")

    features.fillna(0, inplace=True)
    numeric_features_to_scale = [
        feature for feature in config.numeric_features if feature in features.columns
    ]
    scaler = StandardScaler()
    features[numeric_features_to_scale] = scaler.fit_transform(
        features[numeric_features_to_scale]
    )

    logger.info("Features selected and scaled successfully")
    return features, target


def preprocess_data(
    appearances: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    transfers: pd.DataFrame,
    clubs: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Starting data preprocessing.")
    try:
        final_dataset = merge_datasets(appearances, games, players, clubs, transfers)

        handle_name_columns(final_dataset)
        handle_market_value_columns(final_dataset)
        calculate_age(final_dataset)
        calculate_goals_per_game(final_dataset)

        logger.info(f"Final dataset columns: {final_dataset.columns}")

        features, target = select_and_scale_features(final_dataset, config)

        logger.info("Data preprocessing completed successfully.")
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

    return features, target
