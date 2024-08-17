from dataclasses import dataclass, field
import os
from typing import Tuple, Dict


@dataclass(frozen=True)
class Config:
    # Data paths
    data_path: str = "data/raw"
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    model_dir: str = "models"
    log_dir: str = "logs"

    # Raw data filenames
    appearances_file: str = "appearances.csv"
    games_file: str = "games.csv"
    players_file: str = "players.csv"
    clubs_file: str = "clubs.csv"
    club_games_file: str = "club_games.csv"
    game_events_file: str = "game_events.csv"
    game_lineups_file: str = "game_lineups.csv"
    transfers_file: str = "transfers.csv"

    # Data processing
    target_column: str = "market_value_in_eur"

    # Selected features for preprocessing
    selected_features: Tuple[str, ...] = (
        "age",
        "height_in_cm",
        "position",
        "foot",
        "highest_market_value_in_eur",
        "goals",
        "assists",
        "minutes_played",
        "yellow_cards",
        "red_cards",
        "total_market_value",
        "home_club_position",
        "away_club_position",
        "foreigners_number",
        "foreigners_percentage",
        "transfer_fee",
        "transfer_date",
        "goals_per_game",  # Aggiunto qui
        "home_club_goals",
        "away_club_goals",
    )

    numeric_features: Tuple[str, ...] = (
        "age",
        "height_in_cm",
        "goals_per_game",
        "goals",
        "assists",
        "minutes_played",
        "yellow_cards",
        "red_cards",
        "total_market_value",
        "foreigners_number",
        "foreigners_percentage",
        "transfer_fee",
    )

    # Model parameters
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 1

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    test_size: float = 0.2
    random_state: int = 42

    # Evaluation metrics
    metrics: Tuple[str, ...] = ("mse", "rmse", "r2")

    # Derived attributes
    processed_x_path: str = field(init=False)
    processed_y_path: str = field(init=False)
    model_path: str = field(init=False)

    def __post_init__(self):
        # Set paths that depend on other attributes
        object.__setattr__(
            self,
            "processed_x_path",
            os.path.join(self.processed_data_dir, "X_processed.csv"),
        )
        object.__setattr__(
            self,
            "processed_y_path",
            os.path.join(self.processed_data_dir, "y_processed.csv"),
        )
        object.__setattr__(
            self, "model_path", os.path.join(self.model_dir, "lstm_model.pth")
        )

        # Ensure directories exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def raw_data_files(self) -> Dict[str, str]:
        return {
            "appearances": os.path.join(self.raw_data_dir, self.appearances_file),
            "games": os.path.join(self.raw_data_dir, self.games_file),
            "players": os.path.join(self.raw_data_dir, self.players_file),
            "clubs": os.path.join(self.raw_data_dir, self.clubs_file),
            "club_games": os.path.join(self.raw_data_dir, self.club_games_file),
            "game_events": os.path.join(self.raw_data_dir, self.game_events_file),
            "game_lineups": os.path.join(self.raw_data_dir, self.game_lineups_file),
            "transfers": os.path.join(self.raw_data_dir, self.transfers_file),
        }

    def with_input_size(self, input_size: int) -> "Config":
        return Config(
            data_path=self.data_path,
            raw_data_dir=self.raw_data_dir,
            processed_data_dir=self.processed_data_dir,
            model_dir=self.model_dir,
            log_dir=self.log_dir,
            appearances_file=self.appearances_file,
            games_file=self.games_file,
            players_file=self.players_file,
            clubs_file=self.clubs_file,
            club_games_file=self.club_games_file,
            game_events_file=self.game_events_file,
            game_lineups_file=self.game_lineups_file,
            transfers_file=self.transfers_file,
            target_column=self.target_column,
            selected_features=self.selected_features,
            numeric_features=self.numeric_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            test_size=self.test_size,
            random_state=self.random_state,
            metrics=self.metrics,
            input_size=input_size,
        )
