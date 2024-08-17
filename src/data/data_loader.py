import pandas as pd
from typing import Tuple
from src.utils.logger import get_logger
from src.config.config import Config

logger = get_logger(__name__)


def load_data(
    cfg: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carica i dati dai file CSV specificati nella configurazione.

    Args:
        cfg (Config): Oggetto di configurazione che contiene i percorsi dei file e altre impostazioni.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        DataFrame per appearances, games, players, e player valuations.
    """
    logger.info("Loading data from CSV files.")
    try:
        appearances_df = pd.read_csv(cfg.raw_data_files["appearances"])
        games_df = pd.read_csv(cfg.raw_data_files["games"])
        players_df = pd.read_csv(cfg.raw_data_files["players"])
        clubs_df = pd.read_csv(cfg.raw_data_files["clubs"])
        club_games_df = pd.read_csv(cfg.raw_data_files["club_games"])
        game_events_df = pd.read_csv(cfg.raw_data_files["game_events"])
        game_lineups_df = pd.read_csv(cfg.raw_data_files["game_lineups"])
        transfers_df = pd.read_csv(cfg.raw_data_files["transfers"])
        logger.info("Data successfully loaded.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

    return (
        appearances_df,
        games_df,
        players_df,
        clubs_df,
        club_games_df,
        game_events_df,
        game_lineups_df,
        transfers_df,
    )
