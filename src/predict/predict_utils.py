# download player data from the API: localhost:8000/players/{player_id}/profile
import requests
import os
import json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_player_data(player_id: int, output_dir: str) -> None:
    """Download player data from the API."""
    logger.info(f"Downloading player data for player ID: {player_id}")
    url = f"http://localhost:8000/players/{player_id}/profile"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        output_path = os.path.join(output_dir, f"{player_id}.json")
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Player data saved to {output_path}")
    else:
        logger.error(f"Failed to download player data: {response.text}")
        raise ValueError("Failed to download player data")
