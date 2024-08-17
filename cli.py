import argparse
import sys
from src.config.config import Config
from src.main import main as train_main
from src.predict.predict import main as predict_main
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train():
    """Handler for the training process."""
    cfg = Config()  # Create a Config instance
    logger.info("Starting training process...")
    train_main(cfg)


def predict(args):
    """Handler for the prediction process."""
    cfg = Config()  # Create a Config instance
    logger.info(f"Starting prediction for player ID {args.player_id}...")
    predict_main(args.player_id, args.output, cfg)


def parse_args():
    """Parses the command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI for managing and executing the football player market value prediction project."
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Train command
    parser_train = subparsers.add_parser(
        "train", help="Train the model with the provided configuration."
    )
    parser_train.set_defaults(func=train)

    # Predict command
    parser_predict = subparsers.add_parser(
        "predict", help="Predict market values using the trained model."
    )
    parser_predict.add_argument(
        "player_id",
        type=int,
        help="The ID of the player to predict the market value for.",
    )
    parser_predict.add_argument(
        "output", type=str, help="Path to save the prediction output."
    )
    parser_predict.set_defaults(func=predict)

    return parser.parse_args()


def main():
    args = parse_args()
    if not args.command:
        print("No command provided. Use --help for more information.")
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
