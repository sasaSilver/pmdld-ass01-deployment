from ..config import config
from .model import model
from .dataset import get_data_loaders
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--train", action="store_true")
argparser.add_argument("--evaluate", action="store_true")
args = argparser.parse_args()


def train():
    train_loader, test_loader = get_data_loaders()
    model.train(train_loader, test_loader)


def evaluate():
    model.load(config.best_model_path)
    train_loader, test_loader = get_data_loaders()
    model.evaluate(test_loader)


if __name__ == "__main__":
    if args.train:
        train()
    elif args.evaluate:
        evaluate()
