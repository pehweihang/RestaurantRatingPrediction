import argparse

import torch.optim as optim
from torch.utils.data import DataLoader

from data import RestaurantDataset
from model import DeepFM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("test_path")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("-bs", "--batch_size", default=8, type=int)
    parser.add_argument("-g", "--gpu", default=True, type=bool)
    return parser.parse_args()


def main(args):
    train_data = RestaurantDataset(args.train_path)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    test_data = RestaurantDataset(args.test_path, train=False)
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    model = DeepFM(train_data.feature_sizes, use_cuda=args.gpu)
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0
    )
    model.fit(
        train_loader,
        test_loader,
        optimizer=optimizer,
        epochs=args.epochs,
        verbose=True,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
