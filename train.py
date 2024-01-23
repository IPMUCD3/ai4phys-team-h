import argparse

import torch

from src.unet import UNet
from src.loader import ImageToImageDataset


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--save_path", type=str, default="./model.pt")
    args = parser.parse_args()

    main(args)
