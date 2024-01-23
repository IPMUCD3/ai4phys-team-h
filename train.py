import argparse
import torch
from torch.utils.data import DataLoader
from src.loader import ImageToImageDataset
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np
from src.unet import UNet
from src.loader import ImageToImageDataset
from src.encoder import VAE_Encoder
from src.decoder import VAE_Decoder
from PIL import Image, ImageChops

def make_dataloaders(path, config, num_workers=2, shuffle=True, limit=None):
    train_dataset, val_dataset = ImageToImageDataset
    train_dl = DataLoader(train_dataset,
                          batch_size=config["batch_size"],
                          num_workers=num_workers,
                          pin_memory=config["pin_memory"],
                          persistent_workers=True,
                          shuffle=shuffle)
    val_dl = DataLoader(val_dataset,
                        batch_size=config["batch_size"],
                        num_workers=num_workers,
                        pin_memory=config["pin_memory"],
                        persistent_workers=True,
                        shuffle=shuffle)
    return train_dl, val_dl

def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(args)

    train_dl, val_dl = make_dataloaders(num_workers=2, limit=35000)

    #TODO remove 
    # args.ckpt = "/home/ec2-user/Color-diffusion/Color_diffusion_v2/23l96nt1/checkpoints/last.ckpt"
    args.ckpt = "./checkpoints/last.ckpt"

    
    encoder = VAE_Encoder
    unet = UNet
    if args.ckpt is not None:
        print(f"Resuming training from checkpoint: {args.ckpt}")
        model = Diffusion.load_from_checkpoint(
            args.ckpt, 
            strict=True, 
            unet=unet, 
            encoder=encoder, 
            train_dl=train_dl, 
            val_dl=val_dl)
    else:
        model = Diffusion(unet=unet,
                               encoder=encoder, 
                               train_dl=train_dl,
                               val_dl=val_dl)
    ckpt_callback = ModelCheckpoint(every_n_train_steps=300, save_top_k=2, save_last=True, monitor="val_loss")

    trainer = pl.Trainer(max_epochs=5,
                        logger=None, 
                        accelerator=device,
                        num_sanity_val_steps=1,
                        devices= "auto",
                        log_every_n_steps=3,
                        callbacks=[ckpt_callback],
                        profiler=None
                        )
    trainer.fit(model, train_dl, val_dl)

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