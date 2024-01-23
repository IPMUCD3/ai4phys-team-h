import numpy as np

import torch
from torchvision.transforms import v2
from torch.utils.data import Dataset

import matplotlib.pyplot as plt


def show_imgs_side_by_side(imgs1, imgs2, ind=4):

    # Create subplots with 5 rows and 6 columns (3 columns for each set of images)
    fig, axes = plt.subplots(5, 6, figsize=(8, 10))

    for i in range(15):
        istd1 = np.std(imgs1[ind, i, :, :])
        istd2 = np.std(imgs2[ind, i, :, :])

        # Determine the row and column for the current image in the subplot grid
        row = i // 3
        col1 = i % 3
        col2 = col1 + 3  # Shift the column index for the second set of images

        # Plot the image from imgs1
        ax1 = axes[row, col1]
        ax1.imshow(imgs1[ind, i, :, :], vmin=-3 * istd1, vmax=3 * istd1)
        ax1.set_title(f"Index {i} (imgs1)", fontsize=6)
        ax1.axis("off")

        # Plot the image from imgs2
        ax2 = axes[row, col2]
        ax2.imshow(imgs2[ind, i, :, :], vmin=-3 * istd2, vmax=3 * istd2)
        ax2.set_title(f"Index {i} (imgs2)", fontsize=6)
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


class ImageToImageDataset(Dataset):
    def __init__(self, imgs1_path, imgs2_path, crop_size=None, eps=1e-6, dtype=np.float32):

        self.eps = eps

        if dtype is None:
            dtype = np.float32

        imgs1 = torch.from_numpy(np.load(imgs1_path).astype(dtype).reshape(1000, 15, 256, 256))
        imgs2 = torch.from_numpy(np.load(imgs2_path).astype(dtype).reshape(1000, 15, 256, 256))

        self.imgs1 = imgs1[:, 4, :, :]
        self.imgs2 = imgs2[:, 4, :, :]

        if crop_size is not None:
            self.imgs1 = v2.functional.center_crop(self.imgs1, crop_size)
            self.imgs2 = v2.functional.center_crop(self.imgs2, crop_size)

    def transform(self, img):
        mean = img.mean(dims=(-2, -1), keepdims=True)
        var = img.var(dims=(-2, -1), keepdims=True)
        return (img - mean) / torch.sqrt(var + self.eps)

    def __len__(self):
        return len(self.imgs1)

    def __getitem__(self, idx):
        image1 = self.transform(self.imgs1[idx])
        image2 = self.transform(self.imgs2[idx])
        return image1, image2
